"""Adapter that wires the kept library code (kg_layer, classical_baseline,
quantum_layer) into the new service Orchestrator.

This is intentionally isolated in its own module: tests don't need it (they
inject fake orchestrators), and importing the kept library can fail in a
minimal environment. Anything fragile lives here, not in core service modules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .orchestration import (
    EntityResolver,
    ExecutionRouter,
    Orchestrator,
    PredictorBundle,
    QuantumExecutorProtocol,
)
from .schemas import ManifestChain, QuantumMode
from .settings import Settings

log = logging.getLogger(__name__)


class _NullQuantumExecutor:
    """Quantum executor stand-in. Flags can be set per instance to reflect
    whatever path is actually live (synthetic, simulator, or IBM)."""
    _ibm_initialized = False
    _ibm_reachable = False
    _simulator_available = False
    _gpu_simulator_available = False
    _is_ibm_hardware = False
    _last_calibration_ts = None

    def current_mode(self) -> QuantumMode:
        if self._is_ibm_hardware and self._ibm_reachable:
            return QuantumMode.IBM_HERON
        if self._gpu_simulator_available:
            return QuantumMode.GPU_SIMULATOR
        if self._simulator_available:
            return QuantumMode.SIMULATOR
        return QuantumMode.UNAVAILABLE

    def is_ibm_hardware_mode(self) -> bool:
        return self._is_ibm_hardware


def _try_load_classical(settings: Settings, chain: Optional[ManifestChain]):
    """Returns (predict_fn, scaler_n_features) or (None, None).

    scaler_n_features tells the embedder which feature format the persisted
    pipeline expects. None when no scaler is found.

    Manifest-only lookup: if no active classical chain or its files are
    missing, returns (None, None) and lets the caller decide. The legacy
    models/ fallback was removed to enforce a single source of truth — run
    `python -m service.scripts.synthesize_manifest_chain` first to register
    on-disk artifacts as a manifest.
    """
    import joblib

    if chain is None:
        log.warning(
            "no active classical manifest chain; predictions disabled. "
            "Run service.scripts.synthesize_manifest_chain to register a chain.",
        )
        return None, None

    model_path = settings.artifacts_dir / "runs" / chain.model_id / "model.joblib"
    if not model_path.exists():
        log.warning(
            "manifest %s active but model.joblib missing at %s",
            chain.model_id, model_path,
        )
        return None, None

    log.info("loading classical model from %s", model_path)
    model = joblib.load(model_path)

    # Scaler lives next to the model, OR in the parent feature_pipeline dir.
    scaler_path = model_path.parent / "scaler.joblib"
    if not scaler_path.exists():
        fp_scaler = (
            settings.artifacts_dir / "runs" / chain.feature_pipeline_id / "scaler.joblib"
        )
        if fp_scaler.exists():
            scaler_path = fp_scaler
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    scaler_n_features = getattr(scaler, "n_features_in_", None) if scaler else None

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X2 = scaler.transform(X) if scaler is not None else X
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X2)
            return np.asarray(proba)[:, 1]
        return np.asarray(model.predict(X2)).astype(float)

    return predict_fn, scaler_n_features


def _try_load_quantum(settings: Settings, quantum_chain: Optional[ManifestChain]):
    """Returns (quantum_predict_fn, quantum_feature_prep, executor).

    quantum_predict_fn and quantum_feature_prep are None unless a quantum
    manifest is active and its weights file is present on disk. The
    quantum_feature_prep loads the model's PCA reducer (if present) so the
    feature shape matches what the VQC was trained on.
    """
    null_exec = _NullQuantumExecutor()
    if quantum_chain is None:
        return None, None, null_exec

    model_dir = settings.artifacts_dir / "runs" / quantum_chain.model_id
    weights_path = model_dir / "quantum_weights.npz"
    config_path = model_dir / "quantum_config.json"

    if not weights_path.exists():
        log.warning("quantum manifest active but weights missing at %s", weights_path)
        return None, None, null_exec

    try:
        with weights_path.open("rb") as f:
            payload = np.load(f, allow_pickle=False)
            weight_vec = np.asarray(payload["weights"], dtype=np.float64)
    except Exception as e:
        log.warning("failed to load quantum weights from %s: %s", weights_path, e)
        return None, None, null_exec

    qconf: dict = {}
    if config_path.exists():
        try:
            qconf = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("failed to read quantum_config.json: %s", e)

    num_qubits = int(qconf.get("num_qubits", weight_vec.shape[-1]))
    backend = str(qconf.get("backend", "synthetic"))
    is_synthetic = bool(qconf.get("synthetic", backend == "synthetic"))

    # Build a quantum predict_fn. For backend="synthetic" we use a
    # deterministic numpy stand-in (clearly labeled in the manifest); for
    # backend="qiskit" we load a cloudpickled QMLLinkPredictor.
    if backend == "qiskit":
        try:
            predict_fn = _build_qiskit_quantum_predictor(model_dir, qconf)
        except Exception as e:
            log.warning(
                "qiskit predictor build failed (%s); falling back to synthetic stand-in", e
            )
            predict_fn = _build_synthetic_quantum_predictor(weight_vec, num_qubits)
            is_synthetic = True
    else:
        predict_fn = _build_synthetic_quantum_predictor(weight_vec, num_qubits)

    # Feature pipeline: optional PCA reducer that brings the classical 3*d
    # vector down to num_qubits dims. Without a PCA file we slice the leading
    # num_qubits features (deterministic, lossy — synthetic-mode only).
    fp_dir = settings.artifacts_dir / "runs" / quantum_chain.feature_pipeline_id
    pca_path = fp_dir / "pca.joblib"
    pca = None
    if pca_path.exists():
        try:
            import joblib
            pca = joblib.load(pca_path)
        except Exception as e:
            log.warning("failed to load quantum PCA at %s: %s", pca_path, e)

    def quantum_feature_prep_factory(classical_prep):
        def quantum_feature_prep(drug_id: str, disease_id: str) -> np.ndarray:
            X = classical_prep(drug_id, disease_id)
            if pca is not None:
                return pca.transform(X)
            # Stand-in: take the first num_qubits dims.
            return X[:, :num_qubits]
        return quantum_feature_prep

    log.info(
        "quantum manifest %s loaded (backend=%s, num_qubits=%d, synthetic=%s)",
        quantum_chain.model_id, backend, num_qubits, is_synthetic,
    )

    qe = _NullQuantumExecutor()
    qe._simulator_available = True  # synthetic counts as "simulator" for routing
    qe._is_ibm_hardware = False
    return predict_fn, quantum_feature_prep_factory, qe


def _build_synthetic_quantum_predictor(weight_vec: np.ndarray, num_qubits: int):
    """Deterministic stand-in: probability = sigmoid(features · weights[0:num_qubits]).

    This is a placeholder so the manifest-chain → predict_fn wiring can be
    exercised end-to-end before a real VQC is trained. The manifest's
    quantum_config.json should set "synthetic": true so the dashboard can
    surface that this isn't a trained quantum model.
    """
    w = weight_vec.reshape(-1)[:num_qubits].astype(np.float64)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X2 = np.atleast_2d(X)[:, :num_qubits]
        z = X2 @ w
        return 1.0 / (1.0 + np.exp(-z))

    return predict_fn


def _build_qiskit_quantum_predictor(model_dir: Path, qconf: dict):
    """Build a Qiskit-backed VQC predictor.

    Loads a cloudpickle-serialized QMLLinkPredictor from
    `model_dir/quantum_predictor.cloudpickle` and wraps its predict_proba.

    QMLLinkPredictor's underlying VQC contains an unpicklable closure
    (qiskit_machine_learning.algorithms.classifiers.vqc.VQC._get_interpret.parity)
    so plain joblib won't work — we use cloudpickle which handles closures.
    """
    import cloudpickle  # type: ignore[import-not-found]

    pkl_path = model_dir / "quantum_predictor.cloudpickle"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"qiskit-backed quantum manifest active but predictor file missing: {pkl_path}"
        )

    log.info("loading cloudpickled QMLLinkPredictor from %s", pkl_path)
    with pkl_path.open("rb") as f:
        predictor = cloudpickle.load(f)

    if not getattr(predictor, "is_fitted", False):
        raise RuntimeError(
            f"loaded predictor at {pkl_path} reports is_fitted=False"
        )

    expected_qubits = int(qconf.get("num_qubits", 0))
    actual_qubits = int(getattr(predictor, "num_qubits", 0))
    if expected_qubits and actual_qubits and expected_qubits != actual_qubits:
        log.warning(
            "quantum_config.json num_qubits=%s but loaded predictor has num_qubits=%s; using loaded",
            expected_qubits, actual_qubits,
        )

    def predict_fn(X: np.ndarray) -> np.ndarray:
        # QMLLinkPredictor.predict_proba returns shape (n, 2); class 1 column.
        proba = np.asarray(predictor.predict_proba(np.atleast_2d(X)))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    return predict_fn


def _try_load_embedder(settings: Settings, chain: Optional[ManifestChain]):
    """Build a feature_prep callable: (drug_id, disease_id) -> ndarray.

    For v1 with no manifest chain, falls back to data/entity_embeddings.npy +
    data/entity_ids.json — the legacy on-disk layout the audited orchestrator
    used. If those are missing, returns None (orchestrator can't run).
    """
    import json

    # Manifest-only lookup: read embeddings from the active EmbeddingManifest.
    if chain is None:
        log.warning("no active classical manifest chain; embedder unavailable")
        return None, None

    emb_dir = settings.artifacts_dir / "runs" / chain.embedding_id
    emb_path = emb_dir / "entity_embeddings.npy"
    ids_path = emb_dir / "entity_ids.json"
    if not emb_path.exists() or not ids_path.exists():
        log.warning(
            "manifest %s active but embedding artifacts missing at %s",
            chain.embedding_id, emb_dir,
        )
        return None, None

    log.info("loading embeddings from %s + %s", emb_path, ids_path)
    embeddings: np.ndarray = np.load(emb_path)
    with ids_path.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    if isinstance(ids, dict) and "entity_to_id" in ids:
        raw_entity_to_id = ids["entity_to_id"]
    elif isinstance(ids, dict):
        raw_entity_to_id = ids
    else:
        raw_entity_to_id = {eid: idx for idx, eid in enumerate(ids)}

    # Hetionet IDs in the on-disk index are prefixed: 'Compound::DB00001',
    # 'Disease::DOID:14330'. Expose unprefixed IDs ('DB00001', 'DOID:14330')
    # to the API and resolve back to prefixed for embedding lookup.
    def _strip(eid: str) -> str:
        if eid.startswith("Compound::"):
            return eid[len("Compound::"):]
        if eid.startswith("Disease::"):
            return eid[len("Disease::"):]
        return eid

    short_to_idx: dict[str, int] = {}
    for full, idx in raw_entity_to_id.items():
        short_to_idx[_strip(full)] = idx

    # Decide feature dimension by probing the loaded scaler/model:
    # if scaler exists and reports n_features_in_, use it to choose between
    # 3-vec '[h,t,|h-t|]' (3*d) and 4-vec '[h,t,|h-t|,h*t]' (4*d).
    feat_dim = embeddings.shape[1]

    def make_feature_prep(target_n_features: Optional[int]):
        if target_n_features is not None and target_n_features == 3 * feat_dim:
            mode = "h_t_abs"
        else:
            mode = "h_t_abs_prod"
        log.info("feature_prep mode=%s (target=%s, emb_dim=%s)",
                 mode, target_n_features, feat_dim)

        def feature_prep(drug_id: str, disease_id: str) -> np.ndarray:
            if drug_id not in short_to_idx or disease_id not in short_to_idx:
                raise KeyError(
                    f"entity not in embedding index: {drug_id} or {disease_id}"
                )
            h = embeddings[short_to_idx[drug_id]]
            t = embeddings[short_to_idx[disease_id]]
            if mode == "h_t_abs":
                feat = np.concatenate([h, t, np.abs(h - t)])
            else:
                feat = np.concatenate([h, t, np.abs(h - t), h * t])
            return feat.reshape(1, -1)

        return feature_prep

    class _MiniEmbedder:
        num_entities = embeddings.shape[0]
        entity_to_id = short_to_idx  # exposed via probes
        feature_dim = feat_dim
        _make_feature_prep = staticmethod(make_feature_prep)

    return _MiniEmbedder._make_feature_prep, _MiniEmbedder()


async def build_orchestrator_from_legacy(
    settings: Settings,
    classical_chain: Optional[ManifestChain],
    quantum_chain: Optional[ManifestChain] = None,
) -> Tuple[Orchestrator, QuantumExecutorProtocol]:
    """Construct an Orchestrator wired to the kept library.

    Raises if essentials are missing — caller (lifespan) should catch and
    mark the tracker as failed so /status reports the underlying cause.
    """
    embedder_pair = _try_load_embedder(settings, classical_chain)
    feature_prep_factory, embedder = embedder_pair
    if feature_prep_factory is None or embedder is None:
        raise RuntimeError("no embedder available; cannot serve predictions")

    classical_predict, scaler_n_features = _try_load_classical(settings, classical_chain)
    if classical_predict is None:
        raise RuntimeError("no classical model available; cannot serve predictions")

    feature_prep = feature_prep_factory(scaler_n_features)

    quantum_predict, quantum_fp_factory, qe = _try_load_quantum(settings, quantum_chain)
    quantum_feature_prep = (
        quantum_fp_factory(feature_prep) if quantum_fp_factory else None
    )

    # EntityResolver: derive valid id sets from the embedder's entity_to_id.
    drug_ids = {k for k in embedder.entity_to_id.keys() if k.startswith("DB")}
    disease_ids = {k for k in embedder.entity_to_id.keys() if k.startswith("DOID:")}
    from .synonyms import SynonymIndex
    syn = SynonymIndex()
    syn.load_external(settings.repo_root / "data")
    resolver = EntityResolver(drug_ids, disease_ids, synonyms=syn)
    log.info("entity resolver: %d drugs, %d diseases, synonyms=%s",
             len(drug_ids), len(disease_ids), syn.stats())

    bundle = PredictorBundle(
        classical_predict=classical_predict,
        quantum_predict=quantum_predict,
    )
    router = ExecutionRouter(bundle, qe)

    # Manifest-only lookup: classical_chain MUST be non-None at this point
    # because both _try_load_embedder and _try_load_classical refuse to
    # return without one. The assert is defensive; if this fires, the
    # earlier checks were bypassed.
    assert classical_chain is not None, "manifest-only mode requires a classical chain"

    orch = Orchestrator(
        resolver=resolver,
        router=router,
        feature_prep=feature_prep,
        classical_chain=classical_chain,
        quantum_chain=quantum_chain,
        quantum_feature_prep=quantum_feature_prep,
    )
    orch._embedder = embedder  # type: ignore[attr-defined]
    return orch, qe
