"""Synthesize a quantum manifest chain — feature_pipeline + model with weights.

Three modes:

  --mode synthetic       (default): writes deterministic placeholder weights to
                         quantum_weights.npz and quantum_config.json with
                         "synthetic": true. The service loads them via
                         _build_synthetic_quantum_predictor — useful for verifying
                         the chain wiring before a real VQC is trained.

  --mode qiskit-mini-train  fits a tiny real VQC on synthetic binary data and
                            cloudpickles the trained QMLLinkPredictor into
                            quantum_predictor.cloudpickle. The service loads it
                            via _build_qiskit_quantum_predictor. Predictions are
                            real Qiskit VQC outputs (with shot-based noise) but
                            the training data is synthetic — manifest records
                            this clearly.

  --mode qiskit          copies a pre-trained QMLLinkPredictor file (pass
                         --predictor-path pointing at a cloudpickle blob).
                         Requires the user to have trained elsewhere.

The active classical chain (LATEST.txt) supplies the parent embedding manifest;
the quantum FP manifest reuses that embedding so both predictors share the
same vector space.

Examples:
    # synthetic (deterministic stand-in; manifest flagged synthetic=true)
    python -m service.scripts.synthesize_quantum_manifest --num-qubits 4

    # real Qiskit VQC trained briefly on synthetic data (slow first time;
    # produces a real cloudpickled predictor the service can load)
    python -m service.scripts.synthesize_quantum_manifest \\
        --mode qiskit-mini-train --num-qubits 4 --max-iter 10

    # pre-trained QMLLinkPredictor from elsewhere
    python -m service.scripts.synthesize_quantum_manifest \\
        --mode qiskit --predictor-path path/to/predictor.cloudpickle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from service.persistence import (  # noqa: E402
    LATEST_QUANTUM_POINTER_FILENAME,
    SCHEMA_VERSION,
    load_active_manifest_chain,
    save_feature_pipeline_manifest,
    save_model_manifest,
    set_active_quantum_model,
    sha256_file,
)
from service.schemas import (  # noqa: E402
    ArtifactRef,
    FeaturePipelineManifest,
    ModelManifest,
    QuantumMode,
)

log = logging.getLogger("synthesize_quantum_manifest")


def _short_id(prefix: str, *contributing_shas: str) -> str:
    h = hashlib.sha256()
    for s in contributing_shas:
        h.update(s.encode())
    return f"{prefix}-{h.hexdigest()[:12]}"


def _write_synthetic_weights(out_path: Path, num_qubits: int, seed: int) -> str:
    """Generate deterministic placeholder weights. Returns sha256 of the file."""
    rng = np.random.default_rng(seed)
    # Shape: (num_qubits,) — the synthetic predictor only uses the first
    # num_qubits entries via dot product against the leading features.
    weights = rng.normal(scale=0.5, size=(num_qubits,)).astype(np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, weights=weights)
    return sha256_file(out_path)


def _write_synthetic_config(
    out_path: Path,
    num_qubits: int,
    seed: int,
    notes: str,
) -> str:
    payload = {
        "schema_version": 1,
        "backend": "synthetic",
        "synthetic": True,
        "num_qubits": num_qubits,
        "seed": seed,
        "notes": notes,
        "feature_map": None,
        "ansatz": None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sha256_file(out_path)


def _qiskit_mini_train(
    staging: Path,
    num_qubits: int,
    max_iter: int,
    seed: int,
    feature_map_type: str,
    ansatz_type: str,
    requested_mode: str = "auto",
    strict: bool = False,
) -> tuple[str, str, Optional[Path]]:
    """Fit a tiny VQC on synthetic binary data and cloudpickle it.

    Returns (weights_sha, config_sha, cloudpickle_path).

    Captures both `requested_mode` and the actual exec_mode the kept
    QuantumExecutor selected at fit time. If they differ and `strict=True`,
    raises rather than persisting a manifest that misrepresents what ran.
    """
    import logging as _logging
    _logging.getLogger("qiskit_machine_learning").setLevel(_logging.WARNING)
    _logging.getLogger("qiskit").setLevel(_logging.WARNING)
    _logging.getLogger("quantum_layer.quantum_executor").setLevel(_logging.WARNING)

    import cloudpickle  # noqa: F401  (import-check; raised below if missing)
    from quantum_layer.qml_model import QMLLinkPredictor

    log.info("fitting tiny VQC (num_qubits=%d, max_iter=%d, feature_map=%s, ansatz=%s)",
             num_qubits, max_iter, feature_map_type, ansatz_type)

    rng = np.random.default_rng(seed)
    n_samples = max(8, 2 * num_qubits)
    X = rng.normal(size=(n_samples, num_qubits)).astype(np.float64)
    y = (X.sum(axis=1) > 0).astype(int)

    predictor = QMLLinkPredictor(
        model_type="VQC",
        encoding_method="feature_map",
        num_qubits=num_qubits,
        ansatz_type=ansatz_type,
        ansatz_reps=1,
        optimizer="COBYLA",
        max_iter=max_iter,
        feature_map_type=feature_map_type,
        feature_map_reps=1,
        random_state=seed,
    )
    predictor.fit(X, y)

    # Capture what the kept QuantumExecutor actually used. This is the audit's
    # "silent simulator fallback" surface — the manifest now records both
    # what was requested and what ran.
    actual_mode = "unknown"
    try:
        qx = getattr(predictor, "_quantum_executor", None)
        if qx is not None and hasattr(qx, "get_execution_metadata"):
            meta = qx.get_execution_metadata()
            actual_mode = str(meta.get("execution_mode") or meta.get("mode") or "unknown")
        elif qx is None:
            # QMLLinkPredictor's local sampler fallback path
            actual_mode = "simulator_local_fallback"
    except Exception:
        actual_mode = "unknown"

    if strict and requested_mode not in ("auto", "", actual_mode):
        raise RuntimeError(
            f"strict mode: requested_mode={requested_mode!r} but actual execution "
            f"was {actual_mode!r}. Refusing to persist a misleading manifest."
        )

    # Persist trained weights (the float array) and the cloudpickled predictor
    # (so the loader can resume without retraining).
    weights = np.asarray(predictor.model.weights, dtype=np.float64)
    weights_path = staging / "quantum_weights.npz"
    np.savez(weights_path, weights=weights)
    weights_sha = sha256_file(weights_path)

    pkl_path = staging / "quantum_predictor.cloudpickle"
    with pkl_path.open("wb") as f:
        cloudpickle.dump(predictor, f)
    log.info("cloudpickled predictor: %d bytes", pkl_path.stat().st_size)

    fallback_used = (
        requested_mode not in ("auto", "", actual_mode)
        and actual_mode in ("simulator", "simulator_statevector",
                            "simulator_local_fallback", "simulator_aer")
    )
    config = {
        "schema_version": 1,
        "backend": "qiskit",
        "synthetic": False,
        "num_qubits": num_qubits,
        "ansatz": ansatz_type,
        "ansatz_reps": 1,
        "feature_map": feature_map_type,
        "feature_map_reps": 1,
        "optimizer": "COBYLA",
        "max_iter": max_iter,
        "training_data": "synthetic",
        "training_n_samples": n_samples,
        "requested_execution_mode": requested_mode,
        "actual_execution_mode": actual_mode,
        "fallback_used_at_train": fallback_used,
        "notes": (
            "VQC fitted on synthetic binary data via "
            "synthesize_quantum_manifest --mode qiskit-mini-train. "
            "Real Qiskit predictor; not trained on Hetionet drug-disease pairs. "
            "Replace with a predictor trained on real data for benchmarking."
            + (
                f" WARNING: fallback was used at training time — requested "
                f"{requested_mode!r}, ran on {actual_mode!r}."
                if fallback_used else ""
            )
        ),
    }
    config_path = staging / "quantum_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    config_sha = sha256_file(config_path)

    return weights_sha, config_sha, pkl_path


def synthesize(
    root: Path,
    mode: str,
    num_qubits: int,
    predictor_source: Optional[Path],
    config_source: Optional[Path],
    seed: int,
    force: bool,
    max_iter: int = 10,
    feature_map_type: str = "Z",
    ansatz_type: str = "RealAmplitudes",
    requested_execution_mode: str = "auto",
    strict_execution_mode: bool = False,
) -> dict:
    classical_chain = load_active_manifest_chain(root)
    if classical_chain is None:
        return {"ok": False, "error": "no active classical manifest chain "
                                       "(run synthesize_manifest_chain first)"}

    runs_root = root / "runs"

    # Stage temporary working dir to compute final IDs from sha256 of contents.
    staging = root / ".staging" / f"qmdl-staging-{int(time.time())}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    weights_path = staging / "quantum_weights.npz"
    config_path = staging / "quantum_config.json"
    pkl_path: Optional[Path] = None

    try:
        if mode == "synthetic":
            weights_sha = _write_synthetic_weights(weights_path, num_qubits, seed)
            notes = (
                "Placeholder weights generated by synthesize_quantum_manifest "
                "in synthetic mode. NOT a trained VQC. The service loads these "
                "via _build_synthetic_quantum_predictor; predictions are a "
                "deterministic sigmoid of feature·weights. Use --mode "
                "qiskit-mini-train for a real (tiny) Qiskit VQC."
            )
            config_sha = _write_synthetic_config(config_path, num_qubits, seed, notes)
        elif mode == "qiskit-mini-train":
            weights_sha, config_sha, pkl_path = _qiskit_mini_train(
                staging, num_qubits, max_iter, seed, feature_map_type, ansatz_type,
                requested_mode=requested_execution_mode,
                strict=strict_execution_mode,
            )
        elif mode == "qiskit":
            if predictor_source is None:
                return {"ok": False,
                        "error": "--predictor-path is required for --mode qiskit "
                                 "(point at a cloudpickled QMLLinkPredictor)"}
            if not predictor_source.exists():
                return {"ok": False, "error": f"predictor file not found: {predictor_source}"}
            pkl_path = staging / "quantum_predictor.cloudpickle"
            shutil.copy2(predictor_source, pkl_path)
            # Save a placeholder weights file (real weights live inside the
            # cloudpickle); existence still satisfies the probe contract.
            weights_sha = _write_synthetic_weights(weights_path, num_qubits, seed)
            if config_source and config_source.exists():
                shutil.copy2(config_source, config_path)
            else:
                config_path.write_text(json.dumps({
                    "schema_version": 1,
                    "backend": "qiskit",
                    "synthetic": False,
                    "num_qubits": num_qubits,
                    "feature_map": "ZZFeatureMap",
                    "ansatz": "RealAmplitudes",
                    "notes": "config inferred; ensure num_qubits matches training",
                }, indent=2, sort_keys=True), encoding="utf-8")
            config_sha = sha256_file(config_path)
        else:
            return {"ok": False, "error": f"unknown mode: {mode}"}

        # Manifest IDs are stable from the artifact contents.
        qfp_id = _short_id("QFP", classical_chain.embedding_id, str(num_qubits))
        qmdl_id = _short_id("QMDL", qfp_id, weights_sha, config_sha)

        # Idempotency
        pointer = root / LATEST_QUANTUM_POINTER_FILENAME
        if (
            pointer.exists()
            and pointer.read_text(encoding="utf-8").strip() == qmdl_id
            and not force
        ):
            log.info("LATEST_QUANTUM.txt already at %s; nothing to do", qmdl_id)
            return {
                "ok": True, "noop": True,
                "feature_pipeline_id": qfp_id, "model_id": qmdl_id,
                "embedding_id": classical_chain.embedding_id,
            }

        qfp_dir = runs_root / qfp_id
        qmdl_dir = runs_root / qmdl_id
        if force:
            for d in (qfp_dir, qmdl_dir):
                if d.exists():
                    shutil.rmtree(d)
        qfp_dir.mkdir(parents=True, exist_ok=True)
        qmdl_dir.mkdir(parents=True, exist_ok=True)

        canonical_weights = qmdl_dir / "quantum_weights.npz"
        canonical_config = qmdl_dir / "quantum_config.json"
        canonical_pkl = qmdl_dir / "quantum_predictor.cloudpickle"
        for p in (canonical_weights, canonical_config, canonical_pkl):
            if p.exists():
                p.unlink()
        shutil.move(str(weights_path), str(canonical_weights))
        shutil.move(str(config_path), str(canonical_config))
        if pkl_path is not None and pkl_path.exists():
            shutil.move(str(pkl_path), str(canonical_pkl))

        # Build manifests
        qfp = FeaturePipelineManifest(
            manifest_id=qfp_id,
            parent_embedding=classical_chain.embedding_id,
            created_at=time.time(),
            qml_dim=num_qubits,
            mode="diff",  # quantum FP applies leading-feature slice; reusing 'diff' label
            artifacts={},
        )
        save_feature_pipeline_manifest(qfp, root)

        artifacts: dict[str, ArtifactRef] = {
            "quantum_weights": ArtifactRef(
                path="quantum_weights.npz",
                sha256=weights_sha,
                size_bytes=canonical_weights.stat().st_size,
            ),
            "quantum_config": ArtifactRef(
                path="quantum_config.json",
                sha256=config_sha,
                size_bytes=canonical_config.stat().st_size,
            ),
        }
        if canonical_pkl.exists():
            artifacts["quantum_predictor"] = ArtifactRef(
                path="quantum_predictor.cloudpickle",
                sha256=sha256_file(canonical_pkl),
                size_bytes=canonical_pkl.stat().st_size,
            )

        if mode == "synthetic":
            model_type_label = "synthetic_placeholder"
        elif mode == "qiskit-mini-train":
            model_type_label = "vqc_mini_trained_synthetic_data"
        else:
            model_type_label = "vqc"

        qmdl = ModelManifest(
            manifest_id=qmdl_id,
            kind="quantum",
            parent_feature_pipeline=qfp_id,
            created_at=time.time(),
            model_type=model_type_label,
            quantum_execution_mode_at_train=QuantumMode.SIMULATOR,
            artifacts=artifacts,
        )
        save_model_manifest(qmdl, root)
        set_active_quantum_model(root, qmdl_id)

        return {
            "ok": True,
            "noop": False,
            "embedding_id": classical_chain.embedding_id,
            "feature_pipeline_id": qfp_id,
            "model_id": qmdl_id,
            "model_type": model_type_label,
            "mode": mode,
            "num_qubits": num_qubits,
            "synthetic": mode == "synthetic",
            "predictor_persisted": canonical_pkl.exists(),
            "schema_version": SCHEMA_VERSION,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--mode",
                   choices=["synthetic", "qiskit-mini-train", "qiskit"],
                   default="synthetic")
    p.add_argument("--num-qubits", type=int, default=4)
    p.add_argument("--predictor-path", type=Path, default=None,
                   help="Cloudpickled QMLLinkPredictor (required for --mode qiskit)")
    p.add_argument("--config-path", type=Path, default=None,
                   help="Optional VQC config .json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=10,
                   help="VQC optimizer max iterations (qiskit-mini-train only)")
    p.add_argument("--feature-map", default="Z", choices=["Z", "ZZ"],
                   help="VQC feature map type (qiskit-mini-train only)")
    p.add_argument("--ansatz", default="RealAmplitudes",
                   choices=["RealAmplitudes", "EfficientSU2"],
                   help="VQC ansatz (qiskit-mini-train only)")
    p.add_argument("--requested-execution-mode", default="auto",
                   help="Mode that was requested (e.g. 'ibm_heron'); manifest "
                        "captures both this and what actually ran")
    p.add_argument("--strict-execution-mode", action="store_true",
                   help="Refuse to persist if requested mode != actual mode "
                        "(prevents silent fallback making it into a manifest)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing manifests with the same IDs")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    summary = synthesize(
        root=args.root,
        mode=args.mode,
        num_qubits=args.num_qubits,
        predictor_source=args.predictor_path,
        config_source=args.config_path,
        seed=args.seed,
        force=args.force,
        max_iter=args.max_iter,
        feature_map_type=args.feature_map,
        ansatz_type=args.ansatz,
        requested_execution_mode=args.requested_execution_mode,
        strict_execution_mode=args.strict_execution_mode,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
