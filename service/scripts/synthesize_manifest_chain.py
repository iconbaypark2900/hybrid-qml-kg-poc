"""Synthesize a real manifest chain from existing models/ + data/ artifacts.

Reads the on-disk legacy artifacts (entity embeddings, scaler, classical model),
computes sha256 + size, builds three linked manifests (Embedding →
FeaturePipeline → Model), copies the artifacts into the runs/ tree under
the expected filenames, and writes LATEST.txt.

After running this script once, the service's manifest-chain probes pass:

    /status -> overall=ok
    /predict -> manifest_chain populated with real IDs

Idempotent: manifest IDs are derived from artifact sha256s, so re-running
with the same files produces the same IDs and short-circuits if LATEST.txt
already points at the resulting model.

Usage:
    python -m service.scripts.synthesize_manifest_chain \\
        [--root artifacts] \\
        [--models-dir models] \\
        [--data-dir data] \\
        [--config-dir config] \\
        [--force]   # overwrite existing manifests for these IDs
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from service.persistence import (  # noqa: E402
    LATEST_POINTER_FILENAME,
    SCHEMA_VERSION,
    save_embedding_manifest,
    save_feature_pipeline_manifest,
    save_model_manifest,
    set_active_model,
    sha256_file,
)
from service.schemas import (  # noqa: E402
    ArtifactRef,
    EmbeddingManifest,
    FeaturePipelineManifest,
    HetionetSource,
    ModelManifest,
    QuantumMode,
)

log = logging.getLogger("synthesize_manifest_chain")


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def _pick_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _git_sha(root: Path) -> str:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _file_artifact(path: Path, label: str) -> ArtifactRef:
    sha = sha256_file(path)
    return ArtifactRef(path=label, sha256=sha, size_bytes=path.stat().st_size)


def _short_id(prefix: str, *contributing_shas: str) -> str:
    """Stable manifest_id derived from the SHAs of the artifacts it covers."""
    import hashlib
    h = hashlib.sha256()
    for s in contributing_shas:
        h.update(s.encode())
    return f"{prefix}-{h.hexdigest()[:12]}"


def _config_files_hash(config_dir: Path) -> dict[str, str]:
    if not config_dir.exists():
        return {}
    out: dict[str, str] = {}
    for p in sorted(config_dir.glob("*.yaml")):
        try:
            out[p.name] = sha256_file(p)
        except OSError:
            continue
    return out


def _detect_embedding_method(emb_filename: str) -> tuple[str, int]:
    """Best-effort guess of (method, dim) from filename. Defaults applied
    when the filename has no clear hints (e.g. 'entity_embeddings.npy')."""
    lower = emb_filename.lower()
    if "complex" in lower:
        method = "ComplEx"
    elif "rotate" in lower:
        method = "RotatE"
    elif "distmult" in lower:
        method = "DistMult"
    elif "transe" in lower:
        method = "TransE"
    else:
        method = "unknown"
    dim = 0
    for token in lower.replace(".npy", "").split("_"):
        if token.endswith("d") and token[:-1].isdigit():
            dim = int(token[:-1])
            break
    return method, dim


# ---------------------------------------------------------------------------
# Manifest synthesis
# ---------------------------------------------------------------------------


def _detect_hetionet_source(data_dir: Path, repo_root: Path) -> HetionetSource:
    """Hash known Hetionet snapshot files if present so the manifest pins
    the exact release used. Falls back to the v1.0 default if no snapshot
    file is in `data_dir`."""
    candidates_nodes = [
        data_dir / "hetionet-v1.0-nodes.tsv",
        data_dir / "hetionet" / "hetionet-v1.0-nodes.tsv",
    ]
    candidates_edges = [
        data_dir / "hetionet-v1.0-edges.sif",
        data_dir / "hetionet" / "hetionet-v1.0-edges.sif",
        data_dir / "hetionet-v1.0-edges.tsv",
    ]
    nodes_path = next((p for p in candidates_nodes if p.exists()), None)
    edges_path = next((p for p in candidates_edges if p.exists()), None)
    version_file = data_dir / "VERSION"
    release_tag = "v1.0"
    if version_file.exists():
        release_tag = version_file.read_text(encoding="utf-8").strip() or "v1.0"
    return HetionetSource(
        release_tag=release_tag,
        release_url="https://github.com/hetio/hetionet",
        nodes_sha256=sha256_file(nodes_path) if nodes_path else None,
        edges_sha256=sha256_file(edges_path) if edges_path else None,
        snapshot_path=str(nodes_path) if nodes_path else None,
    )


def synthesize(
    root: Path,
    models_dir: Path,
    data_dir: Path,
    config_dir: Path,
    repo_root: Path,
    force: bool = False,
) -> dict:
    # 1. Discover artifacts
    emb_path = _pick_first_existing([
        data_dir / "entity_embeddings.npy",
        data_dir / "rotate_128d_entity_embeddings.npy",
        data_dir / "complex_128d_entity_embeddings.npy",
        data_dir / "rotate_256d_entity_embeddings.npy",
    ])
    if emb_path is None:
        return {"ok": False, "error": "no entity_embeddings.npy found"}

    ids_path = emb_path.with_name(
        emb_path.stem.replace("_embeddings", "_ids") + ".json"
    )
    if not ids_path.exists():
        ids_path = data_dir / "entity_ids.json"
    if not ids_path.exists():
        return {"ok": False, "error": f"no entity ids file alongside {emb_path}"}

    model_path = _pick_first_existing([
        models_dir / "classical_serving.joblib",
        models_dir / "classical_logisticregression.joblib",
        models_dir / "classical_best.joblib",
    ])
    if model_path is None:
        return {"ok": False, "error": f"no classical model found in {models_dir}"}

    scaler_path = _pick_first_existing([
        models_dir / "scaler.joblib",
        models_dir / "classical_best_scaler.joblib",
    ])

    # 2. Hash artifacts
    emb_sha = sha256_file(emb_path)
    ids_sha = sha256_file(ids_path)
    model_sha = sha256_file(model_path)
    scaler_sha = sha256_file(scaler_path) if scaler_path else ""

    # 3. Manifest IDs (deterministic from contributing SHAs)
    emb_id = _short_id("EMB", emb_sha, ids_sha)
    fp_id = _short_id("FP", emb_id, scaler_sha)
    mdl_id = _short_id("MDL", fp_id, model_sha)

    # 4. Idempotency check
    pointer = root / LATEST_POINTER_FILENAME
    if pointer.exists() and pointer.read_text(encoding="utf-8").strip() == mdl_id and not force:
        log.info("LATEST.txt already points at %s; nothing to do (use --force to rebuild)", mdl_id)
        return {
            "ok": True, "noop": True,
            "embedding_id": emb_id, "feature_pipeline_id": fp_id, "model_id": mdl_id,
        }

    # 5. Determine feature-pipeline mode by inspecting the scaler.
    #    n_features_in_ tells us whether the saved pipeline expects 3*d ('diff')
    #    or 4*d ('both'). If no scaler, default to 'classical_only' as the safe
    #    label — orchestrator will then refuse quantum strict (no quantum FP).
    feature_mode: str = "classical_only"
    qml_dim = 0
    embedding_dim = 0
    try:
        import numpy as np
        embeddings = np.load(emb_path, mmap_mode="r")
        embedding_dim = int(embeddings.shape[1])
    except Exception as e:
        log.warning("could not read embedding shape: %s", e)

    if scaler_path is not None:
        try:
            import joblib
            scaler = joblib.load(scaler_path)
            n = int(getattr(scaler, "n_features_in_", 0))
            if embedding_dim and n == 3 * embedding_dim:
                feature_mode = "diff"
                qml_dim = embedding_dim
            elif embedding_dim and n == 4 * embedding_dim:
                feature_mode = "both"
                qml_dim = embedding_dim
            else:
                log.warning(
                    "scaler n_features_in_=%s does not match 3*d=%s or 4*d=%s; "
                    "leaving feature_mode=classical_only",
                    n, 3 * embedding_dim, 4 * embedding_dim,
                )
        except Exception as e:
            log.warning("could not read scaler n_features_in_: %s", e)

    # 6. Discover embedding metadata
    method, declared_dim = _detect_embedding_method(emb_path.name)
    if not declared_dim and embedding_dim:
        declared_dim = embedding_dim

    config_hashes = _config_files_hash(config_dir)
    git_sha = _git_sha(repo_root)
    now = time.time()

    # 7. Copy artifacts into runs/ tree under canonical filenames
    runs_root = root / "runs"
    emb_dir = runs_root / emb_id
    fp_dir = runs_root / fp_id
    mdl_dir = runs_root / mdl_id

    if force:
        for d in (emb_dir, fp_dir, mdl_dir):
            if d.exists():
                shutil.rmtree(d)

    emb_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)

    canonical_emb = emb_dir / "entity_embeddings.npy"
    canonical_ids = emb_dir / "entity_ids.json"
    canonical_scaler = fp_dir / "scaler.joblib"
    canonical_model = mdl_dir / "model.joblib"

    if not canonical_emb.exists():
        shutil.copy2(emb_path, canonical_emb)
    if not canonical_ids.exists():
        shutil.copy2(ids_path, canonical_ids)
    if scaler_path and not canonical_scaler.exists():
        shutil.copy2(scaler_path, canonical_scaler)
    if not canonical_model.exists():
        shutil.copy2(model_path, canonical_model)

    # 8. Build & write manifests
    hetionet_src = _detect_hetionet_source(data_dir, repo_root)
    emb_manifest = EmbeddingManifest(
        manifest_id=emb_id,
        created_at=now,
        git_sha=git_sha,
        seed=42,
        relation="CtD",
        max_entities=0,  # 0 = uncapped; legacy data may not record this faithfully
        method=method,
        dim=declared_dim,
        epochs=0,
        artifacts={
            "entity_embeddings": ArtifactRef(
                path="entity_embeddings.npy",
                sha256=emb_sha,
                size_bytes=emb_path.stat().st_size,
            ),
            "entity_ids": ArtifactRef(
                path="entity_ids.json",
                sha256=ids_sha,
                size_bytes=ids_path.stat().st_size,
            ),
        },
        config_files=config_hashes,
        hetionet=hetionet_src,
    )
    save_embedding_manifest(emb_manifest, root)

    fp_artifacts: dict[str, ArtifactRef] = {}
    if scaler_path is not None:
        fp_artifacts["scaler"] = ArtifactRef(
            path="scaler.joblib", sha256=scaler_sha,
            size_bytes=scaler_path.stat().st_size,
        )
    fp_manifest = FeaturePipelineManifest(
        manifest_id=fp_id,
        parent_embedding=emb_id,
        created_at=now,
        qml_dim=qml_dim,
        mode=feature_mode,  # type: ignore[arg-type]  # validated by Pydantic
        artifacts=fp_artifacts,
    )
    save_feature_pipeline_manifest(fp_manifest, root)

    mdl_manifest = ModelManifest(
        manifest_id=mdl_id,
        kind="classical",
        parent_feature_pipeline=fp_id,
        created_at=now,
        model_type="logistic_regression",
        quantum_execution_mode_at_train=QuantumMode.UNAVAILABLE,
        artifacts={
            "model": ArtifactRef(
                path="model.joblib", sha256=model_sha,
                size_bytes=model_path.stat().st_size,
            ),
        },
    )
    save_model_manifest(mdl_manifest, root)

    # 9. Activate
    set_active_model(root, mdl_id)

    return {
        "ok": True,
        "noop": False,
        "embedding_id": emb_id,
        "feature_pipeline_id": fp_id,
        "model_id": mdl_id,
        "embedding_method": method,
        "embedding_dim": declared_dim,
        "feature_mode": feature_mode,
        "qml_dim": qml_dim,
        "sources": {
            "entity_embeddings": str(emb_path),
            "entity_ids": str(ids_path),
            "scaler": str(scaler_path) if scaler_path else None,
            "classical_model": str(model_path),
        },
        "schema_version": SCHEMA_VERSION,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts",
                   help="Service artifacts directory (default: <repo>/artifacts)")
    p.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--config-dir", type=Path, default=REPO_ROOT / "config")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing manifest dirs for the resulting IDs")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    summary = synthesize(
        root=args.root,
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        repo_root=REPO_ROOT,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
