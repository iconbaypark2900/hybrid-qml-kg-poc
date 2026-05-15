"""Manifest DAG, JSONL evaluations, LATEST.txt pointer.

Layout under settings.artifacts_dir:
    artifacts/
        LATEST.txt                            # contains active model_manifest_id
        runs/
            <model_manifest_id>/
                manifest.json                 # ModelManifest
                model.joblib                  # classical or quantum weights
                ... (kind-specific artifacts)
            <feature_pipeline_manifest_id>/
                manifest.json                 # FeaturePipelineManifest
                pca_fit.joblib
                scaler.joblib
            <embedding_manifest_id>/
                manifest.json                 # EmbeddingManifest
                entity_embeddings.npy
                relation_embeddings.npy
                vocab.json
        evaluations/
            <tenant_id>.jsonl                 # one JSONL per tenant, append-only
        jobs/
            <tenant_id>/
                <job_id>.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from .async_runtime import run_io_bound
from .schemas import (
    EmbeddingManifest,
    EvaluationRecord,
    FeaturePipelineManifest,
    JobRecord,
    ManifestChain,
    ModelManifest,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
LATEST_POINTER_FILENAME = "LATEST.txt"
LATEST_QUANTUM_POINTER_FILENAME = "LATEST_QUANTUM.txt"
EVALUATIONS_DIR = "evaluations"
RUNS_DIR = "runs"
JOBS_DIR = "jobs"


# ---------------------------------------------------------------------------
# Manifest DAG
# ---------------------------------------------------------------------------


def _manifest_path(root: Path, manifest_id: str) -> Path:
    return root / RUNS_DIR / manifest_id / "manifest.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_embedding_manifest(m: EmbeddingManifest, root: Path) -> Path:
    path = _manifest_path(root, m.manifest_id)
    _write_json(path, {"schema_version": SCHEMA_VERSION, **m.model_dump()})
    return path


def save_feature_pipeline_manifest(m: FeaturePipelineManifest, root: Path) -> Path:
    path = _manifest_path(root, m.manifest_id)
    _write_json(path, {"schema_version": SCHEMA_VERSION, **m.model_dump()})
    return path


def save_model_manifest(m: ModelManifest, root: Path) -> Path:
    path = _manifest_path(root, m.manifest_id)
    _write_json(path, {"schema_version": SCHEMA_VERSION, **m.model_dump()})
    return path


def load_embedding_manifest(root: Path, manifest_id: str) -> EmbeddingManifest:
    payload = _read_json(_manifest_path(root, manifest_id))
    payload.pop("schema_version", None)
    return EmbeddingManifest.model_validate(payload)


def load_feature_pipeline_manifest(root: Path, manifest_id: str) -> FeaturePipelineManifest:
    payload = _read_json(_manifest_path(root, manifest_id))
    payload.pop("schema_version", None)
    return FeaturePipelineManifest.model_validate(payload)


def load_model_manifest(root: Path, manifest_id: str) -> ModelManifest:
    payload = _read_json(_manifest_path(root, manifest_id))
    payload.pop("schema_version", None)
    return ModelManifest.model_validate(payload)


def set_active_model(root: Path, model_manifest_id: str) -> None:
    """Write LATEST.txt pointing at the new active classical model_manifest_id.
    NOT a symlink — Windows-safe."""
    pointer = root / LATEST_POINTER_FILENAME
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_suffix(".tmp")
    tmp.write_text(model_manifest_id, encoding="utf-8")
    tmp.replace(pointer)


def set_active_quantum_model(root: Path, model_manifest_id: str) -> None:
    """Write LATEST_QUANTUM.txt pointing at the active quantum model_manifest_id."""
    pointer = root / LATEST_QUANTUM_POINTER_FILENAME
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_suffix(".tmp")
    tmp.write_text(model_manifest_id, encoding="utf-8")
    tmp.replace(pointer)


def load_active_quantum_chain(root: Path) -> Optional[ManifestChain]:
    """Resolve LATEST_QUANTUM.txt → ModelManifest → FeaturePipelineManifest →
    EmbeddingManifest. Returns None if pointer missing or chain broken."""
    pointer = root / LATEST_QUANTUM_POINTER_FILENAME
    if not pointer.exists():
        return None
    try:
        model_id = pointer.read_text(encoding="utf-8").strip()
        if not model_id:
            return None
        model_m = load_model_manifest(root, model_id)
        if model_m.kind != "quantum":
            log.warning("LATEST_QUANTUM.txt points at non-quantum model %s", model_id)
            return None
        fp_m = load_feature_pipeline_manifest(root, model_m.parent_feature_pipeline)
        emb_m = load_embedding_manifest(root, fp_m.parent_embedding)
    except FileNotFoundError as e:
        log.warning("quantum manifest chain broken at %s: %s", pointer, e)
        return None
    except Exception as e:
        log.warning("failed to load active quantum manifest chain: %s", e)
        return None
    return ManifestChain(
        embedding_id=emb_m.manifest_id,
        feature_pipeline_id=fp_m.manifest_id,
        model_id=model_m.manifest_id,
    )


def load_active_manifest_chain(root: Path) -> Optional[ManifestChain]:
    """Resolve LATEST.txt → ModelManifest → FeaturePipelineManifest → EmbeddingManifest.
    Returns None if pointer missing or chain broken (logs a warning)."""
    pointer = root / LATEST_POINTER_FILENAME
    if not pointer.exists():
        return None
    try:
        model_id = pointer.read_text(encoding="utf-8").strip()
        if not model_id:
            return None
        model_m = load_model_manifest(root, model_id)
        fp_m = load_feature_pipeline_manifest(root, model_m.parent_feature_pipeline)
        emb_m = load_embedding_manifest(root, fp_m.parent_embedding)
    except FileNotFoundError as e:
        log.warning("manifest chain broken at %s: %s", pointer, e)
        return None
    except Exception as e:
        log.warning("failed to load active manifest chain: %s", e)
        return None
    return ManifestChain(
        embedding_id=emb_m.manifest_id,
        feature_pipeline_id=fp_m.manifest_id,
        model_id=model_m.manifest_id,
    )


# ---------------------------------------------------------------------------
# Evaluations (JSONL, per-tenant)
# ---------------------------------------------------------------------------


def evaluation_path(root: Path, tenant_id: str) -> Path:
    return root / EVALUATIONS_DIR / f"{tenant_id}.jsonl"


def _append_jsonl_sync(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def append_evaluation(record: EvaluationRecord, root: Path) -> None:
    path = evaluation_path(root, record.tenant_id)
    payload = {"schema_version": SCHEMA_VERSION, **record.model_dump()}
    line = json.dumps(payload, sort_keys=True)
    await run_io_bound(_append_jsonl_sync, path, line)


def _read_jsonl_sync(
    path: Path,
    model_manifest_id: Optional[str],
    limit: int,
) -> list[EvaluationRecord]:
    if not path.exists():
        return []
    seen: set[str] = set()
    out: list[EvaluationRecord] = []
    # Iterate from newest to oldest. For a small dataset, read entire file
    # and reverse; for very large files this should be replaced with an index.
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            log.warning("evaluations: skipping unparseable line in %s", path.name)
            continue
        if payload.get("schema_version") != SCHEMA_VERSION:
            log.warning("evaluations: skipping schema_version=%s in %s",
                        payload.get("schema_version"), path.name)
            continue
        payload.pop("schema_version", None)
        try:
            rec = EvaluationRecord.model_validate(payload)
        except Exception as e:
            log.warning("evaluations: validation failed in %s: %s", path.name, e)
            continue
        if rec.evaluation_id in seen:
            continue
        if model_manifest_id and rec.manifest_chain.model_id != model_manifest_id:
            continue
        seen.add(rec.evaluation_id)
        out.append(rec)
        if len(out) >= limit:
            break
    return out


async def list_evaluations(
    root: Path,
    tenant_id: str,
    model_manifest_id: Optional[str] = None,
    limit: int = 50,
) -> list[EvaluationRecord]:
    path = evaluation_path(root, tenant_id)
    return await run_io_bound(_read_jsonl_sync, path, model_manifest_id, limit)


# ---------------------------------------------------------------------------
# Job records (per-tenant directories for isolation)
# ---------------------------------------------------------------------------


def job_path(root: Path, tenant_id: str, job_id: str) -> Path:
    return root / JOBS_DIR / tenant_id / f"{job_id}.json"


def _persist_job_sync(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


async def persist_job(record: JobRecord, root: Path) -> None:
    path = job_path(root, record.tenant_id, record.job_id)
    await run_io_bound(_persist_job_sync, path, record.model_dump())


def _load_job_sync(path: Path) -> Optional[JobRecord]:
    if not path.exists():
        return None
    try:
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("failed to load job %s: %s", path, e)
        return None


async def load_job(root: Path, tenant_id: str, job_id: str) -> Optional[JobRecord]:
    path = job_path(root, tenant_id, job_id)
    return await run_io_bound(_load_job_sync, path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def now_ts() -> float:
    return time.time()
