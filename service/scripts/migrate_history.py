"""One-shot migration: benchmarking/experiment_history.csv -> evaluations.jsonl.

Strategy: synthesize a single 'LEGACY' manifest chain that all pre-rebuild
rows point to. All migrated rows go into the system 'legacy' tenant's
evaluations file (read-only — no real API key resolves to this tenant).

The script is idempotent: rerunning skips evaluation_ids already present in
the destination file.

Usage:
    python -m service.scripts.migrate_history \\
        --csv benchmarking/experiment_history.csv \\
        --root artifacts \\
        [--tenant-id legacy] \\
        [--legacy-manifest-id LEGACY-PRE-REBUILD] \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

# Ensure imports work when run as `python -m service.scripts.migrate_history`
SERVICE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVICE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from service.persistence import (  # noqa: E402
    SCHEMA_VERSION,
    evaluation_path,
    save_embedding_manifest,
    save_feature_pipeline_manifest,
    save_model_manifest,
)
from service.schemas import (  # noqa: E402
    EmbeddingManifest,
    EvaluationRecord,
    FeaturePipelineManifest,
    ManifestChain,
    ModelManifest,
    QuantumMode,
)
from service.tenants import SYSTEM_LEGACY_TENANT_ID  # noqa: E402

log = logging.getLogger("migrate_history")


_METRIC_BASES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                 "brier", "mcc", "ece")
# Bare column names produced by toy/synthetic CSVs.
# Prefixed columns (classical_pr_auc, quantum_pr_auc) come from the real
# benchmarking/experiment_history.csv schema. Both forms are accepted.
_METRIC_PREFIXES = ("", "classical_", "quantum_")
METRIC_KEYS = tuple(p + base for p in _METRIC_PREFIXES for base in _METRIC_BASES)


def synthesize_legacy_chain(root: Path, base_id: str) -> ManifestChain:
    """Write three placeholder manifests representing pre-rebuild state.

    Idempotent — if the manifest files already exist, leaves them in place.
    """
    chain = ManifestChain(
        embedding_id=f"{base_id}-EMB",
        feature_pipeline_id=f"{base_id}-FP",
        model_id=f"{base_id}-MODEL",
    )
    now = time.time()

    emb_path = root / "runs" / chain.embedding_id / "manifest.json"
    if not emb_path.exists():
        save_embedding_manifest(EmbeddingManifest(
            manifest_id=chain.embedding_id,
            created_at=now,
            git_sha="legacy",
            seed=42,
            relation="CtD",
            max_entities=0,
            method="legacy-unknown",
            dim=0,
            epochs=0,
            artifacts={},
            config_files={},
        ), root)

    fp_path = root / "runs" / chain.feature_pipeline_id / "manifest.json"
    if not fp_path.exists():
        save_feature_pipeline_manifest(FeaturePipelineManifest(
            manifest_id=chain.feature_pipeline_id,
            parent_embedding=chain.embedding_id,
            created_at=now,
            qml_dim=0,
            mode="classical_only",
            artifacts={},
        ), root)

    model_path = root / "runs" / chain.model_id / "manifest.json"
    if not model_path.exists():
        save_model_manifest(ModelManifest(
            manifest_id=chain.model_id,
            kind="classical",
            parent_feature_pipeline=chain.feature_pipeline_id,
            created_at=now,
            model_type="legacy-unknown",
            quantum_execution_mode_at_train=QuantumMode.UNAVAILABLE,
            artifacts={},
        ), root)

    return chain


def existing_evaluation_ids(path: Path) -> set[str]:
    """Read evaluation_ids already present so re-runs are idempotent."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = payload.get("evaluation_id")
            if eid:
                ids.add(eid)
    return ids


def row_to_evaluation(
    row: dict,
    chain: ManifestChain,
    tenant_id: str,
) -> Optional[EvaluationRecord]:
    """Convert a CSV row to EvaluationRecord. Returns None if no metrics found.

    Stable evaluation_id derived from row content so re-running gives the same id.
    """
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        v = row.get(key)
        if v in (None, "", "nan", "NaN", "NA"):
            continue
        try:
            metrics[key] = float(v)
        except (ValueError, TypeError):
            continue
    if not metrics:
        return None

    timestamp_str = row.get("timestamp") or row.get("ts") or row.get("run_timestamp_utc") or ""
    try:
        created_at = float(timestamp_str)
    except (ValueError, TypeError):
        # Try ISO-8601 (run_timestamp_utc is in that form)
        try:
            from datetime import datetime
            created_at = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            created_at = 0.0

    cv_folds_raw = row.get("cv_folds") or row.get("k") or ""
    try:
        cv_folds = int(cv_folds_raw) if cv_folds_raw else None
    except ValueError:
        cv_folds = None

    raw_id_source = (
        row.get("run_id")
        or row.get("experiment_id")
        or json.dumps(row, sort_keys=True)
    )
    eval_id = "LEGACY-" + hashlib.sha256(raw_id_source.encode("utf-8")).hexdigest()[:16]

    return EvaluationRecord(
        evaluation_id=eval_id,
        tenant_id=tenant_id,
        manifest_chain=chain,
        created_at=created_at,
        test_set_hash="legacy-unknown",
        metrics=metrics,
        cv_folds=cv_folds,
        notes=f"migrated from {row.get('experiment_name', 'experiment_history.csv')}",
    )


def migrate(
    csv_path: Path,
    root: Path,
    tenant_id: str,
    legacy_manifest_id: str,
    dry_run: bool,
) -> dict:
    if not csv_path.exists():
        log.error("CSV file not found: %s", csv_path)
        return {"written": 0, "skipped": 0, "malformed": 0, "no_metrics": 0,
                "error": f"missing {csv_path}"}

    chain = synthesize_legacy_chain(root, legacy_manifest_id)
    out_path = evaluation_path(root, tenant_id)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = existing_evaluation_ids(out_path) if not dry_run else set()

    written = skipped = malformed = no_metrics = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out_handle = open(out_path, "a", encoding="utf-8") if not dry_run else None
        try:
            for row in reader:
                try:
                    rec = row_to_evaluation(row, chain, tenant_id)
                except Exception as e:
                    malformed += 1
                    log.warning("malformed row skipped: %s", e)
                    continue
                if rec is None:
                    no_metrics += 1
                    continue
                if rec.evaluation_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(rec.evaluation_id)
                payload = {"schema_version": SCHEMA_VERSION, **rec.model_dump()}
                line = json.dumps(payload, sort_keys=True)
                if dry_run:
                    print(line)
                else:
                    assert out_handle is not None
                    out_handle.write(line + "\n")
                written += 1
        finally:
            if out_handle is not None:
                out_handle.close()

    return {"written": written, "skipped": skipped,
            "malformed": malformed, "no_metrics": no_metrics}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path,
                        help="Service artifacts directory")
    parser.add_argument("--tenant-id", default=SYSTEM_LEGACY_TENANT_ID)
    parser.add_argument("--legacy-manifest-id", default="LEGACY-PRE-REBUILD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    summary = migrate(
        csv_path=args.csv,
        root=args.root,
        tenant_id=args.tenant_id,
        legacy_manifest_id=args.legacy_manifest_id,
        dry_run=args.dry_run,
    )
    log.info("migration summary: %s", json.dumps(summary))
    if "error" in summary:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
