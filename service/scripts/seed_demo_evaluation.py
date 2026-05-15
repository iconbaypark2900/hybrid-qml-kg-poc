"""Append a demo EvaluationRecord against the active classical chain.

This is a tool for partner demos so the Experiment page has something to
render. The recorded metrics are computed from the live model on a tiny
synthetic test set — they're real numbers, but real-looking-not-real-research.

The notes field flags this clearly; the manifest_chain.model_id is the real
active classical model.

Usage:
    python -m service.scripts.seed_demo_evaluation --tenant-id demo
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from service.persistence import (  # noqa: E402
    append_evaluation,
    load_active_manifest_chain,
)
from service.schemas import EvaluationRecord  # noqa: E402


def _compute_demo_metrics(root: Path, chain) -> dict[str, float]:
    """Run the active classical model on a tiny synthetic test set so the
    recorded numbers are at least real model outputs, not fabricated."""
    import joblib

    model_path = root / "runs" / chain.model_id / "model.joblib"
    scaler_path = root / "runs" / chain.feature_pipeline_id / "scaler.joblib"
    if not model_path.exists():
        return {"pr_auc": 0.0, "f1": 0.0, "note_no_real_eval": 1.0}
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    n_features = (
        getattr(scaler, "n_features_in_", None)
        or getattr(model, "n_features_in_", 48)
    )

    # Synthetic balanced binary task — same shape as training
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, n_features))
    y = (X.sum(axis=1) > 0).astype(int)
    X_in = scaler.transform(X) if scaler is not None else X

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_in))[:, 1]
    else:
        proba = np.asarray(model.predict(X_in)).astype(float)
    pred = (proba >= 0.5).astype(int)

    # Compute metrics from sklearn — same path the real benchmarking uses
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )
    metrics = {
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    return metrics


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--tenant-id", default="demo")
    p.add_argument("--cv-folds", type=int, default=5)
    args = p.parse_args(argv)

    chain = load_active_manifest_chain(args.root)
    if chain is None:
        print("no active classical chain; run synthesize_manifest_chain first", file=sys.stderr)
        return 1

    metrics = _compute_demo_metrics(args.root, chain)

    # Stable evaluation_id — re-running this script produces the same id
    # because the inputs (chain + test seed) are deterministic. Idempotent.
    eid_input = f"demo|{chain.model_id}|{json.dumps(metrics, sort_keys=True)}"
    eid = "DEMO-" + hashlib.sha256(eid_input.encode()).hexdigest()[:16]

    rec = EvaluationRecord(
        evaluation_id=eid,
        tenant_id=args.tenant_id,
        manifest_chain=chain,
        created_at=time.time(),
        test_set_hash="synthetic-demo-seed-0",
        metrics=metrics,
        cv_folds=args.cv_folds,
        notes=(
            "Demo evaluation seeded by seed_demo_evaluation.py. Metrics are "
            "computed from the active classical model on a 64-sample synthetic "
            "balanced binary test set — real model outputs, not real research. "
            "Replace with metrics from a real held-out Hetionet test set "
            "before any partner-facing claim."
        ),
    )
    asyncio.run(append_evaluation(rec, args.root))

    print(json.dumps({
        "ok": True,
        "evaluation_id": rec.evaluation_id,
        "tenant_id": rec.tenant_id,
        "model_id": chain.model_id,
        "metrics": metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
