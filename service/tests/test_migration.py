"""CSV → JSONL migration: idempotent, malformed-tolerant, manifest synthesized."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from service.persistence import (
    SCHEMA_VERSION,
    evaluation_path,
    load_embedding_manifest,
    load_feature_pipeline_manifest,
    load_model_manifest,
)
from service.scripts.migrate_history import migrate, synthesize_legacy_chain


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_synthesize_legacy_chain_creates_three_manifests(artifacts_dir):
    chain = synthesize_legacy_chain(artifacts_dir, "LEGACY-X")
    assert chain.embedding_id == "LEGACY-X-EMB"
    assert chain.feature_pipeline_id == "LEGACY-X-FP"
    assert chain.model_id == "LEGACY-X-MODEL"
    e = load_embedding_manifest(artifacts_dir, chain.embedding_id)
    f = load_feature_pipeline_manifest(artifacts_dir, chain.feature_pipeline_id)
    m = load_model_manifest(artifacts_dir, chain.model_id)
    assert e.method == "legacy-unknown"
    assert f.parent_embedding == chain.embedding_id
    assert m.parent_feature_pipeline == chain.feature_pipeline_id


def test_synthesize_is_idempotent(artifacts_dir):
    chain1 = synthesize_legacy_chain(artifacts_dir, "LEGACY-X")
    chain2 = synthesize_legacy_chain(artifacts_dir, "LEGACY-X")
    assert chain1 == chain2


def test_migrate_writes_jsonl_and_returns_summary(tmp_path, artifacts_dir):
    csv_path = tmp_path / "history.csv"
    _write_csv(csv_path, [
        {"run_id": "r1", "experiment_name": "exp1",
         "timestamp": "1700000000", "pr_auc": "0.82", "roc_auc": "0.88",
         "f1": "0.75"},
        {"run_id": "r2", "experiment_name": "exp1",
         "timestamp": "1700001000", "pr_auc": "0.79", "f1": "0.72"},
    ])
    summary = migrate(csv_path, artifacts_dir, "legacy", "LEGACY-X", dry_run=False)
    assert summary["written"] == 2
    assert summary["skipped"] == 0

    out_path = evaluation_path(artifacts_dir, "legacy")
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["evaluation_id"].startswith("LEGACY-")
    assert payload["tenant_id"] == "legacy"
    assert payload["manifest_chain"]["model_id"] == "LEGACY-X-MODEL"


def test_migrate_idempotent(tmp_path, artifacts_dir):
    csv_path = tmp_path / "history.csv"
    _write_csv(csv_path, [
        {"run_id": "r1", "pr_auc": "0.82"},
    ])
    s1 = migrate(csv_path, artifacts_dir, "legacy", "LEGACY-X", dry_run=False)
    s2 = migrate(csv_path, artifacts_dir, "legacy", "LEGACY-X", dry_run=False)
    assert s1["written"] == 1 and s1["skipped"] == 0
    assert s2["written"] == 0 and s2["skipped"] == 1


def test_migrate_skips_rows_without_metrics(tmp_path, artifacts_dir):
    csv_path = tmp_path / "history.csv"
    _write_csv(csv_path, [
        {"run_id": "r1", "experiment_name": "no_metrics_just_metadata"},
        {"run_id": "r2", "pr_auc": "0.81"},
    ])
    summary = migrate(csv_path, artifacts_dir, "legacy", "LEGACY-X", dry_run=False)
    assert summary["written"] == 1
    assert summary["no_metrics"] == 1


def test_migrate_handles_malformed_metric_values(tmp_path, artifacts_dir):
    csv_path = tmp_path / "history.csv"
    _write_csv(csv_path, [
        {"run_id": "r1", "pr_auc": "not_a_number", "f1": "0.7"},
    ])
    summary = migrate(csv_path, artifacts_dir, "legacy", "LEGACY-X", dry_run=False)
    # f1 is parseable, so the row produces an EvaluationRecord with just f1
    assert summary["written"] == 1


def test_migrate_dry_run_writes_nothing(tmp_path, artifacts_dir, capsys):
    csv_path = tmp_path / "history.csv"
    _write_csv(csv_path, [{"run_id": "r1", "pr_auc": "0.81"}])
    summary = migrate(csv_path, artifacts_dir, "legacy", "LEGACY-X", dry_run=True)
    assert summary["written"] == 1
    out_path = evaluation_path(artifacts_dir, "legacy")
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "evaluation_id" in captured.out


def test_migrate_missing_csv_returns_error(tmp_path, artifacts_dir):
    summary = migrate(tmp_path / "nope.csv", artifacts_dir, "legacy", "LEGACY-X",
                      dry_run=False)
    assert "error" in summary
