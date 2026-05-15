"""Manifest DAG round-trip + LATEST.txt + JSONL evaluations + corruption handling."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from service.persistence import (
    SCHEMA_VERSION,
    append_evaluation,
    evaluation_path,
    list_evaluations,
    load_active_manifest_chain,
    load_embedding_manifest,
    load_feature_pipeline_manifest,
    load_model_manifest,
    save_embedding_manifest,
    save_feature_pipeline_manifest,
    save_model_manifest,
    set_active_model,
)
from service.schemas import (
    EmbeddingManifest,
    EvaluationRecord,
    FeaturePipelineManifest,
    ManifestChain,
    ModelManifest,
    QuantumMode,
)


def _make_chain(root: Path) -> ManifestChain:
    emb = EmbeddingManifest(
        manifest_id="emb-1", created_at=time.time(),
        git_sha="abc", seed=42, relation="CtD",
        max_entities=0, method="TransE", dim=32, epochs=200,
        artifacts={}, config_files={"kg_layer_config.yaml": "deadbeef"},
    )
    save_embedding_manifest(emb, root)
    fp = FeaturePipelineManifest(
        manifest_id="fp-1", parent_embedding="emb-1",
        created_at=time.time(), qml_dim=4, mode="diff", artifacts={},
    )
    save_feature_pipeline_manifest(fp, root)
    mdl = ModelManifest(
        manifest_id="mdl-1", kind="classical",
        parent_feature_pipeline="fp-1", created_at=time.time(),
        model_type="logistic_regression",
        quantum_execution_mode_at_train=None, artifacts={},
    )
    save_model_manifest(mdl, root)
    set_active_model(root, "mdl-1")
    return ManifestChain(embedding_id="emb-1", feature_pipeline_id="fp-1", model_id="mdl-1")


def test_manifest_round_trip(artifacts_dir):
    chain = _make_chain(artifacts_dir)
    e = load_embedding_manifest(artifacts_dir, chain.embedding_id)
    assert e.method == "TransE" and e.dim == 32
    f = load_feature_pipeline_manifest(artifacts_dir, chain.feature_pipeline_id)
    assert f.parent_embedding == "emb-1" and f.qml_dim == 4
    m = load_model_manifest(artifacts_dir, chain.model_id)
    assert m.kind == "classical" and m.parent_feature_pipeline == "fp-1"


def test_load_active_manifest_chain(artifacts_dir):
    _make_chain(artifacts_dir)
    chain = load_active_manifest_chain(artifacts_dir)
    assert chain is not None
    assert chain.model_id == "mdl-1"
    assert chain.feature_pipeline_id == "fp-1"
    assert chain.embedding_id == "emb-1"


def test_missing_latest_returns_none(artifacts_dir):
    assert load_active_manifest_chain(artifacts_dir) is None


def test_broken_chain_returns_none_not_exception(artifacts_dir, caplog):
    # LATEST.txt points to nonexistent model
    set_active_model(artifacts_dir, "nope")
    assert load_active_manifest_chain(artifacts_dir) is None


def test_evaluation_jsonl_round_trip(artifacts_dir):
    chain = _make_chain(artifacts_dir)
    rec = EvaluationRecord(
        evaluation_id="eval-1",
        tenant_id="a",
        manifest_chain=chain,
        created_at=time.time(),
        test_set_hash="hash1",
        metrics={"pr_auc": 0.85, "roc_auc": 0.91},
        cv_folds=5,
    )
    asyncio.run(append_evaluation(rec, artifacts_dir))
    out = asyncio.run(list_evaluations(artifacts_dir, "a"))
    assert len(out) == 1
    assert out[0].evaluation_id == "eval-1"
    assert out[0].metrics["pr_auc"] == 0.85


def test_evaluation_per_tenant_isolation(artifacts_dir):
    chain = _make_chain(artifacts_dir)
    rec_a = EvaluationRecord(
        evaluation_id="eval-a", tenant_id="a", manifest_chain=chain,
        created_at=time.time(), test_set_hash="h", metrics={"pr_auc": 0.8},
    )
    rec_b = EvaluationRecord(
        evaluation_id="eval-b", tenant_id="b", manifest_chain=chain,
        created_at=time.time(), test_set_hash="h", metrics={"pr_auc": 0.7},
    )
    asyncio.run(append_evaluation(rec_a, artifacts_dir))
    asyncio.run(append_evaluation(rec_b, artifacts_dir))
    out_a = asyncio.run(list_evaluations(artifacts_dir, "a"))
    out_b = asyncio.run(list_evaluations(artifacts_dir, "b"))
    assert {r.evaluation_id for r in out_a} == {"eval-a"}
    assert {r.evaluation_id for r in out_b} == {"eval-b"}


def test_evaluation_dedup_by_id(artifacts_dir):
    chain = _make_chain(artifacts_dir)
    rec = EvaluationRecord(
        evaluation_id="eval-dup", tenant_id="a", manifest_chain=chain,
        created_at=time.time(), test_set_hash="h", metrics={"pr_auc": 0.8},
    )
    # Write the same record twice (simulates a re-run that didn't check)
    asyncio.run(append_evaluation(rec, artifacts_dir))
    asyncio.run(append_evaluation(rec, artifacts_dir))
    out = asyncio.run(list_evaluations(artifacts_dir, "a"))
    assert len(out) == 1


def test_evaluation_skips_wrong_schema_version(artifacts_dir, caplog):
    path = evaluation_path(artifacts_dir, "a")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write one row with a future schema_version, one with the current one
    chain = _make_chain(artifacts_dir)
    legit_rec = EvaluationRecord(
        evaluation_id="eval-legit", tenant_id="a", manifest_chain=chain,
        created_at=time.time(), test_set_hash="h", metrics={"pr_auc": 0.8},
    )
    legit_payload = {"schema_version": SCHEMA_VERSION, **legit_rec.model_dump()}
    bogus_payload = {"schema_version": 9999, "junk": "fields"}
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(bogus_payload) + "\n")
        f.write(json.dumps(legit_payload, sort_keys=True) + "\n")
    out = asyncio.run(list_evaluations(artifacts_dir, "a"))
    assert {r.evaluation_id for r in out} == {"eval-legit"}


def test_evaluation_skips_unparseable_lines(artifacts_dir):
    path = evaluation_path(artifacts_dir, "a")
    path.parent.mkdir(parents=True, exist_ok=True)
    chain = _make_chain(artifacts_dir)
    legit = EvaluationRecord(
        evaluation_id="eval-ok", tenant_id="a", manifest_chain=chain,
        created_at=time.time(), test_set_hash="h", metrics={"pr_auc": 0.8},
    )
    payload = {"schema_version": SCHEMA_VERSION, **legit.model_dump()}
    with path.open("w", encoding="utf-8") as f:
        f.write("{ this is not json\n")
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.write("\n")  # blank line
    out = asyncio.run(list_evaluations(artifacts_dir, "a"))
    assert {r.evaluation_id for r in out} == {"eval-ok"}


def test_one_tenant_corruption_does_not_break_another(artifacts_dir):
    """Tenant A's file is corrupt; tenant B's reads should still work."""
    chain = _make_chain(artifacts_dir)
    # Corrupt tenant A
    path_a = evaluation_path(artifacts_dir, "a")
    path_a.parent.mkdir(parents=True, exist_ok=True)
    path_a.write_text("not json\n", encoding="utf-8")
    # Healthy tenant B
    rec_b = EvaluationRecord(
        evaluation_id="eval-b", tenant_id="b", manifest_chain=chain,
        created_at=time.time(), test_set_hash="h", metrics={"pr_auc": 0.85},
    )
    asyncio.run(append_evaluation(rec_b, artifacts_dir))
    out_a = asyncio.run(list_evaluations(artifacts_dir, "a"))
    out_b = asyncio.run(list_evaluations(artifacts_dir, "b"))
    assert out_a == []
    assert {r.evaluation_id for r in out_b} == {"eval-b"}
