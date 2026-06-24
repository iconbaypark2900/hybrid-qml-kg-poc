from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from single_cell_layer.count_matrix import (
    CountMatrixValidationError,
    attach_metadata,
    compute_qc_summary,
    load_count_matrix,
    normalize_total_log1p,
    run_pydeseq2_de,
    run_simple_de,
)


FIXTURE_DIR = Path("tests/fixtures/rnaseq_counts")


def test_count_matrix_loader_validates_matching_metadata_ids(tmp_path: Path) -> None:
    bad_metadata = tmp_path / "metadata.csv"
    bad_metadata.write_text(
        "sample_id,condition\ncase_a,disease\nmissing_sample,control\n",
        encoding="utf-8",
    )

    with pytest.raises(CountMatrixValidationError, match="exactly match"):
        load_count_matrix(FIXTURE_DIR / "counts.csv", bad_metadata)


def test_count_matrix_loader_requires_case_and_control_labels(tmp_path: Path) -> None:
    bad_metadata = tmp_path / "metadata.csv"
    bad_metadata.write_text(
        "sample_id,condition\ncase_a,disease\ncase_b,disease\ncontrol_a,disease\ncontrol_b,disease\n",
        encoding="utf-8",
    )

    with pytest.raises(CountMatrixValidationError, match="Missing"):
        load_count_matrix(FIXTURE_DIR / "counts.csv", bad_metadata)


def test_count_matrix_loader_and_de_are_deterministic() -> None:
    adata = load_count_matrix(FIXTURE_DIR / "counts.csv", FIXTURE_DIR / "metadata.csv")
    normalize_total_log1p(adata)
    summary = compute_qc_summary(adata)
    de = run_simple_de(adata)

    assert summary["n_cells"] == 4
    assert summary["n_genes"] == 6
    assert summary["condition_counts"] == {"control": 2, "disease": 2}
    assert de.iloc[0]["names"] in {"GENE_CASE", "GENE_CTRL"}
    assert {"names", "scores", "logfoldchanges", "pvals_adj"}.issubset(de.columns)


def test_attach_metadata_requires_exact_observation_match(tmp_path: Path) -> None:
    adata = load_count_matrix(FIXTURE_DIR / "counts.csv", FIXTURE_DIR / "metadata.csv")
    bad_metadata = tmp_path / "metadata.csv"
    bad_metadata.write_text(
        "sample_id,condition\ncase_a,disease\ncase_b,disease\ncontrol_a,control\nextra,control\n",
        encoding="utf-8",
    )

    with pytest.raises(CountMatrixValidationError, match="exactly match"):
        attach_metadata(adata, bad_metadata)


def test_pydeseq2_de_uses_raw_counts_and_real_adjusted_pvalues(tmp_path: Path) -> None:
    sample_ids = [f"case_{idx}" for idx in range(8)] + [f"control_{idx}" for idx in range(8)]
    count_rows = [
        ["GENE_CASE", *([500, 520, 510, 530, 490, 515, 505, 525] + [20, 18, 22, 19, 21, 17, 23, 20])],
        ["GENE_CTRL", *([25, 22, 27, 24, 26, 23, 28, 25] + [450, 470, 460, 480, 440, 465, 455, 475])],
        ["GENE_STABLE", *([100] * 16)],
        ["GENE_LOW", *([0] * 15 + [1])],
    ]
    counts_path = tmp_path / "counts.csv"
    pd.DataFrame(count_rows, columns=["gene", *sample_ids]).to_csv(counts_path, index=False)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "condition": ["disease"] * 8 + ["control"] * 8,
        }
    ).to_csv(metadata_path, index=False)

    adata = load_count_matrix(counts_path, metadata_path)
    de = run_pydeseq2_de(adata, min_total_count=10, n_cpus=1)

    assert set(de["names"]) == {"GENE_CASE", "GENE_CTRL", "GENE_STABLE"}
    assert set(de["de_method"]) == {"pydeseq2_wald"}
    assert de["pvals_adj"].notna().any()
    assert de.iloc[0]["names"] in {"GENE_CASE", "GENE_CTRL"}


def test_rnaseq_counts_cli_exports_expected_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "single_cell"
    signatures_dir = tmp_path / "signatures"
    cmd = [
        sys.executable,
        "scripts/run_rnaseq_counts_pipeline.py",
        "--input",
        str(FIXTURE_DIR / "counts.csv"),
        "--format",
        "count-matrix",
        "--metadata",
        str(FIXTURE_DIR / "metadata.csv"),
        "--out-dir",
        str(out_dir),
        "--signatures-dir",
        str(signatures_dir),
        "--disease-id",
        "Disease::TEST",
        "--top-n",
        "3",
    ]

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    outputs = json.loads(completed.stdout)

    expected_paths = [
        out_dir / "qc" / "qc_summary_table.csv",
        out_dir / "qc" / "qc_summary_table.md",
        out_dir / "normalized_counts.csv",
        out_dir / "differential_expression.csv",
        out_dir / "rnaseq_counts_manifest.json",
        signatures_dir / "disease_signature.json",
        signatures_dir / "cell_type_signatures.json",
    ]
    for path in expected_paths:
        assert path.exists(), path

    signature = json.loads((signatures_dir / "disease_signature.json").read_text(encoding="utf-8"))
    assert signature["disease"] == "Disease::TEST"
    assert len(signature["ranked_genes"]) > 0
    assert outputs["disease_signature"].endswith("disease_signature.json")

    qc = pd.read_csv(out_dir / "qc" / "qc_summary_table.csv")
    assert "n_cells" in set(qc["metric"])


def test_rnaseq_counts_cli_requires_metadata_for_10x() -> None:
    cmd = [
        sys.executable,
        "scripts/run_rnaseq_counts_pipeline.py",
        "--input",
        "missing_10x_dir",
        "--format",
        "10x",
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "--metadata is required for --format 10x" in completed.stderr


def test_rnaseq_counts_cli_can_normalize_without_deriving_validation_signature(tmp_path: Path) -> None:
    out_dir = tmp_path / "validation"
    signatures_dir = tmp_path / "signatures"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_rnaseq_counts_pipeline.py",
            "--input",
            str(FIXTURE_DIR / "counts.csv"),
            "--format",
            "count-matrix",
            "--metadata",
            str(FIXTURE_DIR / "metadata.csv"),
            "--out-dir",
            str(out_dir),
            "--signatures-dir",
            str(signatures_dir),
            "--skip-de",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    outputs = json.loads(completed.stdout)
    assert outputs["de_skipped"] is True
    assert (out_dir / "normalized_counts.csv").exists()
    assert not (out_dir / "differential_expression.csv").exists()
    assert not (signatures_dir / "disease_signature.json").exists()
