from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/prepare_gdc_tcga_counts_cohort.py")
SPEC = importlib.util.spec_from_file_location("prepare_gdc_tcga_counts_cohort", SCRIPT_PATH)
cohort = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["prepare_gdc_tcga_counts_cohort"] = cohort
SPEC.loader.exec_module(cohort)


def _write_star_counts(path: Path, scale: int) -> None:
    path.write_text(
        "# gene-model: GENCODE v36\n"
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
        "N_unmapped\t\t\t10\t10\t10\t\t\t\n"
        f"ENSG000001.1\tGENE1\tprotein_coding\t{10 * scale}\t0\t0\t0\t0\t0\n"
        f"ENSG000002.4\tGENE2\tprotein_coding\t{20 * scale}\t0\t0\t0\t0\t0\n"
        f"ENSG000003.2\tMIRTEST\tmiRNA\t{30 * scale}\t0\t0\t0\t0\t0\n",
        encoding="utf-8",
    )


def test_select_balanced_cohort_is_unique_and_deterministic() -> None:
    records = [
        {"case_submitter_id": "CASE-B", "sample_type": "Primary Tumor", "file_id": "f2", "file_name": "b.tsv", "project_id": "TCGA-X"},
        {"case_submitter_id": "CASE-A", "sample_type": "Primary Tumor", "file_id": "f1", "file_name": "a.tsv", "project_id": "TCGA-X"},
        {"case_submitter_id": "CASE-A", "sample_type": "Primary Tumor", "file_id": "f3", "file_name": "a2.tsv", "project_id": "TCGA-X"},
        {"case_submitter_id": "CTRL-B", "sample_type": "Solid Tissue Normal", "file_id": "c2", "file_name": "cb.tsv", "project_id": "TCGA-X"},
        {"case_submitter_id": "CTRL-A", "sample_type": "Solid Tissue Normal", "file_id": "c1", "file_name": "ca.tsv", "project_id": "TCGA-X"},
    ]

    selected = cohort.select_balanced_cohort(
        records,
        case_sample_type="Primary Tumor",
        control_sample_type="Solid Tissue Normal",
        n_case=2,
        n_control=2,
    )

    assert [row["sample_id"] for row in selected] == ["CTRL-A", "CTRL-B", "CASE-A", "CASE-B"]
    assert [row["condition"] for row in selected] == ["control", "control", "disease", "disease"]
    assert len({row["sample_id"] for row in selected}) == 4


def test_parse_and_build_star_count_matrix(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_star_counts(raw / "case.tsv", 2)
    _write_star_counts(raw / "control.tsv", 1)
    selected = [
        {
            "sample_id": "CASE",
            "condition": "disease",
            "file_name": "case.tsv",
            "file_id": "f1",
            "case_submitter_id": "CASE",
            "sample_type": "Primary Tumor",
            "project_id": "TCGA-X",
        },
        {
            "sample_id": "CTRL",
            "condition": "control",
            "file_name": "control.tsv",
            "file_id": "f2",
            "case_submitter_id": "CTRL",
            "sample_type": "Solid Tissue Normal",
            "project_id": "TCGA-X",
        },
    ]

    counts, gene_map = cohort.build_count_matrix(
        selected,
        raw,
        count_column="unstranded",
        gene_types=["protein_coding"],
        min_total_count=1,
    )

    assert counts["gene"].tolist() == ["ENSG000001", "ENSG000002"]
    assert counts["CASE"].tolist() == [20, 40]
    assert counts["CTRL"].tolist() == [10, 20]
    assert gene_map["gene_symbol"].tolist() == ["GENE1", "GENE2"]
