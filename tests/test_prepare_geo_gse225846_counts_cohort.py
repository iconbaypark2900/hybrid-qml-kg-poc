from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/prepare_geo_gse225846_counts_cohort.py")
SPEC = importlib.util.spec_from_file_location("prepare_geo_gse225846_counts_cohort", SCRIPT_PATH)
cohort = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["prepare_geo_gse225846_counts_cohort"] = cohort
SPEC.loader.exec_module(cohort)


def test_geo_parser_maps_soft_samples_and_preserves_fractional_rsem_counts(tmp_path: Path) -> None:
    soft_path = tmp_path / "family.soft.gz"
    soft_text = (
        "^SAMPLE = GSM_CASE\n"
        "!Sample_title = S_case_column\n"
        "!Sample_characteristics_ch1 = accession: PATIENT_1\n"
        "!Sample_characteristics_ch1 = type: tumor\n"
        "!Sample_characteristics_ch1 = race: Test\n"
        "^SAMPLE = GSM_CONTROL\n"
        "!Sample_title = S_control_column\n"
        "!Sample_characteristics_ch1 = accession: PATIENT_1\n"
        "!Sample_characteristics_ch1 = type: normal\n"
        "!Sample_characteristics_ch1 = race: Test\n"
    )
    with gzip.open(soft_path, "wt", encoding="utf-8") as handle:
        handle.write(soft_text)

    counts_path = tmp_path / "counts.txt.gz"
    counts_text = (
        '"gene_id"\t"case_column"\t"control_column"\n'
        '"ENSG000001.2_GENE1"\t10.5\t2\n'
        '"ENSG000002.1_GENE2"\t1\t9.25\n'
    )
    with gzip.open(counts_path, "wt", encoding="utf-8") as handle:
        handle.write(counts_text)

    metadata = cohort.parse_soft_samples(soft_path)
    counts, gene_map, diagnostics = cohort.parse_counts(counts_path, metadata, min_total_count=0)

    assert metadata["sample_id"].tolist() == ["GSM_CASE", "GSM_CONTROL"]
    assert metadata["condition"].tolist() == ["disease", "control"]
    assert metadata["patient_id"].nunique() == 1
    assert counts.columns.tolist() == ["gene", "GSM_CASE", "GSM_CONTROL"]
    assert counts.loc[counts["gene"] == "ENSG000001", "GSM_CASE"].iloc[0] == 10.5
    assert gene_map.loc[gene_map["gene"] == "ENSG000002", "gene_symbol"].iloc[0] == "GENE2"
    assert diagnostics["integer_valued"] is False
    assert diagnostics["n_fractional_values"] == 2

    allowlist_path = tmp_path / "allowlist.csv"
    allowlist_path.write_text("gene_id\nENSG000002.1\n", encoding="utf-8")
    allowlist = cohort.load_gene_allowlist(allowlist_path)
    restricted, _, _ = cohort.parse_counts(
        counts_path,
        metadata,
        min_total_count=1000,
        gene_allowlist=allowlist,
    )
    assert restricted["gene"].tolist() == ["ENSG000002"]
