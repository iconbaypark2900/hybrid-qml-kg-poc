# GSE225846 Independent RNA-seq Validation

This workflow validates models developed on `TCGA-BRCA` against the independent
NCI-Maryland breast tumor/normal cohort `GSE225846`. It uses only open NCBI GEO
files and requires no account or token.

## Design

- Development: 60 TCGA-BRCA samples, 30 primary tumor and 30 adjacent normal.
- Validation: 155 GSE225846 samples, 80 tumor and 75 normal.
- Validation patients: 83, including 72 matched tumor/normal pairs.
- Endpoint: bulk breast tumor versus normal tissue.
- Model selection, feature selection, scaling, PCA, and fitting use TCGA only.
- External uncertainty uses patient-cluster bootstrap resampling.
- External null tests swap labels within matched pairs and permute labels among
  unmatched samples.

This is a cross-study technical validation. It does not establish screening
performance in patients, clinical utility, causal biomarkers, or drug efficacy.

## Prepare GEO Counts

```bash
.venv/bin/python scripts/prepare_geo_gse225846_counts_cohort.py \
  --out-dir artifacts/external/geo_gse225846
```

The converter downloads the deposited STAR/RSEM matrix and GEO SOFT metadata,
maps count columns to GSM accessions, preserves patient pairing and clinical
annotations, strips Ensembl versions, and records SHA-256 hashes.

The deposited RSEM expected counts are not fully integer-valued: 521,181 of
9,124,850 retained matrix values are fractional. They are preserved rather than
rounded. This cohort is valid for library-size normalization and external
prediction but is not passed to the integer-count PyDESeq2 path.

## Harmonize Before Normalization

TCGA and GEO use different annotation releases. Intersect their raw count
matrices first, then normalize both over the identical ordered gene universe:

```bash
.venv/bin/python scripts/harmonize_rnaseq_cohorts.py \
  --development-counts artifacts/external/gdc_tcga_brca/converted/tcga_brca_counts.csv \
  --validation-counts artifacts/external/geo_gse225846/converted/gse225846_counts.csv \
  --development-cohort TCGA-BRCA \
  --validation-cohort GSE225846 \
  --out-dir artifacts/single_cell/tcga_brca_gse225846_harmonized
```

The harmonization manifest records genes missing from either annotation,
library totals, input hashes, the exact shared universe, and confirms that no
labels were used. The observed shared universe contains 18,255 genes; 369 TCGA
genes are absent from the older GEO annotation.

## Optional QC Without Validation DE

```bash
.venv/bin/python scripts/run_rnaseq_counts_pipeline.py \
  --input artifacts/external/geo_gse225846/converted/gse225846_counts.csv \
  --format count-matrix \
  --metadata artifacts/external/geo_gse225846/converted/gse225846_metadata.csv \
  --out-dir artifacts/single_cell/gse225846_external \
  --signatures-dir artifacts/signatures/gse225846_external \
  --case-label disease \
  --control-label control \
  --skip-de
```

`--skip-de` is required for this protocol. It prevents a validation-derived
signature from entering the model workflow. The locked model uses the
harmonized matrices above, not this optional standalone normalized matrix.

## Run Locked External Validation

First complete the leakage-safe TCGA command in
`docs/guides/TCGA_BRCA_RNASEQ_COHORT.md`. Then run:

```bash
.venv/bin/python scripts/run_rnaseq_external_validation.py \
  --development-counts artifacts/single_cell/tcga_brca_gse225846_harmonized/development_normalized_counts.csv \
  --development-metadata artifacts/external/gdc_tcga_brca/converted/tcga_brca_metadata.csv \
  --development-verdict artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/quantum_value_verdict.json \
  --development-manifest artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/benchmark_manifest.json \
  --harmonization-manifest artifacts/single_cell/tcga_brca_gse225846_harmonized/harmonization_manifest.json \
  --validation-counts artifacts/single_cell/tcga_brca_gse225846_harmonized/validation_normalized_counts.csv \
  --validation-metadata artifacts/external/geo_gse225846/converted/gse225846_metadata.csv \
  --development-cohort TCGA-BRCA \
  --validation-cohort GSE225846 \
  --out-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external \
  --top-genes 32 \
  --bootstrap 2000 \
  --permutations 10000
```

Observed harmonized locked-model results:

| Metric | Logistic regression | QSVC |
|---|---:|---:|
| ROC-AUC | 0.9730 | 0.9223 |
| Balanced accuracy | 0.9163 | 0.8288 |

Quantum-minus-classical ROC-AUC is `-0.0507`; the patient-cluster 95% interval
is `[-0.0826, -0.0245]`. Exact McNemar testing gives `p=0.0072`, with 17
samples correct only for logistic regression and four correct only for QSVC.
Both models discriminate externally under 10,000 pair-aware permutations
(`p=0.0001`), but the data do not support quantum advantage.

## Audit

```bash
.venv/bin/python scripts/audit_rnaseq_quantum_evidence.py \
  --benchmark-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized \
  --external-validation artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/external_validation.json
```

The harmonized evidence bundle passes all ten configured technical
external-review gates. The allowed conclusion is narrow: both locked models
generalize for this cross-study tumor-versus-normal endpoint, and the tested
QSVC underperforms logistic regression. Do not describe this as quantum
advantage or clinical validation.

Verify hashes, paths, dimensions, shared genes, leakage controls, outputs, audit
gates, and claim consistency:

```bash
.venv/bin/python scripts/verify_rnaseq_evidence_bundle.py \
  --harmonization-manifest artifacts/single_cell/tcga_brca_gse225846_harmonized/harmonization_manifest.json \
  --benchmark-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized \
  --external-validation artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/external_validation.json \
  --out artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/evidence_bundle_verification.json
```
