# TCGA-BRCA Counts Cohort

This workflow creates a larger development bulk RNA-seq cohort from open NCI
Genomic Data Commons files. It requires no login, token, or paid service.

## Cohort Definition

- Project: `TCGA-BRCA`
- Workflow: `STAR - Counts`
- Count field: `unstranded`
- Case group: `Primary Tumor`
- Control group: `Solid Tissue Normal`
- Samples: 30 case and 30 control
- Genes: protein-coding genes with at least 10 total reads across the cohort
- Selection: deterministic sorting by case submitter ID and file ID, with one
  file per unique case

The preparation manifest records every GDC file ID and selected case:

`artifacts/external/gdc_tcga_brca/converted/cohort_manifest.json`

## Prepare Counts

```bash
.venv/bin/python scripts/prepare_gdc_tcga_counts_cohort.py \
  --project-id TCGA-BRCA \
  --n-case 30 \
  --n-control 30 \
  --out-dir artifacts/external/gdc_tcga_brca \
  --workers 6 \
  --gene-types protein_coding \
  --min-total-count 10
```

Expected converted dimensions:

```text
samples: 60
disease: 30
control: 30
genes: 18,624
```

## Differential Expression

Use PyDESeq2 on raw integer counts. The `simple` DE method is intended only for
fixtures and smoke tests.

```bash
.venv/bin/python scripts/run_rnaseq_counts_pipeline.py \
  --input artifacts/external/gdc_tcga_brca/converted/tcga_brca_counts.csv \
  --format count-matrix \
  --metadata artifacts/external/gdc_tcga_brca/converted/tcga_brca_metadata.csv \
  --out-dir artifacts/single_cell/tcga_brca_60 \
  --signatures-dir artifacts/signatures/tcga_brca_60 \
  --case-label disease \
  --control-label control \
  --disease-id Disease::Breast_invasive_carcinoma \
  --tissue breast \
  --top-n 250 \
  --lfc-threshold 0.5 \
  --padj-threshold 0.05 \
  --de-method pydeseq2 \
  --de-min-total-count 10 \
  --de-n-cpus 4
```

Observed PyDESeq2 results:

```text
tested genes: 18,624
adjusted p <= 0.05: 10,677
adjusted p <= 0.05 and |log2FC| >= 0.5: 8,581
```

## Leakage-safe Quantum Benchmark

Classifier feature selection starts from all 18,624 measured genes and occurs
inside each training fold. The cohort-wide PyDESeq2 signature is used only for
CREEDS perturbation ranking. Restricting classifier inputs to that signature
before cross-validation leaks outcome information and invalidates classifier
metrics.

The 32-gene setting is also the smallest tested ranking signature size that
gives at least 50 human CREEDS profiles with three or more overlapping genes.

```bash
.venv/bin/python scripts/run_rnaseq_quantum_benchmark.py \
  --normalized-counts artifacts/single_cell/tcga_brca_gse225846_harmonized/development_normalized_counts.csv \
  --metadata artifacts/external/gdc_tcga_brca/converted/tcga_brca_metadata.csv \
  --signature artifacts/signatures/tcga_brca_60/disease_signature.json \
  --creeds-signatures artifacts/external/creeds/single_drug_perturbations-v1.0.json \
  --gene-map artifacts/external/gdc_tcga_brca/converted/tcga_brca_gene_map.csv \
  --out-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized \
  --case-label disease \
  --control-label control \
  --top-genes 32 \
  --qml-dims 2,3,4 \
  --qsvc-reps-list 1,2 \
  --min-profile-gene-overlap 3 \
  --max-ranking-profiles 100 \
  --bootstrap 1000 \
  --permutations 1000 \
  --full-permutations 100 \
  --full-permutation-qml-dims 2 \
  --full-permutation-qsvc-reps-list 1
```

## Evidence Audit

```bash
.venv/bin/python scripts/audit_rnaseq_quantum_evidence.py \
  --benchmark-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized \
  --external-validation artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/external_validation.json
```

Current result:

```text
readiness: review_ready
classifier sample gate: pass
full retraining permutation gate: pass
real ranking evidence gate: pass
classical classifier signal gate: pass
independent external validation gate: pass
external statistical evidence gate: pass
external quantum value claim gate: pass
```

Both classical and quantum classifiers have significant signal under the full
retraining permutation null. The tested QSVC matches classical ROC-AUC but has
lower balanced accuracy. Quantum scores do not materially alter the CREEDS
ranking, so this run does not support a quantum-advantage claim.

TCGA tumor-versus-adjacent-normal classification is a strong but easy technical
benchmark, and tissue composition or collection differences can contribute to
separation. The independent GSE225846 validation is documented in
`docs/guides/GSE225846_EXTERNAL_VALIDATION.md`. It supports cross-study
discrimination but shows that QSVC underperforms logistic regression.

The earlier `artifacts/benchmarks/rnaseq_quantum_tcga_brca_60` classifier result
is superseded because its classifier feature universe was restricted by a
cohort-wide signature before cross-validation. Do not cite that artifact.
