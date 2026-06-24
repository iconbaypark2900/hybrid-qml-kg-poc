# RNA-seq Quantum Value Benchmark

This benchmark tests whether the quantum classifier contributes measurable lift
over classical baselines for RNA-seq evidence. It is designed to avoid treating
the quantum layer as novelty: every run emits a verdict comparing the best QSVC
configuration against the best classical model.

For the reproducible 60-sample open TCGA-BRCA workflow, see
`docs/guides/TCGA_BRCA_RNASEQ_COHORT.md`.
For independent TCGA-to-GEO validation, see
`docs/guides/GSE225846_EXTERNAL_VALIDATION.md`.

Classifier benchmarking uses all measured genes as its feature universe and
selects the requested top genes inside each training fold. A cohort-wide DE
signature must not restrict classifier inputs before cross-validation. The
signature remains valid for the separate perturbation-ranking path.

## Local Airway Pilot

Run after the counts-first RNA-seq pipeline has produced the airway pilot
artifacts:

```bash
.venv/bin/python scripts/run_rnaseq_quantum_benchmark.py \
  --demo-ranking \
  --out-dir artifacts/benchmarks/rnaseq_quantum_airway \
  --top-genes 16 \
  --qml-dims 2,3,4 \
  --qsvc-reps-list 1,2
```

Key outputs:

- `classifier_metrics.csv`: classical and QSVC metrics for every tested quantum config.
- `classifier_predictions.csv`: held-out predictions per model and sample.
- `quantum_value_verdict.json`: machine-readable value verdict.
- `benchmark_report.md`: compact human-readable summary.
- `benchmark_manifest.json`: inputs, selected genes, CV mode, and quantum sweep config.

The verdict includes guardrails intended for scientific review:

- `classifier_evidence_grade`: flags tiny pilots separately from screening-scale
  or analysis-ready sample counts.
- `permutation_auc_best_quantum` and `permutation_auc_best_classical`: fixed-score
  label-permutation context for ROC-AUC. This is not a full model-retraining
  permutation test.
- `full_retraining_permutation`: optional stronger null test that shuffles
  labels before feature selection and cross-validation, then reruns the
  classifier workflow.
- `ranking_quantum_materiality`: whether quantum scores materially changed the
  KG+omics candidate order.
- `ranking_quantum_changes_top_k`: whether the top-k candidate set changed.

Current airway result:

```text
classifier_verdict: quantum_underperforms_classical
best_quantum_model: qsvc_quantum_dim2_reps1
best quantum ROC-AUC: 0.625
best classical ROC-AUC: 1.000
classifier_evidence_grade: pilot_underpowered
```

Interpretation: on the available real airway RNA-seq sample-classification
benchmark, the tested local QSVC configurations do not add value over the
classical baselines.

## Real Ranking Inputs

The ranking path is only real evidence when supplied with compound perturbation
profiles. Without those profiles, `--demo-ranking` runs a deterministic
signature challenge and should not be interpreted as drug-repurposing evidence.

Preferred real input format:

```csv
compound,gene,score
drug_a,GENE1,-2.1
drug_a,GENE2,1.4
drug_b,GENE1,0.7
```

Run with real profiles:

```bash
.venv/bin/python scripts/run_rnaseq_quantum_benchmark.py \
  --cmap-signatures path/to/compound_gene_scores.csv \
  --min-profile-gene-overlap 5 \
  --out-dir artifacts/benchmarks/rnaseq_quantum_real_profiles \
  --top-genes 24 \
  --qml-dims 2,3,4 \
  --qsvc-reps-list 1,2
```

The script also accepts CMap-style columns understood by
`perturbation_layer.cmap_loader.load_cmap_csv`.

CREEDS drug perturbation JSON can be used without a login:

```bash
.venv/bin/python scripts/run_rnaseq_quantum_benchmark.py \
  --creeds-signatures artifacts/external/creeds/single_drug_perturbations-v1.0.json \
  --gene-map artifacts/external/airway/converted/airway_gene_map.csv \
  --out-dir artifacts/benchmarks/rnaseq_quantum_creeds_airway \
  --top-genes 16 \
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

For the current airway+CREEDS pilot, the defensible interpretation is that the
classifier comparison is real but underpowered, and the quantum ranking effect
is negligible unless the report shows lower KG+omics-vs-quantum correlation,
larger rank shifts, or changed top-k candidates on a larger dataset.

Use `--full-permutations 0` for quick exploratory runs. For a result intended
for outside review, include a nonzero full retraining permutation count and
report `classifier_full_permutation.csv` alongside the verdict.

## Evidence Audit

After a benchmark run, audit the evidence bundle before making claims:

```bash
.venv/bin/python scripts/audit_rnaseq_quantum_evidence.py \
  --benchmark-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized \
  --external-validation artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/external_validation.json
```

The audit writes:

- `evidence_audit.json`: machine-readable gates and readiness status.
- `evidence_audit.md`: compact review summary.

Default review gates require:

- at least 60 classifier samples and at least 25 samples in the smallest class
  for the default review sample-size gate; this threshold is necessary but not
  sufficient for publication;
- at least 100 valid full retraining permutations;
- real perturbation-ranking evidence with at least 50 candidates;
- statistically supported positive quantum-value claims when
  `quantum_adds_value` is true.
- an independent cohort with at least 60 samples and 25 samples in the smaller
  class, locked development configuration, explicit leakage controls,
  patient-cluster bootstrap, and pair-aware permutations.

The current airway+CREEDS bundle should audit as `pilot_only`, not
`review_ready`, because it has only 8 samples. That is an intentional guardrail:
the pipeline should make weak evidence obvious rather than turn it into a strong
claim.

## IBM Runtime

Local simulator mode is the default. IBM execution remains gated:

```bash
.venv/bin/python scripts/run_rnaseq_quantum_benchmark.py \
  --quantum-mode ibm \
  --allow-ibm-submit \
  ...
```

Do not use IBM mode as the first evidence run. Establish local simulator
baselines first, then submit only selected small configurations if there is a
clear reason to test hardware behavior.
