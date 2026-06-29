# RNA-seq + Repurposing Run Log

**Date:** 2026-06-29  
**Machine:** DGX Spark (local)  
**Goal:** Exercise counts → signature → quantum benchmark → repurposing fusion paths and record what is real vs demo.

---

## Phase 1 — Counts pipeline (smoke, fixture data)

**Command:**

```bash
python scripts/run_rnaseq_counts_pipeline.py \
  --input tests/fixtures/rnaseq_counts/counts.csv \
  --format count-matrix \
  --metadata tests/fixtures/rnaseq_counts/metadata.csv \
  --out-dir artifacts/single_cell/pilot_run \
  --signatures-dir artifacts/signatures/pilot_run \
  --disease-id Disease::DOID:1612 \
  --tissue blood \
  --de-method simple \
  --top-n 10
```

**Findings:**

| Output | Path |
|--------|------|
| QC summary | `artifacts/single_cell/pilot_run/qc/qc_summary_table.csv` |
| Normalized counts | `artifacts/single_cell/pilot_run/normalized_counts.csv` |
| DE table | `artifacts/single_cell/pilot_run/differential_expression.csv` |
| Disease signature | `artifacts/signatures/pilot_run/disease_signature.json` |
| Manifest | `artifacts/single_cell/pilot_run/rnaseq_counts_manifest.json` |

Signature correctly identifies up genes (`GENE_CASE`, `GENE_UP`) and down genes (`GENE_DOWN`, `GENE_CTRL`) from 4 samples (2 disease / 2 control). **Pipeline wiring works end-to-end.**

**Limitation:** 4-sample fixture is for smoke only — not publishable evidence.

---

## Phase 2 — RNA-seq quantum benchmark (pilot)

**Command:**

```bash
python scripts/run_rnaseq_quantum_benchmark.py \
  --normalized-counts artifacts/single_cell/pilot_run/normalized_counts.csv \
  --metadata tests/fixtures/rnaseq_counts/metadata.csv \
  --signature artifacts/signatures/pilot_run/disease_signature.json \
  --out-dir artifacts/benchmarks/rnaseq_quantum_pilot_run \
  --case-label disease \
  --control-label control \
  --demo-ranking \
  --top-genes 4 \
  --qml-dims 2,3 \
  --qsvc-reps-list 1,2 \
  --bootstrap 200 \
  --permutations 200
```

**Note:** Default `--case-label trt` / `--control-label untr` does not match fixture metadata; use `disease` / `control`.

| Metric | Result |
|--------|--------|
| Classifier verdict | `quantum_underperforms_classical` |
| Evidence grade | `pilot_underpowered` (n=4, 2 per class) |
| Best classical ROC-AUC | 1.0 (logistic regression) |
| Best quantum ROC-AUC | 0.0 |
| Quantum adds value | **No** |
| Ranking evidence | `demo_signature_challenge` — **not real perturbation data** |

Full report: `artifacts/benchmarks/rnaseq_quantum_pilot_run/benchmark_report.md`

**Interpretation:** Expected on a toy cohort. Real evidence requires ≥20 samples (TCGA-BRCA 60-sample cohort per `docs/guides/TCGA_BRCA_RNASEQ_COHORT.md`).

---

## Phase 3 — Full repurposing pipeline

### 3a. KG-only (demo candidates)

```bash
python scripts/run_full_repurposing_pipeline.py \
  --mode kg-only --top-n 12 \
  --output results/rnaseq_repurposing_run/kg_only
```

| Field | Value |
|-------|-------|
| Top candidate | sertraline |
| Score | 0.456 |
| Tier | 4 (exploratory) |
| Omics | zero-filled |

### 3b. KG + omics + clinical validation (all demo diseases)

```bash
python scripts/run_full_repurposing_pipeline.py \
  --mode kg+omics --validate --top-n 12 \
  --output results/rnaseq_repurposing_run/kg_omics_all
```

| Field | Value |
|-------|-------|
| Top candidate | sertraline |
| Score | 0.716 |
| Tier | 3 (moderate) |
| Reversal scores | **synthetic** (perturbation registry empty) |
| Clinical trials | queried for top-12 (some moderate clinical evidence) |

### 3c. Breast cancer only (`Disease::DOID:1612`)

```bash
python scripts/run_full_repurposing_pipeline.py \
  --mode kg+omics --validate --disease Disease::DOID:1612 \
  --output results/rnaseq_repurposing_run/kg_omics_breast_cancer
```

| Field | Value |
|-------|-------|
| Top candidate | **tamoxifen** → breast cancer |
| Score | 0.690 |
| Tier | 4 |
| Clinical evidence | moderate (0.5) — known-indication lookup partial |

Reports: `results/rnaseq_repurposing_run/kg_omics_*/final_repurposing_report.md`

**Limitation:** No upstream `--kg-scores` JSON was passed; demo seeded KG/QSVC scores used. See **Phase 5** for real CtD scores.

---

## Phase 5 — Real CtD scores wired (`results/rerun_256d_moa/`)

**Export (5-seed mean, top 200 test pairs):**

```bash
python scripts/export_kg_qml_scores.py \
  --results-dir results/rerun_256d_moa \
  --seeds 42 7 13 99 2026 \
  --top-n 200 --aggregate mean \
  --out results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json
```

**Repurposing with real scores:**

```bash
python scripts/run_full_repurposing_pipeline.py \
  --mode kg+omics --validate \
  --kg-scores results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json \
  --top-n 25 \
  --output results/rnaseq_repurposing_run/repurposing_real_kg_omics
```

| Field | Value |
|-------|-------|
| Candidates loaded | 200 (CtD test pairs, 5-seed mean) |
| Top overall | **Toremifene** (score 0.613, tier 4) |
| Breast cancer filter (`DOID:1612`) | 11 pairs; top **Vinblastine** (0.590) |
| QSVC scores in export | ~0 (quantum kernel near chance on CtD) |
| Reversal | still **synthetic** until perturbation registry wired |

Breast report: `results/rnaseq_repurposing_run/repurposing_breast_cancer_real/final_repurposing_report.md`

**Code fixes:** `predictions_compare.csv` now uses best PR-AUC classical (HistGBDT); `--disease` filter applies to loaded `--kg-scores`.

---

## Phase 6 — CREEDS perturbation profiles

```bash
python scripts/download_creeds_signatures.py
```

| Field | Value |
|-------|-------|
| Records | 875 total, 450 human |
| Path | `artifacts/external/creeds/single_drug_perturbations-v1.0.json` |

**Harmonized TCGA benchmark + CREEDS ranking:**

```bash
python scripts/run_rnaseq_quantum_benchmark.py \
  --normalized-counts artifacts/single_cell/tcga_brca_gse225846_harmonized/development_normalized_counts.csv \
  --creeds-signatures artifacts/external/creeds/single_drug_perturbations-v1.0.json \
  --out-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized_session \
  ...
```

| Metric | Value |
|--------|-------|
| Ranking evidence | **`creeds_signatures`** (real) |
| Top CREEDS reverser | Anastrozole (rev 0.696, breast biopsy GSE33658) |
| Quantum ranking materiality | **negligible** (rankings unchanged) |
| Classifier (TCGA dev) | QSVC matches LR (ROC-AUC 1.0); no quantum lift |

Report: `artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized_session/benchmark_report.md`  
Ranking: `.../ranking_comparison.csv` (67 CREEDS profiles)

---

## Phase 7 — GSE225846 external validation

**GEO download + harmonization:**

```bash
python scripts/prepare_geo_gse225846_counts_cohort.py --out-dir artifacts/external/geo_gse225846
python scripts/harmonize_rnaseq_cohorts.py \
  --development-counts artifacts/external/gdc_tcga_brca/converted/tcga_brca_counts.csv \
  --validation-counts artifacts/external/geo_gse225846/converted/gse225846_counts.csv \
  --out-dir artifacts/single_cell/tcga_brca_gse225846_harmonized
```

Shared gene universe: **18,255 genes** (369 TCGA-only genes dropped for GEO annotation).

**Locked external validation** (`bootstrap=2000`, `permutations=10000`):

| Metric | Logistic regression | QSVC |
|--------|--------------------:|-----:|
| External ROC-AUC | **0.973** | 0.922 |
| Balanced accuracy | **0.916** | 0.829 |
| Δ ROC-AUC (Q − C) | **−0.051** | |
| Patient-cluster bootstrap Δ 95% CI | [−0.083, −0.025] | |
| McNemar p (discordant) | **0.007** (17 LR-only vs 4 QSVC-only correct) | |

Verdict: `quantum_underperforms_classical_external` — both discriminate (permutation p=0.0001) but **no quantum advantage**.

Outputs: `artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external_session/`  
Audit: `artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized_session/evidence_audit.md`

---

## Pre-existing BRCA evidence (repo artifacts)

Prior work already packaged a stronger breast-cancer bundle:

| Artifact | Location |
|----------|----------|
| Evidence bundle | `artifacts/repurposing/brca_external_validation/repurposing_evidence_bundle.md` |
| External classical ROC-AUC | **0.973** |
| External quantum ROC-AUC | 0.922 |
| Quantum delta | **−0.051** (classical wins) |
| Mapped candidates | 8 with structure coverage |
| Claim policy | Research hypotheses only — no cure/clinical efficacy claims |

Source ranking: `artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/` (TCGA 60 + GSE225846 external validation). **Not re-run in this session** — TCGA counts not present locally yet.

---

## Phase 4 — TCGA-BRCA cohort (completed this session)

**Download (GDC open access, 30 tumor + 30 normal):**

```bash
python scripts/prepare_gdc_tcga_counts_cohort.py \
  --project-id TCGA-BRCA --n-case 30 --n-control 30 \
  --out-dir artifacts/external/gdc_tcga_brca
```

Log: `results/rnaseq_repurposing_run/tcga_brca_prepare.log`  
Outputs: `artifacts/external/gdc_tcga_brca/converted/tcga_brca_counts.csv` (18,624 genes × 60 samples)

**PyDESeq2 differential expression** (required `pip install pydeseq2` — also in `requirements-omics.txt`):

| Stat | Value |
|------|-------|
| Tested genes | 18,624 |
| adj p ≤ 0.05 | 10,675 |
| adj p ≤ 0.05 & \|log2FC\| ≥ 0.5 | 8,579 |
| Signature up / down genes | 240 / 250 |

Artifacts: `artifacts/single_cell/tcga_brca_60/`, `artifacts/signatures/tcga_brca_60/`  
Log: `results/rnaseq_repurposing_run/tcga_brca_de.log`

**TCGA quantum classifier benchmark (completed):**

| Metric | Value |
|--------|-------|
| Evidence grade | `analysis_ready` (60 samples) |
| Classifier verdict | `quantum_matches_classical_within_delta` |
| Best classical ROC-AUC | 1.0 (logistic regression) |
| Best quantum ROC-AUC | 1.0 (QSVC dim=2, reps=1) |
| Quantum adds value | **No** (Δ ROC-AUC = 0.0) |
| Ranking | `demo_signature_challenge` — need CREEDS profiles |

Report: `artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_session/benchmark_report.md`  
Log: `results/rnaseq_repurposing_run/tcga_brca_quantum_benchmark.log`

CREEDS perturbation profiles not downloaded yet — ranking leg uses `--demo-ranking` until
`artifacts/external/creeds/` is populated.

---

## Summary table

| Track | Status | Evidence quality |
|-------|--------|------------------|
| Counts → signature | ✅ | Fixture + TCGA PyDESeq2 |
| Real CtD → repurposing | ✅ | 200 pairs from 256D+MoA multiseed |
| CREEDS drug ranking | ✅ | 67 profiles; quantum materiality negligible |
| GSE225846 external | ✅ | Matches prior published numbers |
| Repurposing omics reversal | ✅ | CREEDS wired; 4/11 human BRCA match (6/11 with `organism=any`) |

---

## Phase 8 — CREEDS reversal in repurposing pipeline

`perturbation_layer/creeds_reversal.py` maps CREEDS drug names → Hetionet via `CompoundMapper` (DrugBank ID, exact name, substring) and scores gene-level reversal vs a disease signature.

### v1 (baseline wiring)

```bash
python scripts/run_full_repurposing_pipeline.py \
  --mode kg+omics --validate \
  --kg-scores results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json \
  --disease Disease::DOID:1612 \
  --creeds-signatures artifacts/external/creeds/single_drug_perturbations-v1.0.json \
  --disease-signature artifacts/signatures/tcga_brca_60/disease_signature.json \
  --output results/rnaseq_repurposing_run/repurposing_breast_creeds
```

| Metric | Value |
|--------|-------|
| Omics source | `creeds` (not synthetic) |
| CREEDS profile match | **4 / 11** breast CtD candidates |
| Top rank | **Vemurafenib** (0.522; reversal 0.542) |
| Unmatched reversal | 0.0 (e.g. Toremifene, Vinblastine — no CREEDS overlap) |

### v2 (matching + aggregation + cosine mode)

**Code changes:**

- Aggressive compound normalization (parenthetical stripping, salt suffixes).
- `SynonymResolver` + direct `Compound::{DrugBank}` index keys.
- Multi-profile aggregation: **mean of top-3** profile scores (metadata uses best single profile).
- `--creeds-reversal-method {gene_overlap,cosine}` (cosine aligns CREEDS gene scores vs disease `ranked_genes` LFC, same `_cosine01` scale as RNA-seq benchmark).
- `--creeds-organism any` optional (includes rat/mouse when human absent).

```bash
python scripts/run_full_repurposing_pipeline.py \
  --mode kg+omics --validate \
  --kg-scores results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json \
  --disease Disease::DOID:1612 \
  --output results/rnaseq_repurposing_run/repurposing_breast_creeds_v2
```

| Metric | v1 | v2 (`gene_overlap`, human) |
|--------|----|----------------------------|
| `n_creeds_matched` | 4 / 11 | **4 / 11** (unchanged) |
| Top-1 | Vemurafenib 0.522 | Vemurafenib **0.520** (rev **0.535**) |
| Top-2 | Cisplatin 0.489 | Cisplatin **0.488** (rev **0.531**) |
| Top-3 | Paclitaxel 0.475 | Paclitaxel **0.474** (rev **0.494**) |

**Why match rate unchanged:** seven breast CtD candidates have **no human CREEDS profile** (Toremifene, Vinblastine, Fingolimod, Capecitabine, Dacarbazine, etc.). Irinotecan and Prednisolone exist only as **rat** profiles (`--creeds-organism any` → **6 / 11** matched).

**Cosine mode** (`--creeds-reversal-method cosine`, output `repurposing_breast_creeds_v2_cosine`):

| Top-3 | Final score | Reversal |
|-------|------------:|---------:|
| Vemurafenib | 0.532 | 0.588 |
| Cisplatin | 0.498 | 0.576 |
| Paclitaxel | 0.469 | 0.469 |

Tests: `tests/test_creeds_reversal.py` (aggregation + cosine unit tests); `tests/test_pipeline_integration.py` (13/13 pass).

---

## Phase 9 — Full 200-pair repurposing (cosine CREEDS)

**Command:**

```bash
python scripts/run_full_repurposing_pipeline.py \
  --mode kg+omics \
  --kg-scores results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json \
  --creeds-reversal-method cosine \
  --output results/rnaseq_repurposing_run/repurposing_full_200_cosine \
  --top-n 200
```

| Metric | Value |
|--------|------:|
| Candidates | 200 (all diseases in kg-scores export) |
| CREEDS matched (human default) | **46 / 200** (23%) |
| Top candidate | Fluticasone furoate (score **0.543**, tier 4) |
| Evidence tiers 1–3 | 0 (all tier 4 — KG+omics fusion without clinical validation) |

Outputs: `repurposing_full_200_cosine/top_candidates.csv`, `final_repurposing_report.md`, `run_summary.json`.

**Interpretation:** Ranking remains KG-dominated for unmatched drugs; cosine lifts reversal signal where CREEDS profiles exist (see breast subset above). Use `--creeds-organism any` to widen match rate at the cost of non-human evidence.

---

## Phase 10 — Workbench artifact bridge (Tier 3)

**One-command rebuild:**

```bash
./scripts/run_repurposing_workbench_refresh.sh
```

Chains: breast pipeline (human + `organism=any`) → `candidates_enriched.json` → `export_repurposing_ranking_comparison.py` → `build_repurposing_evidence_bundle.py` → fig4 correlation.

| Bundle | Disease API id | CREEDS match | Top candidate |
|--------|----------------|--------------|---------------|
| Human (default) | `brca_external_validation` | **4 / 11** | **Vemurafenib** (0.508) |
| organism=any | `brca_external_validation_organism_any` | **6 / 11** | **Prednisolone** (0.525) |

Artifacts:

- `artifacts/repurposing/brca_external_validation/repurposing_evidence_bundle.json`
- `artifacts/repurposing/brca_external_validation_organism_any/repurposing_evidence_bundle.json`
- `results/rnaseq_repurposing_run/repurposing_breast_bundle_human/`
- `results/kg_rnaseq_correlation_metrics.json` (breast matched n=4, rho=1.0; full 200 matched n=46)

**UI:** `/v2/repurposing?disease_id=brca_external_validation` loads artifact-backed candidates with CREEDS evidence component + match badges (human / non-human / unmatched).

---

## Phase 11 — Tier 3 science follow-ups (2026-06-29)

### CREEDS coverage policy

Documented in [`docs/repurposing/CREEDS_COVERAGE.md`](../../docs/repurposing/CREEDS_COVERAGE.md): human default, `organism=any` alternate, unmatched = missing data not negative evidence.

### Fusion weight ablation

`scripts/ablate_evidence_fusion_weights.py` sweeps `signature_reversal_multiplier` (1.0–5.0) with `zero_unmatched_reversal`. Results: `results/repurposing_fusion_ablation/breast_human.json`.

At multiplier **5.0**, breast top-3 stays **Vemurafenib / Cisplatin** (matched); reversal weight lifts matched scores without reordering below unmatched KG-only drugs into #1.

Config: `config/evidence_fusion_config.yaml` (`weights`, `matched_omics`).

### Cell-type / pathway decoupled

CREEDS path sets `cell_type_reversal_score=0`, `pathway_reversal_score=0` (not computed). Fusion weights for those features are **0**. Explanations and workbench `rnaseq_signature` label them `not_computed`.

### 200-pair workbench

API disease id **`all_pairs_kg_omics`** → `artifacts/repurposing/all_pairs_kg_omics/` (top 50 of 200, 46 CREEDS matched in full run).

### RNA-seq benchmark + kg-scores

`run_rnaseq_quantum_benchmark.py` accepts `--kg-scores` + `--creeds-signatures` (+ optional `--disease-hetionet-id`) to merge multiseed KG scores into CREEDS ranking profiles (`kg_scores_plus_creeds` evidence level).

Example (when TCGA normalized counts exist):

```bash
python scripts/run_rnaseq_quantum_benchmark.py \
  --normalized-counts artifacts/single_cell/tcga_brca_60/normalized_counts.csv \
  --metadata artifacts/external/gdc_tcga_brca/metadata.csv \
  --signature artifacts/signatures/tcga_brca_60/disease_signature.json \
  --kg-scores results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json \
  --creeds-signatures artifacts/external/creeds/single_drug_perturbations-v1.0.json \
  --gene-map artifacts/external/gdc_tcga_brca/converted/tcga_brca_gene_map.csv \
  --disease-hetionet-id Disease::DOID:1612 \
  --out-dir results/rnaseq_repurposing_run/benchmark_kg_scores_breast
```

### Optuna

Deferred — post–`fast_mode` fix available; diminishing returns vs reproducible **0.7805**.

---

## Phase 12 — Fusion weight ablation (detailed, 2026-06-29)

**Setup:** 11 breast CtD candidates from multiseed KG export; **4/11** human CREEDS matches. Config: `zero_unmatched_reversal=true`, `signature_reversal_multiplier` sweep **1.0–5.0** via [`scripts/ablate_evidence_fusion_weights.py`](../../scripts/ablate_evidence_fusion_weights.py). Source: [`results/repurposing_fusion_ablation/breast_human.json`](../repurposing_fusion_ablation/breast_human.json).

| Multiplier | Top-1 | Top-1 score | Top-3 (matched in top-3) |
|------------|-------|-------------|--------------------------|
| 1.0 | Vemurafenib | 0.553 | 2/3 (Vemurafenib, Cisplatin, **Toremifene**) |
| 2.0 | Vemurafenib | 0.557 | 2/3 |
| 3.0 | Vemurafenib | 0.561 | 2/3 |
| 5.0 | Vemurafenib | 0.566 | 2/3 |

**Toremifene** (rank #3 at all multipliers): `kg_rotate_score=0.992`, `signature_reversal_score=0.0` (no CREEDS profile) — unchanged by multiplier because `zero_unmatched_reversal=true`.

**Interpretation:** Increasing reversal weight up to 5× raises matched-drug fusion scores (+0.013 Vemurafenib, +0.024 Cisplatin) without changing top-1 rank or promoting unmatched KG-only candidates to #1. KG dominates when CREEDS coverage is 36% (4/11). Policy: [`config/evidence_fusion_config.yaml`](../../config/evidence_fusion_config.yaml) `matched_omics` block.

**Methods sentence (optional supplement):** Fusion weights were ablated on the breast CtD subset; matched-omics reversal multipliers did not reorder the top candidate despite lifting matched scores.

---

## Phase 13 — Harmonized kg-scores benchmark + external validation (2026-06-29)

### Development benchmark (`kg_scores_plus_creeds`)

**Command (full ranking pool, harmonized TCGA development counts):**

```bash
.venv/bin/python scripts/run_rnaseq_quantum_benchmark.py \
  --normalized-counts artifacts/single_cell/tcga_brca_gse225846_harmonized/development_normalized_counts.csv \
  --metadata artifacts/external/gdc_tcga_brca/converted/tcga_brca_metadata.csv \
  --signature artifacts/signatures/tcga_brca_60/disease_signature.json \
  --kg-scores results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json \
  --creeds-signatures artifacts/external/creeds/single_drug_perturbations-v1.0.json \
  --gene-map artifacts/external/gdc_tcga_brca/converted/tcga_brca_gene_map.csv \
  --out-dir artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized \
  --top-genes 32 --min-profile-gene-overlap 3 --max-ranking-profiles 100 \
  --bootstrap 1000 --permutations 1000 --full-permutations 100
```

| Metric | Value |
|--------|------:|
| Ranking evidence level | **`kg_scores_plus_creeds`** (real multiseed KG + CREEDS) |
| Ranking candidates (full pool) | **18** (KG export ∩ CREEDS human profiles with ≥3 gene overlap) |
| Ranking candidates (breast filter) | **7** (`--disease-hetionet-id Disease::DOID:1612`; archived under `benchmark_kg_scores_breast/`) |
| Classifier verdict | `quantum_matches_classical_within_delta` |
| Best classical (development) | logistic_regression ROC-AUC **1.0** |
| Quantum ranking materiality | `too_few_candidates` for audit threshold (18 < 50) |

Breast-filtered run log: `results/rnaseq_repurposing_run/benchmark_kg_scores_harmonized.log`. Full-pool log: `benchmark_kg_scores_harmonized_full.log`.

**vs prior `creeds_signatures` session run:** Same classifier metrics; ranking now uses exported 256D+MoA multiseed scores instead of demo `kg_rotate`. Intersection of KG top-200 and CREEDS profiles yields fewer candidates (18) than CREEDS-only (67) — expected when requiring both evidence sources.

### GSE225846 external validation

| Model | External ROC-AUC | Balanced accuracy |
|-------|-----------------:|------------------:|
| Logistic regression | **0.973** | **0.916** |
| QSVC (locked) | **0.922** | **0.829** |

Delta ROC-AUC: **−0.051** (QSVC underperforms). Verdict: `quantum_underperforms_classical_external`. Artifacts: `artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/`.

### Evidence audit

| Gate | Status |
|------|--------|
| Classifier sample size | pass |
| Full retraining permutation | pass |
| Independent external validation | pass |
| External classical signal | pass |
| Ranking evidence | **warn** (18 candidates < 50 threshold) |
| Ranking quantum materiality | **fail** (`too_few_candidates`) |
| Readiness | **`not_review_ready`** (ranking-count gate only) |

Integrity verification: 48/50 checks pass (`scripts/verify_rnaseq_evidence_bundle.py`); 2 audit-only failures. Workbench bundles use real proof artifacts with audit flagged as **warn** in bundle metadata.

### CREEDS coverage (Methods pointer)

Policy and match tables: [`docs/repurposing/CREEDS_COVERAGE.md`](../../docs/repurposing/CREEDS_COVERAGE.md). Breast CtD: **4/11** human, **6/11** with `organism=any`, **5/11** permanently unmatched in CREEDS v1.0.

### Pathway GSEA

**Not computed** in this release (`pathway_reversal_score=0`, fusion weight **0** in `config/evidence_fusion_config.yaml`; UI labels `not_computed`).

---

## Claim policy

Do **not** infer cures or clinical efficacy from demo/synthetic runs. Ranked outputs are **research hypotheses** requiring real omics, real perturbation profiles, and independent validation before any product or publication claim.
