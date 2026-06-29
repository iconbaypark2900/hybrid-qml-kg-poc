# RNA-seq repurposing validation (Tier 3 Methods)

Supplement to the hybrid QML–KG paper. Source run log: `results/rnaseq_repurposing_run/FINDINGS.md` (Phases 12–13).

## Cohort and external validation

RNA-seq evidence uses a harmonized **TCGA-BRCA development** cohort (60 samples) with **GSE225846** held out for external validation. Classifier comparison on the external set:

| Model | External ROC-AUC | Balanced accuracy |
|-------|-----------------:|------------------:|
| Logistic regression | **0.973** | **0.916** |
| QSVC (locked hyperparameters) | **0.922** | **0.829** |

Delta ROC-AUC: **−0.051** (QSVC underperforms). We do **not** claim quantum advantage on this RNA-seq task.

## Ranking: `kg_scores_plus_creeds`

`scripts/run_rnaseq_quantum_benchmark.py` merges exported 256D+MoA multiseed KG scores (`--kg-scores`) with CREEDS perturbation profiles (`--creeds-signatures`). Ranking evidence level: **`kg_scores_plus_creeds`**.

- **18** candidates in the full pool (intersection of KG export and CREEDS human profiles with ≥3 gene overlap).
- Quantum ranking does **not** change ranks relative to classical under this intersection.
- Audit flags `too_few_candidates` (18 < 50 ranking-count threshold); integrity checks pass.

## CREEDS coverage

Policy and match tables: [CREEDS_COVERAGE.md](CREEDS_COVERAGE.md). Breast CtD subset: **4/11** human CREEDS matches, **6/11** with `organism=any`, **5/11** permanently unmatched in CREEDS v1.0.

## Fusion ablation (optional)

On the 11-candidate breast subset, sweeping `signature_reversal_multiplier` (1.0–5.0) with `zero_unmatched_reversal=true` lifts matched-drug fusion scores without reordering the top candidate (Vemurafenib). Artifacts: `results/repurposing_fusion_ablation/breast_human.json`.

## Audit caveat

Workbench readiness is **`not_review_ready`** only on the ranking-count gate (18 < 50). Classifier sample size, external validation, and bundle integrity checks pass (`scripts/verify_rnaseq_evidence_bundle.py`: 48/50).
