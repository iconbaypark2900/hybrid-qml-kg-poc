# Figure Reproduction Plan

Maps each manuscript figure and table to the script / notebook that generates it, and its current status.

Status: ✅ Reproducible now · 🔧 Partial (script exists, needs polish) · ❌ Missing (planned sprint noted)

---

## Figures

### Figure 1 — System Architecture Diagram

| Item | Detail |
|---|---|
| Type | Static diagram |
| Source | `docs/ARCHITECTURE.md` (text) |
| Output | `figures/architecture.png` (static asset) |
| Status | ✅ |
| Notes | Update diagram once single-cell + evidence layers are wired |

---

### Figure 2 — PR-AUC / ROC-AUC Curves (Model Comparison)

| Item | Detail |
|---|---|
| Type | Line plot |
| Script | `benchmarking/metrics_tracker.py` + `scripts/analyze_results.py` |
| Output | `figures/pr_auc_comparison.png` |
| Status | 🔧 Partial — curves generated; needs final style pass |
| Reproduce | `python scripts/analyze_results.py --results_dir results/ --out figures/` |

---

### Figure 3 — Quantum vs. Classical Feature Separation

| Item | Detail |
|---|---|
| Type | Scatter / PCA plot |
| Script | `scripts/diagnose_quantum_separability.py` |
| Output | `figures/quantum_separability.png` |
| Status | 🔧 Partial |
| Reproduce | `python scripts/diagnose_quantum_separability.py` |

---

### Figure 4 — UMAP Cell-State Map (Disease vs. Control)

| Item | Detail |
|---|---|
| Type | UMAP scatter colored by cluster / condition |
| Script | `single_cell_layer/cell_state_embedding.py` |
| Output | `artifacts/single_cell/cell_states/umap_embeddings.csv` → `figures/umap_cell_states.png` |
| Status | ❌ Missing — Sprint 3 |
| Reproduce | `python -m single_cell_layer.cell_state_embedding --config config/single_cell_config.yaml` |

---

### Figure 5 — Disease Gene Signature Heatmap

| Item | Detail |
|---|---|
| Type | Heatmap (genes × cell types, colored by log-FC) |
| Script | `single_cell_layer/differential_expression.py` + `single_cell_layer/signature_export.py` |
| Output | `artifacts/signatures/disease_signature.json` → `figures/disease_signature_heatmap.png` |
| Status | ❌ Missing — Sprint 4 |
| Reproduce | `python -m single_cell_layer.disease_signature --config config/single_cell_config.yaml` |

---

### Figure 6 — Drug Reversal Score Heatmap

| Item | Detail |
|---|---|
| Type | Heatmap (compounds × cell types, colored by reversal score) |
| Script | `perturbation_layer/reversal_report.py` |
| Output | `artifacts/perturbations/reversal_scores.csv` → `figures/reversal_heatmap.png` |
| Status | ❌ Missing — Sprint 5 |
| Reproduce | `python -m perturbation_layer.reversal_report --demo` |

---

### Figure 7 — Evidence Breakdown for Top Candidates

| Item | Detail |
|---|---|
| Type | Stacked bar / radar chart |
| Script | `evidence_layer/evidence_report.py` |
| Output | `artifacts/reports/final_repurposing_report.md` → `figures/evidence_breakdown.png` |
| Status | ❌ Missing — Sprint 6 |

---

## Tables

### Table 1 — Model Performance Comparison

| Item | Detail |
|---|---|
| Script | `scripts/analyze_results.py` |
| Output | `results/model_comparison.csv` |
| Status | 🔧 Partial |
| Columns | Model · PR-AUC · ROC-AUC · F1 · CI-lower · CI-upper |
| Reproduce | `python scripts/analyze_results.py` |

---

### Table 2 — Top-20 Drug Candidates with Validation Status

| Item | Detail |
|---|---|
| Script | `scripts/run_full_repurposing_pipeline.py --mode kg+omics --validate` |
| Output | `artifacts/predictions/top_candidates.csv` |
| Status | ❌ Missing — Sprint 6 + Sprint 8 |
| Columns | Rank · Compound · Disease · Score · Tier · Known · Trial Phase · Literature |

---

### Table 3 — Ablation: KG-only vs. KG+Omics

| Item | Detail |
|---|---|
| Script | `scripts/run_full_repurposing_pipeline.py` (both modes) |
| Output | `results/ablation_kg_vs_omics.csv` |
| Status | ❌ Missing — Sprint 6 |

---

## Supplementary Figures

| Figure | Script | Status |
|---|---|---|
| S1 — ZZ feature map comparison | `scripts/find_best_quantum_config.py` | 🔧 Partial |
| S2 — Noisy vs ideal simulator | `benchmarking/ideal_vs_noisy_compare.py` | ✅ |
| S3 — Batch correction before/after UMAP | `single_cell_layer/harmonization.py` | ❌ Sprint 3 |
| S4 — Pathway enrichment dot plot | `single_cell_layer/pathway_enrichment.py` | ❌ Sprint 4 |
| S5 — Reversal score distribution | `perturbation_layer/reversal_score.py` | ❌ Sprint 5 |
