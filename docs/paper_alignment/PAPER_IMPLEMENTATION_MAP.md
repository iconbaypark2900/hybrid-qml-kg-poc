# Paper Implementation Map

Maps every section of the submitted manuscript to the corresponding repo module.

Status legend: ✅ Implemented · 🔧 Partial · ❌ Missing · 🔲 Planned (this sprint)

---

## Methods

| Paper Section | Description | Repo Module | Status | Notes |
|---|---|---|---|---|
| §2.1 Knowledge Graph | Hetionet v1.0 loading & edge extraction | `kg_layer/kg_loader.py` | ✅ | `load_hetionet_edges`, `extract_task_edges` |
| §2.2 KG Embeddings | RotatE / ComplEx / DistMult training | `kg_layer/kg_embedder.py`, `kg_layer/advanced_embeddings.py` | ✅ | Pre-trained 128/256/512d embeddings in `data/` |
| §2.3 Feature Engineering | Compound-disease pair feature construction | `kg_layer/enhanced_features.py`, `kg_layer/improved_feature_engineering.py` | ✅ | Hadamard, L1, L2, avg operators |
| §2.4 Quantum Feature Maps | Pauli + ZZ feature maps | `quantum_layer/quantum_feature_maps.py` | ✅ | QSVC_FEATURE_MAP_TYPE = PauliFeatureMap |
| §2.5 QSVC | Quantum support vector classifier | `quantum_layer/qml_model.py`, `quantum_layer/qml_trainer.py` | ✅ | Main quantum model |
| §2.6 VQC | Variational quantum classifier | `quantum_layer/qml_model.py` | ✅ | Secondary quantum model |
| §2.7 Classical Baselines | LR, RF, GBM, XGB baselines | `classical_baseline/train_baseline.py` | ✅ | |
| §2.8 Stacking Ensemble | Quantum-classical stacking | `quantum_layer/quantum_classical_ensemble.py` | ✅ | Headline config |
| §2.9 Hard Negatives | Hard negative sampling | `scripts/hard_negatives_experiment.py` | ✅ | |
| §2.10 Evaluation | PR-AUC, ROC-AUC, nested CV | `benchmarking/metrics_tracker.py`, `scripts/nested_cv.py` | ✅ | PR-AUC 0.7987 |
| §3.1 Single-Cell QC | scRNA-seq QC and filtering | `single_cell_layer/qc.py` | ❌ | Sprint 3 |
| §3.2 Batch Correction | Harmony batch correction | `single_cell_layer/batch_correction.py` | ❌ | Sprint 3 |
| §3.3 Differential Expression | Disease vs. control DE | `single_cell_layer/differential_expression.py` | ❌ | Sprint 4 |
| §3.4 Disease Signatures | Cell-type-specific gene signatures | `single_cell_layer/disease_signature.py` | ❌ | Sprint 4 |
| §3.5 Drug Perturbation | LINCS/CMap signature loading | `perturbation_layer/lincs_loader.py` | ❌ | Sprint 5 |
| §3.6 Reversal Scoring | Disease-drug reversal score | `perturbation_layer/reversal_score.py` | ❌ | Sprint 5 |
| §3.7 Evidence Fusion | KG + QML + omics feature fusion | `evidence_layer/feature_fusion.py` | ❌ | Sprint 6 |
| §3.8 Clinical Validation | ClinicalTrials.gov + literature | `validation_layer/clinical_trials_validator.py` | ❌ | Sprint 8 |

---

## Results

| Paper Component | Repo Artifact | Status | Script |
|---|---|---|---|
| Table 1: Model comparison | `results/` | 🔧 Partial | `scripts/analyze_results.py` |
| Table 2: Top-N candidates | `artifacts/predictions/top_candidates.csv` | ❌ | `scripts/run_full_repurposing_pipeline.py` |
| Figure 1: Architecture diagram | `docs/` | ✅ | Static figure |
| Figure 2: PR-AUC curves | `figures/` | 🔧 | `benchmarking/metrics_tracker.py` |
| Figure 3: UMAP cell states | `artifacts/single_cell/cell_states/` | ❌ | `single_cell_layer/cell_state_embedding.py` |
| Figure 4: Reversal heatmap | `artifacts/perturbations/` | ❌ | `perturbation_layer/reversal_report.py` |
| Figure 5: Evidence breakdown | Dashboard screenshot | ❌ | `benchmarking/components/evidence_card.py` |

---

## Reproducibility

| Item | Location | Status |
|---|---|---|
| Random seeds | `utils/preregistered_constants.py` | ✅ |
| Hetionet version hash | `scripts/record_hetionet_hash.py` | ✅ |
| Nested CV protocol | `scripts/nested_cv.py` | ✅ |
| OSF preregistration | `preregistration/` | ✅ |
| Full pipeline smoke test | `scripts/e2e_smoke.py` | ✅ |
| DGX smoke test | `scripts/dgx/run_smoke_test.sh` | ❌ Sprint 10 |
