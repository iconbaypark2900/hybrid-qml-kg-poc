# Paper Implementation Map

Maps every section of the submitted manuscript to the corresponding repo module.

Status legend: ✅ Implemented · 🔧 Partial · ❌ Missing · 🔲 Planned (this sprint)

Last updated: 2026-05-15 (end of Sprint 12)

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
| §2.11 Error Mitigation | Linear + Richardson ZNE | `quantum_layer/advanced_error_mitigation.py` | ✅ | Sprint 0 (preregistration) |
| §2.12 Bootstrap CI | Paired bootstrap for H1/H1b | `utils/bootstrap_ci.py`, `scripts/run_bootstrap_ci.py` | ✅ | Sealed test set wrapper in `utils/sealed_test_set.py` |
| §3.1 Single-Cell QC | scRNA-seq QC and filtering | `single_cell_layer/qc.py`, `single_cell_layer/mitochondrial_metrics.py`, `single_cell_layer/doublet_detection.py` | ✅ | Sprint 3 |
| §3.2 Batch Correction | Harmony batch correction | `single_cell_layer/batch_correction.py`, `single_cell_layer/harmonization.py` | ✅ | Sprint 3 |
| §3.3 Differential Expression | Disease vs. control DE | `single_cell_layer/differential_expression.py` | ✅ | Sprint 4 |
| §3.4 Disease Signatures | Cell-type-specific gene signatures | `single_cell_layer/disease_signature.py`, `single_cell_layer/cell_type_signature.py` | ✅ | Sprints 4, 6 |
| §3.5 Drug Perturbation | LINCS/CMap signature loading | `perturbation_layer/lincs_loader.py`, `perturbation_layer/cmap_loader.py` | ✅ | Sprint 5 |
| §3.6 Reversal Scoring | Disease-drug reversal score | `perturbation_layer/reversal_score.py`, `perturbation_layer/cell_type_reversal.py`, `perturbation_layer/pathway_reversal.py` | ✅ | Sprint 5 |
| §3.7 Evidence Fusion | KG + QML + omics feature fusion | `evidence_layer/feature_fusion.py`, `evidence_layer/evidence_schema.py`, `evidence_layer/confidence_tiering.py` | ✅ | Sprint 6 |
| §3.8 Clinical Validation | ClinicalTrials.gov + literature + DrugBank + Open Targets | `validation_layer/clinical_trials_validator.py`, `literature_validator.py`, `drugbank_mapper.py`, `opentargets_mapper.py` | ✅ | Sprint 8 |
| §3.9 Pipeline Orchestrator | End-to-end `--mode {kg-only,kg+omics}` with `--validate` | `scripts/run_full_repurposing_pipeline.py` | ✅ | Sprint 6-8 wiring |
| §3.10 Entity Resolution | HGNC / DOID / DrugBank → Hetionet node IDs | `entity_resolution/{hetionet_resolver,gene_mapper,disease_mapper,compound_mapper}.py` | ✅ | Sprint 1 |

---

## Results

| Paper Component | Repo Artifact | Status | Producer |
|---|---|---|---|
| Table 1: Headline classification | `artifacts/predictions/sealed_test_metrics.json` | ✅ | `scripts/run_optimized_pipeline.py` + `scripts/build_paper_tables.py --table 1` |
| Table 2: Per-model comparison | `artifacts/predictions/per_model_metrics.csv` | 🔧 | covered by Table 1 path |
| Table 3: Sensitivity (Pauli/ZZ, stack/weight) | `docs/results/sensitivity_*.md` | 🔧 | consolidation deferred to S13 |
| Table 4: Bootstrap CI | `artifacts/predictions/bootstrap_ci.json` | ✅ | `scripts/run_bootstrap_ci.py` + `build_paper_tables.py --table 4` |
| Table 5: Hetionet stats | `artifacts/data/hetionet_stats.json` | ✅ | `scripts/record_hetionet_hash.py` + `build_paper_tables.py --table 5` |
| Table 6: Single-cell QC summary | `artifacts/single_cell/qc/qc_summary_table.csv` | ✅ | `scripts/aggregate_qc_summary.py` + `build_paper_tables.py --table 6` |
| Table 7: Signature catalog | `artifacts/signatures/signature_catalog.csv` | ✅ | `scripts/build_signature_catalog.py` + `build_paper_tables.py --table 7` |
| Table 8: Top-N candidates | `artifacts/predictions/top_candidates.csv` | ✅ | `scripts/run_full_repurposing_pipeline.py` + `build_paper_tables.py --table 8` |
| Table 9: KG-only vs KG+omics Δ | `artifacts/predictions/mode_comparison.csv` | ✅ | `scripts/compare_pipeline_modes.py` + `build_paper_tables.py --table 9` |
| Figure 1: Architecture diagram | `docs/ARCHITECTURE.md` | ✅ | Static figure |
| Figure 2: PR-AUC curves | `figures/pr_auc_comparison.png` | 🔧 | `benchmarking/metrics_tracker.py` |
| Figure 3: Quantum vs classical separability | `figures/quantum_separability.png` | 🔧 | `scripts/diagnose_quantum_separability.py` |
| Figure 4: UMAP cell states | `figures/umap_cell_states.png` | 🔧 | `single_cell_layer/cell_state_embedding.py` + playbook 01 |
| Figure 5: Reversal heatmap | `artifacts/perturbations/` | 🔧 | `perturbation_layer/reversal_report.py` + playbook 03 |
| Figure 6: Evidence breakdown | Dashboard screenshot | ✅ | `benchmarking/components/evidence_card.py` (Sprint 9) |
| Figure 7: Per-disease ΔPR-AUC | `artifacts/predictions/mode_comparison.md` | ✅ | `scripts/compare_pipeline_modes.py` |

---

## Reproducibility

| Item | Location | Status |
|---|---|---|
| Random seeds | `utils/preregistered_constants.py` | ✅ |
| Hetionet version hash | `scripts/record_hetionet_hash.py` | ✅ |
| Nested CV protocol | `scripts/nested_cv.py` | ✅ |
| OSF preregistration | `preregistration/osf_preregistration_v1.md` | ✅ |
| Sealed test set | `utils/sealed_test_set.py` + `tests/test_sealed_test_set.py` | ✅ |
| Full pipeline smoke test | `scripts/dgx/run_smoke_test.sh` (6 steps, 6/6 pass) | ✅ Sprint 10 |
| GitHub Actions CI | `.github/workflows/ci.yml` (smoke + pytest + reproducibility) | ✅ Sprint 11 |
| Integration test coverage | `tests/test_*.py` (88 tests, 51% layer coverage) | ✅ Sprint 11 |
| Reproducibility report | `docs/paper_alignment/REPRODUCIBILITY_REPORT.md` | ✅ Sprint 7 |
| Table reproduction plan | `docs/paper_alignment/TABLE_REPRODUCTION_PLAN.md` | ✅ Sprint 7+12 |
| DGX operational runbook | `docs/deployment/DGX_RUNBOOK.md` | ✅ Sprint 10 |
| Playbook notebooks (00-07) | `playbooks/` | ✅ Sprints 6, 9, 10 |

---

## Sprints completed

| Sprint | Period | Output |
|--------|--------|--------|
| S1 | May 19 – Jun 1 | Paper alignment docs + entity_resolution package |
| S2 | Jun 2 – Jun 15 | Single-cell ingestion (h5ad, 10x, registry) |
| S3 | Jun 16 – Jun 29 | QC, GPU/CPU backends, clustering, batch correction |
| S4 | Jun 30 – Jul 13 | DE + disease signatures + pathway enrichment |
| S5 | Jul 14 – Jul 27 | Perturbation layer + reversal scoring |
| S6 | Jul 28 – Aug 10 | Evidence fusion + `--mode` flag |
| S7 | Aug 11 – Aug 31 | OSF preprint scaffolding |
| S8 | Aug 25 – Sep 7 | Validation layer (ClinicalTrials.gov, DrugBank, Open Targets) |
| S9 | Sep 8 – Sep 21 | Dashboard evidence pages |
| S10 | Sep 22 – Oct 5 | DGX polish + playbooks 04/05 |
| S11 | Oct 6 – Oct 19 | Integration tests (88) + GitHub Actions CI |
| S12 | Oct 20 – Nov 2 | Benchmarking + paper alignment finalization |
| S13 | Nov 3 – Nov 30 | Paper writing support + `v1.0.0-preprint` tag |
| S14-S16 | Dec – Jan 2027 | Co-author review → *Quantum Machine Intelligence* submission |
