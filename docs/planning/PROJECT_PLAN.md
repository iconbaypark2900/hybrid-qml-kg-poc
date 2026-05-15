# Project Plan — Hybrid QML-KG Biomedical Drug Repurposing Platform

**Created:** 2026-05-15  
**Target:** OSF preprint Q3 2026 · *Quantum Machine Intelligence* submission Q1 2027  
**Strategic goal:** evolve from QML-KG CtD benchmark → omics-informed biomedical drug repurposing platform

---

## Product Objectives (POs / Epics)

| ID | Epic | Gap(s) | Milestone gate |
|----|------|--------|----------------|
| PO-1 | Scientific Reproducibility & Paper Alignment | Gap 12 | M1 |
| PO-2 | Entity Resolution & ID Mapping | Gap 11 | M1 |
| PO-3 | Single-Cell Data Ingestion & QC | Gaps 1–2 | M2 |
| PO-4 | GPU-Accelerated Omics Backend | Gap 3 | M2 |
| PO-5 | Cell-State Discovery & Batch Correction | Gaps 4–5 | M2 |
| PO-6 | Disease Signature Generation | Gap 6 | M3 |
| PO-7 | Perturbation Reversal Engine | Gaps 7–8 | M3 |
| PO-8 | Evidence Fusion & Drug Candidate Ranking | Gap 9 | M4 |
| PO-9 | Clinical & Literature Validation | Gap 10 | M6 |
| PO-10 | Dashboard Evidence Explorer | Gap 15 | M6 |
| PO-11 | NVIDIA-Style Notebook Playbooks | Gap 13 | M7 |
| PO-12 | DGX Spark Operational Polish | Gap 14 | M7 |

---

## Milestones

| ID | Date | Description | Exit criteria |
|----|------|-------------|---------------|
| M0 | 2026-05-19 | Kick-off | Backlog groomed, sprint 1 started |
| M1 | 2026-06-01 | Reproducibility scaffold complete | `docs/paper_alignment/` populated; `entity_resolution/` scaffolded and tested |
| M2 | 2026-06-29 | `single_cell_layer` MVP | Load h5ad, run QC, cluster, GPU/CPU backends both pass smoke test |
| M3 | 2026-07-27 | Biological evidence generated | Disease signatures exported; reversal scores computed for demo disease-drug pairs |
| M4 | 2026-08-10 | Evidence fusion integrated | Full pipeline runs: KG+QML+omics → ranked candidates; KG-only mode still works |
| M5 | 2026-08-31 | OSF preprint submitted | Artifacts committed; preprint draft uploaded |
| M6 | 2026-09-21 | Validation + dashboard complete | Clinical validation labels on top-N candidates; all 10 dashboard pages functional |
| M7 | 2026-10-05 | DGX playbooks + smoke tests | `run_full_repurposing_pipeline.sh` executes end-to-end from clean env |
| M8 | 2026-11-30 | Full system hardened + paper draft | Reproducibility report passes; journal draft in review |
| M9 | 2027-01-31 | *Quantum Machine Intelligence* submitted | Camera-ready submission uploaded |

---

## Sprints

Each sprint is 2 weeks. Owner = Beale (lead) unless noted. Today = 2026-05-15.

---

### Sprint 1 — 2026-05-19 → 2026-06-01 · *Alignment & Scaffolding*

**Goal:** establish shared vocabulary between the paper, the repo, and the new layers before touching any implementation.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S1-01 | Create `docs/paper_alignment/PAPER_IMPLEMENTATION_MAP.md` | PO-1 | Every paper section mapped to existing or missing module |
| S1-02 | Create `docs/paper_alignment/METHOD_REPRODUCTION_CHECKLIST.md` | PO-1 | Checklist of reproducible methods |
| S1-03 | Create `docs/paper_alignment/ASSUMPTIONS_AND_DEVIATIONS.md` | PO-1 | All known assumptions documented |
| S1-04 | Create `docs/paper_alignment/FIGURE_REPRODUCTION_PLAN.md` | PO-1 | Each paper figure mapped to code or "missing" |
| S1-05 | Scaffold `entity_resolution/` package | PO-2 | `__init__.py` + stub modules for gene, disease, compound, ontology mappers |
| S1-06 | Implement `entity_resolution/gene_mapper.py` | PO-2 | HGNC symbol → Hetionet node ID, tested |
| S1-07 | Implement `entity_resolution/disease_mapper.py` | PO-2 | DOID/MONDO → Hetionet node ID, tested |
| S1-08 | Implement `entity_resolution/compound_mapper.py` | PO-2 | DrugBank/PubChem → Hetionet compound ID, tested |

**Sprint 1 acceptance:** M1 docs written; `entity_resolution/` imports cleanly and passes unit tests.

---

### Sprint 2 — 2026-06-02 → 2026-06-15 · *Single-Cell Ingestion*

**Goal:** load real single-cell datasets into the pipeline.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S2-01 | Add `requirements-omics.txt` (scanpy, anndata, harmonypy) | PO-3 | Pinned, installable |
| S2-02 | Create `single_cell_layer/__init__.py` + `metadata_schema.py` | PO-3 | AnnData metadata contract defined |
| S2-03 | Implement `single_cell_layer/ingest_h5ad.py` | PO-3 | Loads `.h5ad`, validates schema, returns AnnData |
| S2-04 | Implement `single_cell_layer/ingest_10x.py` | PO-3 | Loads 10x matrix folder, validates |
| S2-05 | Implement `single_cell_layer/dataset_registry.py` + `dataset_manifest.py` | PO-3 | Manifest JSON written to `artifacts/single_cell/manifests/` |
| S2-06 | Implement `single_cell_layer/loaders.py` (dispatch) | PO-3 | Auto-detects format and calls correct loader |
| S2-07 | Write `config/single_cell_config.yaml` | PO-3 | Schema matches gap doc spec |
| S2-08 | Unit tests for ingest modules | PO-3 | Synthetic fixture + real h5ad smoke test |

**Sprint 2 acceptance:** `python -c "from single_cell_layer import loaders; loaders.load('demo.h5ad')"` succeeds.

---

### Sprint 3 — 2026-06-16 → 2026-06-29 · *QC, Backends & Clustering*

**Goal:** QC + GPU/CPU backend + cell-state discovery. Closes M2.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S3-01 | Implement `single_cell_layer/device_detection.py` | PO-4 | Returns GPU/CPU availability, RAPIDS version |
| S3-02 | Implement `single_cell_layer/backend.py` (dispatcher) | PO-4 | Routes to RAPIDS or Scanpy based on config/hardware |
| S3-03 | Implement `single_cell_layer/cpu_scanpy_backend.py` | PO-4 | Full Scanpy preprocessing path |
| S3-04 | Implement `single_cell_layer/gpu_rapids_backend.py` | PO-4 | RAPIDS-singlecell path; graceful import guard |
| S3-05 | Implement `single_cell_layer/qc.py` + `filtering.py` | PO-3 | Filter cells/genes per config thresholds |
| S3-06 | Implement `single_cell_layer/mitochondrial_metrics.py` | PO-3 | Mito % computed and attached to AnnData |
| S3-07 | Implement `single_cell_layer/doublet_detection.py` | PO-3 | Scrublet-based doublet flags |
| S3-08 | Implement `single_cell_layer/qc_report.py` | PO-3 | Writes `artifacts/single_cell/qc/qc_report.md` + plots |
| S3-09 | Implement `single_cell_layer/dimensionality.py` + `neighbors.py` | PO-5 | PCA → kNN graph |
| S3-10 | Implement `single_cell_layer/clustering.py` + `marker_genes.py` | PO-5 | Leiden clusters + marker genes CSV |
| S3-11 | Implement `single_cell_layer/batch_correction.py` + `harmonization.py` | PO-5 | Harmony correction, before/after metrics |
| S3-12 | Write `playbooks/01_single_cell_preprocessing.ipynb` | PO-11 | Walks steps 1–QC–cluster end-to-end |

**Sprint 3 acceptance:** demo h5ad loads → QC → clusters → `qc_report.md` present; GPU path tested on DGX, CPU fallback tested locally. **M2 closes.**

---

### Sprint 4 — 2026-06-30 → 2026-07-13 · *Disease Signatures*

**Goal:** generate cell-type-specific disease vs. control signatures.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S4-01 | Implement `single_cell_layer/differential_expression.py` | PO-6 | Wilcoxon / t-test DE per cell type |
| S4-02 | Implement `single_cell_layer/disease_signature.py` | PO-6 | Up/down gene lists + ranked genes JSON |
| S4-03 | Implement `single_cell_layer/cell_type_signature.py` | PO-6 | Cell-type-stratified signatures |
| S4-04 | Implement `single_cell_layer/pathway_enrichment.py` | PO-6 | GSEA / ORA against MSigDB/KEGG |
| S4-05 | Implement `single_cell_layer/signature_export.py` | PO-6 | Writes `artifacts/signatures/disease_signature.json` |
| S4-06 | Wire entity resolution into disease signature (DOID → Hetionet node) | PO-2 | Disease ID normalized before export |
| S4-07 | Write `playbooks/02_disease_signature_generation.ipynb` | PO-11 | Guided walkthrough |
| S4-08 | Tests: disease signature output schema validation | PO-6 | Schema matches gap doc JSON spec |

**Sprint 4 acceptance:** `artifacts/signatures/disease_signature.json` written for demo disease with correct schema.

---

### Sprint 5 — 2026-07-14 → 2026-07-27 · *Perturbation Layer & Reversal Scoring*

**Goal:** load compound signatures, compute reversal scores. Closes M3.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S5-01 | Scaffold `perturbation_layer/` package | PO-7 | `__init__.py`, directory |
| S5-02 | Implement `perturbation_layer/lincs_loader.py` | PO-7 | Loads LINCS L1000 gctx/csv |
| S5-03 | Implement `perturbation_layer/cmap_loader.py` | PO-7 | Loads CMap-style signatures |
| S5-04 | Implement `perturbation_layer/drug_signature_builder.py` | PO-7 | Builds per-compound up/down gene lists |
| S5-05 | Implement `perturbation_layer/signature_standardizer.py` | PO-7 | Normalizes gene symbols + compound IDs |
| S5-06 | Implement `perturbation_layer/perturbation_registry.py` | PO-7 | Registry of loaded compound signatures |
| S5-07 | Implement `perturbation_layer/compound_mapping.py` | PO-7 | Maps perturbation IDs → Hetionet compound nodes |
| S5-08 | Write `config/perturbation_config.yaml` | PO-7 | Matches gap doc spec |
| S5-09 | Implement `perturbation_layer/reversal_score.py` | PO-7 | Gene-level rank correlation disease ↔ compound |
| S5-10 | Implement `perturbation_layer/cell_type_reversal.py` | PO-7 | Per-cell-type reversal scores |
| S5-11 | Implement `perturbation_layer/pathway_reversal.py` | PO-7 | Pathway-level reversal scores |
| S5-12 | Implement `perturbation_layer/reversal_features.py` | PO-7 | Model-ready feature vector |
| S5-13 | Write `playbooks/03_drug_signature_reversal.ipynb` | PO-11 | End-to-end reversal walkthrough |
| S5-14 | Write `artifacts/perturbations/reversal_scores.csv` smoke test | PO-7 | File present after pipeline run |

**Sprint 5 acceptance:** reversal scores computed for at least 10 demo compound-disease pairs; `reversal_scores.csv` written. **M3 closes.**

---

### Sprint 6 — 2026-07-28 → 2026-08-10 · *Evidence Fusion*

**Goal:** fuse all evidence streams into the CtD prediction pipeline. Closes M4.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S6-01 | Scaffold `evidence_layer/` package | PO-8 | `__init__.py`, directory |
| S6-02 | Implement `evidence_layer/evidence_schema.py` | PO-8 | Pydantic schema for full feature vector |
| S6-03 | Implement `evidence_layer/feature_fusion.py` | PO-8 | Merges KG + QML + omics reversal features |
| S6-04 | Implement `evidence_layer/evidence_weights.py` | PO-8 | Configurable per-source weights |
| S6-05 | Implement `evidence_layer/confidence_tiering.py` | PO-8 | Tier 1–4 thresholds from config |
| S6-06 | Implement `evidence_layer/explanation_builder.py` | PO-8 | Generates text explanation per candidate |
| S6-07 | Write `config/evidence_fusion_config.yaml` | PO-8 | Matches gap doc spec |
| S6-08 | Wire evidence fusion into `scripts/run_full_repurposing_pipeline.py` | PO-8 | `--mode kg-only` and `--mode kg+omics` flags |
| S6-09 | Evaluation split: KG-only vs KG+omics PR-AUC comparison | PO-8 | Results written to `artifacts/predictions/` |
| S6-10 | Write `playbooks/06_candidate_ranking_and_validation.ipynb` | PO-11 | Guided ranking walkthrough |

**Sprint 6 acceptance:** full pipeline runs in both modes; delta PR-AUC logged. **M4 closes.**

---

### Sprint 7 — 2026-08-11 → 2026-08-31 · *OSF Preprint Prep*

**Goal:** harden artifacts, reproducibility pass, submit preprint. Closes M5.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S7-01 | Complete `docs/paper_alignment/REPRODUCIBILITY_REPORT.md` | PO-1 | Every result linked to script + artifact |
| S7-02 | Complete `docs/paper_alignment/TABLE_REPRODUCTION_PLAN.md` | PO-1 | All tables reproducible or noted as extensions |
| S7-03 | Run full pipeline from scratch on clean env → record hash | PO-1 | `artifacts/pipeline_run_hash.txt` committed |
| S7-04 | Write `evidence_layer/evidence_report.py` | PO-8 | HTML/MD report of top-N candidates |
| S7-05 | Write `perturbation_layer/reversal_report.py` | PO-7 | Reversal breakdown report |
| S7-06 | Final `artifacts/reports/final_repurposing_report.md` generated | PO-8 | Passes automated schema checks |
| S7-07 | OSF preprint draft uploaded | — | DOI returned |

**Sprint 7 acceptance:** preprint link exists. **M5 closes.**

---

### Sprint 8 — 2026-08-25 → 2026-09-07 · *Clinical & Literature Validation*

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S8-01 | Scaffold `validation_layer/` package | PO-9 | `__init__.py` |
| S8-02 | Implement `validation_layer/known_indications_validator.py` | PO-9 | DrugCentral / ChEMBL indications lookup |
| S8-03 | Implement `validation_layer/clinical_trials_validator.py` | PO-9 | ClinicalTrials.gov API; phase label attached |
| S8-04 | Implement `validation_layer/literature_validator.py` | PO-9 | PubMed co-occurrence count |
| S8-05 | Implement `validation_layer/drugbank_mapper.py` | PO-9 | DrugBank ID → known indications |
| S8-06 | Implement `validation_layer/opentargets_mapper.py` | PO-9 | Open Targets evidence score |
| S8-07 | Implement `validation_layer/validation_report.py` | PO-9 | JSON + MD report per candidate |
| S8-08 | Wire validation into full pipeline | PO-9 | `--validate` flag calls validation layer |

---

### Sprint 9 — 2026-09-08 → 2026-09-21 · *Dashboard Evidence Explorer*

**Goal:** upgrade dashboard from model demo to biomedical evidence explorer. Closes M6.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S9-01 | Add `benchmarking/components/` package | PO-10 | Directory + `__init__.py` |
| S9-02 | Implement `evidence_card.py` | PO-10 | Streamlit component: score + tier + evidence bullets |
| S9-03 | Implement `signature_view.py` | PO-10 | Disease signature heatmap / gene table |
| S9-04 | Implement `reversal_view.py` | PO-10 | Drug reversal score breakdown by cell type + pathway |
| S9-05 | Implement `clinical_validation_view.py` | PO-10 | Trial phase badge + literature count |
| S9-06 | Dashboard page: Disease Explorer | PO-10 | Select disease → top candidates listed |
| S9-07 | Dashboard page: Candidate Drug Rankings | PO-10 | Sortable, filterable ranked table |
| S9-08 | Dashboard page: Evidence Breakdown | PO-10 | Per-candidate evidence card |
| S9-09 | Dashboard page: Quantum/Classical Comparison | PO-10 | KG-only vs KG+omics scores side by side |
| S9-10 | Dashboard page: Clinical Validation View | PO-10 | Validation status for top-N |
| S9-11 | Write `playbooks/07_dashboard_demo.ipynb` | PO-11 | Step-by-step dashboard launch guide |

**Sprint 9 acceptance:** all 10 dashboard pages render; evidence card displays for 3 demo candidates. **M6 closes.**

---

### Sprint 10 — 2026-09-22 → 2026-10-05 · *DGX Spark & Playbooks*

**Goal:** match NVIDIA workflow quality for one-command execution. Closes M7.

| # | Story | PO | Deliverable |
|---|-------|----|-------------|
| S10-01 | Create `scripts/dgx/` directory | PO-12 | Directory |
| S10-02 | Write `scripts/dgx/check_environment.sh` | PO-12 | Checks CUDA, RAPIDS, Python, config |
| S10-03 | Write `scripts/dgx/install_gpu_omics.sh` | PO-12 | Installs `requirements-omics.txt` with GPU extras |
| S10-04 | Write `scripts/dgx/launch_jupyter.sh` | PO-12 | Launches Jupyter on DGX port |
| S10-05 | Write `scripts/dgx/launch_dashboard.sh` | PO-12 | Launches Streamlit dashboard |
| S10-06 | Write `scripts/dgx/run_smoke_test.sh` | PO-12 | E2E smoke with synthetic data, exits non-zero on failure |
| S10-07 | Write `scripts/dgx/run_single_cell_pipeline.sh` | PO-12 | Runs single-cell layer only |
| S10-08 | Write `scripts/dgx/run_full_repurposing_pipeline.sh` | PO-12 | Runs full pipeline end-to-end |
| S10-09 | Write `scripts/dgx/collect_artifacts.sh` | PO-12 | Bundles `artifacts/` with manifest + hashes |
| S10-10 | Write `playbooks/00_environment_check.ipynb` | PO-11 | Notebook version of env check |
| S10-11 | Write `playbooks/04_kg_embedding_training.ipynb` | PO-11 | KG embedding training walkthrough |
| S10-12 | Write `playbooks/05_hybrid_qml_prediction.ipynb` | PO-11 | QML prediction walkthrough |
| S10-13 | Write `docs/deployment/DGX_SINGLE_CELL_REPURPOSING.md` | PO-12 | DGX setup and launch guide |

**Sprint 10 acceptance:** `./scripts/dgx/run_smoke_test.sh` exits 0 from clean env. **M7 closes.**

---

### Sprint 11 — 2026-10-06 → 2026-10-19 · *Integration Testing & Reproducibility*

| # | Story | Deliverable |
|---|-------|-------------|
| S11-01 | Integration test: full pipeline (KG-only) | Passes CI |
| S11-02 | Integration test: full pipeline (KG+omics) | Passes CI |
| S11-03 | Integration test: validation layer | Known indication found for at least 1 demo pair |
| S11-04 | Integration test: dashboard loads all pages | No import errors |
| S11-05 | `entity_resolution` mapping report for demo run | `mapping_report.md` written |
| S11-06 | `single_cell_layer/integration_report.py` | Before/after batch correction metrics written |
| S11-07 | `entity_resolution/ontology_mapper.py` + `synonym_resolver.py` | Synonym resolution tested |
| S11-08 | Cross-layer ID consistency audit | All gene/disease/compound IDs trace to Hetionet nodes |

---

### Sprint 12 — 2026-10-20 → 2026-11-02 · *Benchmarking & Paper Alignment Finalization*

| # | Story | Deliverable |
|---|-------|-------------|
| S12-01 | Run full benchmarking suite (all model configs) | Results in `benchmarking/` |
| S12-02 | Paper alignment: confirm all figures reproducible | `FIGURE_REPRODUCTION_PLAN.md` updated |
| S12-03 | Ablation: omics evidence layer on/off PR-AUC | Written to `results/` |
| S12-04 | `entity_resolution/mapping_report.py` | Generates mapping quality stats |
| S12-05 | Requirements audit: pin all omics/validation deps | `requirements-omics.txt` and `requirements-full.txt` locked |
| S12-06 | CI pipeline extended to cover new layers | GitHub Actions updated |

---

### Sprint 13 — 2026-11-03 → 2026-11-30 · *Paper Writing Support & Hardening*

| # | Story | Deliverable |
|---|-------|-------------|
| S13-01 | Write `docs/paper_alignment/REPRODUCIBILITY_REPORT.md` (final) | Complete |
| S13-02 | Figures for paper: cell-state UMAPs, reversal heatmaps, evidence bar charts | In `figures/` |
| S13-03 | Tables for paper: top-20 candidates with validation status | In `results/` |
| S13-04 | Manuscript draft sections: Methods (new layers) | In `docs/paper_qGG_*.tex` |
| S13-05 | Final smoke test from clean checkout | Passes |
| S13-06 | Tag `v1.0.0-preprint` | Git tag |

**Sprint 13 acceptance:** paper draft ready for co-author review. **M8 closes.**

---

### Sprints 14–16 — 2026-12-01 → 2027-01-31 · *Paper Revision & Submission*

Co-author review → revisions → journal submission. **M9 closes.**

---

## Backlog (Full Prioritized)

Priority: **P0** = blocks M2/M3, must start Sprint 2–5 · **P1** = needed for OSF · **P2** = needed for journal · **P3** = polish

### PO-1: Scientific Reproducibility

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-001 | `docs/paper_alignment/PAPER_IMPLEMENTATION_MAP.md` | P0 | S1 |
| B-002 | `docs/paper_alignment/METHOD_REPRODUCTION_CHECKLIST.md` | P0 | S1 |
| B-003 | `docs/paper_alignment/ASSUMPTIONS_AND_DEVIATIONS.md` | P0 | S1 |
| B-004 | `docs/paper_alignment/FIGURE_REPRODUCTION_PLAN.md` | P1 | S1 |
| B-005 | `docs/paper_alignment/TABLE_REPRODUCTION_PLAN.md` | P1 | S7 |
| B-006 | `docs/paper_alignment/REPRODUCIBILITY_REPORT.md` (final) | P1 | S7, S13 |

### PO-2: Entity Resolution

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-010 | `entity_resolution/__init__.py` | P0 | S1 |
| B-011 | `entity_resolution/gene_mapper.py` | P0 | S1 |
| B-012 | `entity_resolution/disease_mapper.py` | P0 | S1 |
| B-013 | `entity_resolution/compound_mapper.py` | P0 | S1 |
| B-014 | `entity_resolution/ontology_mapper.py` | P1 | S11 |
| B-015 | `entity_resolution/synonym_resolver.py` | P1 | S11 |
| B-016 | `entity_resolution/mapping_report.py` | P2 | S12 |

### PO-3: Single-Cell Ingestion & QC

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-020 | `requirements-omics.txt` | P0 | S2 |
| B-021 | `single_cell_layer/__init__.py` | P0 | S2 |
| B-022 | `single_cell_layer/metadata_schema.py` | P0 | S2 |
| B-023 | `single_cell_layer/ingest_h5ad.py` | P0 | S2 |
| B-024 | `single_cell_layer/ingest_10x.py` | P0 | S2 |
| B-025 | `single_cell_layer/dataset_registry.py` | P0 | S2 |
| B-026 | `single_cell_layer/dataset_manifest.py` | P0 | S2 |
| B-027 | `single_cell_layer/loaders.py` | P0 | S2 |
| B-028 | `config/single_cell_config.yaml` | P0 | S2 |
| B-029 | `single_cell_layer/qc.py` | P0 | S3 |
| B-030 | `single_cell_layer/filtering.py` | P0 | S3 |
| B-031 | `single_cell_layer/mitochondrial_metrics.py` | P0 | S3 |
| B-032 | `single_cell_layer/doublet_detection.py` | P0 | S3 |
| B-033 | `single_cell_layer/qc_report.py` | P0 | S3 |

### PO-4: GPU-Accelerated Omics Backend

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-040 | `single_cell_layer/device_detection.py` | P0 | S3 |
| B-041 | `single_cell_layer/backend.py` | P0 | S3 |
| B-042 | `single_cell_layer/cpu_scanpy_backend.py` | P0 | S3 |
| B-043 | `single_cell_layer/gpu_rapids_backend.py` | P0 | S3 |

### PO-5: Cell-State Discovery & Batch Correction

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-050 | `single_cell_layer/dimensionality.py` | P0 | S3 |
| B-051 | `single_cell_layer/neighbors.py` | P0 | S3 |
| B-052 | `single_cell_layer/clustering.py` | P0 | S3 |
| B-053 | `single_cell_layer/cell_state_embedding.py` | P1 | S3 |
| B-054 | `single_cell_layer/marker_genes.py` | P0 | S3 |
| B-055 | `single_cell_layer/batch_correction.py` | P0 | S3 |
| B-056 | `single_cell_layer/harmonization.py` | P0 | S3 |
| B-057 | `single_cell_layer/batch_metrics.py` | P1 | S11 |
| B-058 | `single_cell_layer/integration_report.py` | P1 | S11 |

### PO-6: Disease Signature Generation

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-060 | `single_cell_layer/differential_expression.py` | P0 | S4 |
| B-061 | `single_cell_layer/disease_signature.py` | P0 | S4 |
| B-062 | `single_cell_layer/cell_type_signature.py` | P0 | S4 |
| B-063 | `single_cell_layer/pathway_enrichment.py` | P0 | S4 |
| B-064 | `single_cell_layer/signature_export.py` | P0 | S4 |

### PO-7: Perturbation Reversal Engine

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-070 | `perturbation_layer/__init__.py` | P0 | S5 |
| B-071 | `perturbation_layer/lincs_loader.py` | P0 | S5 |
| B-072 | `perturbation_layer/cmap_loader.py` | P1 | S5 |
| B-073 | `perturbation_layer/drug_signature_builder.py` | P0 | S5 |
| B-074 | `perturbation_layer/signature_standardizer.py` | P0 | S5 |
| B-075 | `perturbation_layer/perturbation_registry.py` | P0 | S5 |
| B-076 | `perturbation_layer/compound_mapping.py` | P0 | S5 |
| B-077 | `config/perturbation_config.yaml` | P0 | S5 |
| B-078 | `perturbation_layer/reversal_score.py` | P0 | S5 |
| B-079 | `perturbation_layer/cell_type_reversal.py` | P0 | S5 |
| B-080 | `perturbation_layer/pathway_reversal.py` | P0 | S5 |
| B-081 | `perturbation_layer/reversal_features.py` | P0 | S5 |
| B-082 | `perturbation_layer/reversal_report.py` | P1 | S7 |

### PO-8: Evidence Fusion & Drug Candidate Ranking

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-090 | `evidence_layer/__init__.py` | P0 | S6 |
| B-091 | `evidence_layer/evidence_schema.py` | P0 | S6 |
| B-092 | `evidence_layer/feature_fusion.py` | P0 | S6 |
| B-093 | `evidence_layer/evidence_weights.py` | P0 | S6 |
| B-094 | `evidence_layer/confidence_tiering.py` | P0 | S6 |
| B-095 | `evidence_layer/explanation_builder.py` | P1 | S6 |
| B-096 | `evidence_layer/evidence_report.py` | P1 | S7 |
| B-097 | `config/evidence_fusion_config.yaml` | P0 | S6 |
| B-098 | `scripts/run_full_repurposing_pipeline.py` — `--mode` flag | P0 | S6 |
| B-099 | PR-AUC ablation: KG-only vs KG+omics | P1 | S6, S12 |

### PO-9: Clinical & Literature Validation

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-100 | `validation_layer/__init__.py` | P1 | S8 |
| B-101 | `validation_layer/known_indications_validator.py` | P1 | S8 |
| B-102 | `validation_layer/clinical_trials_validator.py` | P1 | S8 |
| B-103 | `validation_layer/literature_validator.py` | P1 | S8 |
| B-104 | `validation_layer/drugbank_mapper.py` | P1 | S8 |
| B-105 | `validation_layer/opentargets_mapper.py` | P2 | S8 |
| B-106 | `validation_layer/validation_report.py` | P1 | S8 |

### PO-10: Dashboard Evidence Explorer

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-110 | `benchmarking/components/__init__.py` | P1 | S9 |
| B-111 | `benchmarking/components/evidence_card.py` | P1 | S9 |
| B-112 | `benchmarking/components/signature_view.py` | P1 | S9 |
| B-113 | `benchmarking/components/reversal_view.py` | P1 | S9 |
| B-114 | `benchmarking/components/clinical_validation_view.py` | P1 | S9 |
| B-115 | Dashboard: Disease Explorer page | P1 | S9 |
| B-116 | Dashboard: Candidate Drug Rankings page | P1 | S9 |
| B-117 | Dashboard: Evidence Breakdown page | P1 | S9 |
| B-118 | Dashboard: Quantum/Classical Comparison page | P2 | S9 |
| B-119 | Dashboard: Clinical Validation View page | P2 | S9 |

### PO-11: NVIDIA-Style Notebook Playbooks

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-120 | `playbooks/00_environment_check.ipynb` | P1 | S10 |
| B-121 | `playbooks/01_single_cell_preprocessing.ipynb` | P1 | S3 |
| B-122 | `playbooks/02_disease_signature_generation.ipynb` | P1 | S4 |
| B-123 | `playbooks/03_drug_signature_reversal.ipynb` | P1 | S5 |
| B-124 | `playbooks/04_kg_embedding_training.ipynb` | P2 | S10 |
| B-125 | `playbooks/05_hybrid_qml_prediction.ipynb` | P2 | S10 |
| B-126 | `playbooks/06_candidate_ranking_and_validation.ipynb` | P1 | S6 |
| B-127 | `playbooks/07_dashboard_demo.ipynb` | P1 | S9 |

### PO-12: DGX Spark Operational Polish

| ID | Item | Priority | Sprint |
|----|------|----------|--------|
| B-130 | `scripts/dgx/check_environment.sh` | P1 | S10 |
| B-131 | `scripts/dgx/install_gpu_omics.sh` | P1 | S10 |
| B-132 | `scripts/dgx/launch_jupyter.sh` | P1 | S10 |
| B-133 | `scripts/dgx/launch_dashboard.sh` | P1 | S10 |
| B-134 | `scripts/dgx/run_smoke_test.sh` | P1 | S10 |
| B-135 | `scripts/dgx/run_single_cell_pipeline.sh` | P1 | S10 |
| B-136 | `scripts/dgx/run_full_repurposing_pipeline.sh` | P1 | S10 |
| B-137 | `scripts/dgx/collect_artifacts.sh` | P1 | S10 |
| B-138 | `docs/deployment/DGX_SINGLE_CELL_REPURPOSING.md` | P1 | S10 |
| B-139 | `requirements-full.txt` (all layers) | P1 | S12 |

---

## Sprint Velocity Summary

| Sprint | Dates | Stories | PO focus | Milestone |
|--------|-------|---------|----------|-----------|
| S1 | May 19 – Jun 1 | 8 | PO-1, PO-2 | → M1 |
| S2 | Jun 2 – Jun 15 | 8 | PO-3 | — |
| S3 | Jun 16 – Jun 29 | 12 | PO-3, PO-4, PO-5, PO-11 | → M2 |
| S4 | Jun 30 – Jul 13 | 8 | PO-6, PO-11 | — |
| S5 | Jul 14 – Jul 27 | 14 | PO-7, PO-11 | → M3 |
| S6 | Jul 28 – Aug 10 | 10 | PO-8, PO-11 | → M4 |
| S7 | Aug 11 – Aug 31 | 7 | PO-1, PO-7, PO-8 | → M5 |
| S8 | Aug 25 – Sep 7 | 8 | PO-9 | — |
| S9 | Sep 8 – Sep 21 | 11 | PO-10, PO-11 | → M6 |
| S10 | Sep 22 – Oct 5 | 13 | PO-11, PO-12 | → M7 |
| S11 | Oct 6 – Oct 19 | 8 | PO-2, PO-5 | — |
| S12 | Oct 20 – Nov 2 | 6 | PO-1, PO-8, PO-12 | — |
| S13 | Nov 3 – Nov 30 | 6 | PO-1 (paper) | → M8 |
| S14–16 | Dec – Jan 2027 | — | Manuscript | → M9 |

**Total backlog items: ~140 stories across 12 product objectives.**
