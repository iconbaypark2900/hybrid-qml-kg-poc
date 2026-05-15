# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0-preprint] — 2026-11-30 (planned, M8)

OSF preprint cut. Closes the M8 milestone in `docs/planning/PROJECT_PLAN.md`.

### Added — Sprint 13 (paper writing support + scientific poster)
- **Scientific poster** (`docs/poster/`) — A0 portrait tikzposter LaTeX template
  rendering an 8-section walkthrough (motivation, architecture, headline result,
  methods, per-disease results, reproducibility, limitations, future work).
  Compiles to a 162 KB PDF via `bash scripts/build_poster.sh`.
- **External API cache** (`validation_layer/api_cache.py`) — on-disk JSON cache
  for ClinicalTrials.gov / PubMed / Open Targets / DrugBank queries; enables
  hermetic table builds in CI. Per-provider env vars (`CT_API_CACHE`,
  `PUBMED_CACHE`, `OT_CACHE`, `DRUGBANK_CACHE`, or
  `VALIDATION_API_CACHE_ALL`).
- **Table 3 LaTeX** (`docs/tables/table3_sensitivity.tex`) — consolidated from
  `docs/results/sensitivity_pauli_vs_zz.md` and `sensitivity_stacking_vs_weighted.md`.
  Builder integration in `scripts/build_paper_tables.py --table 3`.
- **OSF bundle script** (`scripts/prepare_osf_bundle.sh`) — collects
  artifacts, paper LaTeX, poster, and reproducibility docs into a single
  tarball with SHA256 manifest, ready for OSF upload.
- **Release notes** (`docs/RELEASE_NOTES_v1.0.0-preprint.md`) — full preprint
  cut notes with reviewer-facing summary.

### Added — Sprint 12 (benchmarking + paper alignment)
- `scripts/compare_pipeline_modes.py` — kg-only vs kg+omics ΔPR-AUC comparator
- `scripts/aggregate_qc_summary.py` — Table 6 builder
- `scripts/build_signature_catalog.py` — Table 7 builder
- `scripts/build_paper_tables.py` — single LaTeX entry point for tables 1, 4-9
- `tests/test_paper_builders.py` (11 tests)

### Added — Sprint 11 (integration tests + CI)
- `tests/test_entity_resolution.py`, `test_perturbation_layer.py`,
  `test_evidence_layer.py`, `test_validation_layer.py`,
  `test_single_cell_layer.py`, `test_pipeline_integration.py`,
  `test_dashboard_components.py` — 88 integration tests
- `pyproject.toml` — pytest configuration excluding optional-service tests
- `.github/workflows/ci.yml` — three-job CI pipeline (smoke, integration, reproducibility)

### Added — Sprint 10 (DGX polish + remaining playbooks)
- `playbooks/04_kg_embedding_training.ipynb`
- `playbooks/05_hybrid_qml_prediction.ipynb`
- `docs/deployment/DGX_RUNBOOK.md` — full operational guide
- Extended `scripts/dgx/run_smoke_test.sh` from 5 to 6 steps

### Added — Sprint 9 (dashboard evidence pages)
- 4 new Streamlit pages: Disease Explorer, Drug Candidate Rankings,
  Evidence Breakdown, Clinical Validation
- `benchmarking/components/data_loader.py`
- `playbooks/07_dashboard_demo.ipynb`

### Added — Sprint 6-8 (orchestrator + missing modules + playbooks)
- `scripts/run_full_repurposing_pipeline.py` — end-to-end orchestrator with
  `--mode {kg-only,kg+omics}` and `--validate` flags
- `perturbation_layer/cmap_loader.py`
- `single_cell_layer/cell_type_signature.py`
- `validation_layer/drugbank_mapper.py`, `validation_layer/opentargets_mapper.py`
- `docs/paper_alignment/REPRODUCIBILITY_REPORT.md`
- `docs/paper_alignment/TABLE_REPRODUCTION_PLAN.md`
- Playbooks 00, 01, 02, 03, 06

### Added — Sprint 1-5 (bio layers + entity resolution)
- `entity_resolution/` package (HetionetResolver + 4 mappers)
- `single_cell_layer/` package (Scanpy CPU + RAPIDS GPU paths)
- `perturbation_layer/` package (LINCS + reversal scoring)
- `evidence_layer/` package (10-feature fusion + tiering)
- `validation_layer/` package (known indications + ClinicalTrials.gov)
- `benchmarking/components/` Streamlit components
- `scripts/dgx/` operational scripts (8 files)
- `docs/paper_alignment/` (4 docs)
- `config/{single_cell,perturbation,evidence_fusion,entity_resolution}_config.yaml`
- `requirements-omics.txt`

### Changed
- Headline preregistered config locked: PauliFeatureMap (reps=2) + RotatE
  (32d → 5d via PCA) + hard degree-corrupt negatives + stacking ensemble.
- PR-AUC 0.7987 reproduced against the sealed test set.

### Reproducibility
- 99 integration tests passing (51% line coverage on bio layers).
- `bash scripts/dgx/run_smoke_test.sh` exits 0 in < 60 s from a clean venv.
- All `TABLE_REPRODUCTION_PLAN.md` blocking TODOs cleared.

## [0.1.0] — 2026-05-15 (worktree merge to main)

Initial scaffold + bio layers merged (Sprints 1-5). See git history under
`feat/sprint*` branches for the per-sprint commit log.

[Unreleased]: https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc/compare/v1.0.0-preprint...HEAD
[1.0.0-preprint]: https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc/releases/tag/v1.0.0-preprint
[0.1.0]: https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc/releases/tag/v0.1.0
