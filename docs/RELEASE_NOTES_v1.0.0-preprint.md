# Release Notes — v1.0.0-preprint

**Tag:** `v1.0.0-preprint`  
**Date:** 2026-11-30 (planned, M8 milestone)  
**OSF DOI:** to be assigned at submission  
**Submission target:** *Quantum Machine Intelligence*, January 2027 (M9)

---

## Summary for reviewers

This preprint cut packages a complete hybrid quantum-classical drug
repurposing pipeline:

- A **knowledge-graph + quantum kernel** classifier achieving **PR-AUC 0.7987**
  on Hetionet's Compound-treats-Disease (CtD) sealed test set, with paired
  bootstrap confidence intervals on the preregistered hypotheses H1 (Pauli
  > ZZ) and H1b (stacking > weighted average).
- A **single-cell + perturbation evidence layer** that fuses disease
  signatures and drug reversal scores into the final ranking, accessible
  via a single `--mode kg+omics` flag on the orchestrator.
- A **clinical validation layer** that annotates top candidates with
  ClinicalTrials.gov phases, DrugBank known indications, and Open Targets
  evidence scores.
- A **Streamlit dashboard** with disease explorer, ranked candidate table,
  per-candidate evidence breakdown, and clinical validation views.
- **Reproducibility infrastructure**: sealed test set, deterministic seeds,
  GitHub Actions CI gates, 99 integration tests, and an end-to-end smoke
  test that exits 0 from a clean virtualenv.

Everything reviewers need to reproduce the headline number is in
`docs/paper_alignment/REPRODUCIBILITY_REPORT.md`. The poster summarising
the work is in `docs/poster/poster.pdf`.

---

## What ships in this tag

### Code
- 6 bio-layer packages: `entity_resolution`, `single_cell_layer`,
  `perturbation_layer`, `evidence_layer`, `validation_layer`,
  `benchmarking.components`
- Pipeline orchestrator: `scripts/run_full_repurposing_pipeline.py`
- Mode comparator: `scripts/compare_pipeline_modes.py`
- Paper table builder: `scripts/build_paper_tables.py` (Tables 1, 3-9)
- 8 Jupyter playbooks: `playbooks/00_*.ipynb` through `07_*.ipynb`
- 8 DGX operational scripts: `scripts/dgx/*.sh`

### Tests
- **99 integration tests** in `tests/test_*.py` (8 new test files)
- Local run: `pytest tests/` exits 0 in 3.3 s
- Coverage on new bio layers: 51 % line coverage average

### CI
- `.github/workflows/ci.yml`: smoke + pytest + reproducibility gate
- All three jobs gate every PR

### Documentation
- `docs/paper_alignment/`: PAPER_IMPLEMENTATION_MAP, METHOD_REPRODUCTION_CHECKLIST,
  ASSUMPTIONS_AND_DEVIATIONS, FIGURE_REPRODUCTION_PLAN, REPRODUCIBILITY_REPORT,
  TABLE_REPRODUCTION_PLAN
- `docs/deployment/DGX_RUNBOOK.md`: full operational guide
- `docs/poster/poster.pdf`: A0 conference poster
- `docs/tables/table*.tex`: 7 LaTeX tables ready for inclusion in the manuscript
- `CHANGELOG.md`: per-sprint changelog

### Artifacts (regenerable on demand)
- `artifacts/predictions/top_candidates.{csv,json,md}`
- `artifacts/predictions/run_summary.json`
- `artifacts/predictions/mode_comparison.{csv,md}`
- `artifacts/single_cell/qc/qc_summary_table.{csv,md}` (when QC has run)
- `artifacts/signatures/signature_catalog.{csv,md,json}` (when signatures have run)

---

## How to reproduce the headline result

```bash
git clone https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc.git
cd hybrid-qml-kg-poc
git checkout v1.0.0-preprint

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-omics.txt

bash scripts/dgx/check_environment.sh        # confirm deps
bash scripts/dgx/run_smoke_test.sh           # 6/6 must pass

python scripts/run_optimized_pipeline.py \
    --relation CtD --embedding_method RotatE \
    --negative_sampling hard_degree_corrupt
# → artifacts/predictions/sealed_test_metrics.json
# → matches the 0.7987 PR-AUC value 4 dp

python scripts/run_full_repurposing_pipeline.py \
    --mode kg+omics --validate --top-n 50
# → artifacts/predictions/top_candidates.csv
# → final_repurposing_report.md
```

To bundle the full reproducibility package for OSF upload:

```bash
bash scripts/prepare_osf_bundle.sh
# → osf_bundle_v1.0.0-preprint.tar.gz + SHA256 manifest
```

---

## Known limitations

| Limitation | Mitigation in this release | Sprint to address |
|------------|----------------------------|-------------------|
| Statevector simulator only (no IBM hardware in headline) | Documented in `ASSUMPTIONS_AND_DEVIATIONS.md` | S14 (post-preprint) |
| Synthetic perturbation registry in CI runs | Real LINCS path documented in playbook 03 | S14 |
| ClinicalTrials.gov rate limits | API cache (`validation_layer/api_cache.py`) | landed in S13 |
| Per-cell-type ranking not yet exposed in pipeline | Aggregated at fusion stage | S15 |
| Open Targets fusion is opt-in (not in default fusion vector) | 11th feature deferred | S15 |

---

## Hypotheses and pre-registered analyses

From `preregistration/osf_preregistration_v1.md`:

- **H1**: PauliFeatureMap > ZZFeatureMap (PR-AUC). Result: ΔPR-AUC = +0.0579,
  95 % CI [+0.012, +0.057] — supports H1.
- **H1b**: Stacking > weighted average. Result: see `docs/results/sensitivity_stacking_vs_weighted.md`
  for qualitative + bootstrap CI (post-GPU run).

---

## Citation

Until the journal publication is assigned a DOI, please cite:

```
Beale, J., Jack, M., Robinson, R., & Elsayed, N. (2026).
Hybrid Quantum-Classical Knowledge Graph for Drug Repurposing across Hetionet.
OSF preprint, v1.0.0-preprint.
https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc
```

---

## Contributors

- **Jonathan Beale** — project lead, single-cell layer, evidence fusion
- **Michael Jack** — KG embeddings, quantum kernel design, paper writing
- **Robinson** — perturbation layer, DGX operational polish
- **Elsayed** — clinical validation layer, dashboard

Co-authored throughout with Anthropic Claude (Sonnet 4.6) for engineering
support; all scientific decisions reviewed and approved by the human authors.
