# Reproducibility Report

**Project:** hybrid-qml-kg-poc — Hybrid Quantum-Classical Drug Repurposing  
**Target venue:** OSF preprint (Q3 2026) → *Quantum Machine Intelligence* (Q1 2027)  
**Report status:** Draft — updated each sprint, finalized at M5 (2026-08-31).

---

## 1. Scope

This document maps every quantitative result in the paper to (a) the script
that produced it, (b) the input data, (c) the environment, and (d) the
artifact filename written under `artifacts/`. The goal: a reader with the
repository and a clean environment can reproduce any number, table, or
figure end-to-end.

---

## 2. Environment

| Component | Pin | Source |
|-----------|-----|--------|
| Python | 3.12.3 | `requirements.txt`, `requirements-omics.txt` |
| numpy | 2.4.x | `requirements.txt` |
| pandas | 2.3.x | `requirements.txt` |
| scikit-learn | ≥1.4 | `requirements.txt` |
| qiskit / qiskit-machine-learning | aligned with `requirements.txt` | core QML |
| qiskit-aer (GPU) | optional | `requirements-gpu.txt` |
| scanpy / anndata | ≥1.10 / ≥0.10 | `requirements-omics.txt` |
| rapids-singlecell | optional (CUDA 12) | `requirements-omics.txt` |
| harmonypy / scrublet / gseapy | optional | `requirements-omics.txt` |
| Hetionet snapshot | hash recorded by `scripts/record_hetionet_hash.py` | `docs/reproducibility/hetionet_snapshot.md` |

Hardware tiers and platform-specific notes are in
[`ASSUMPTIONS_AND_DEVIATIONS.md`](ASSUMPTIONS_AND_DEVIATIONS.md).

---

## 3. Result → Artifact Map

### 3.1 Headline result (PR-AUC 0.7987)

| Field | Value |
|-------|-------|
| Script | `scripts/run_optimized_pipeline.py` |
| Config | Pauli feature map + RotatE + hard negatives + stacking ensemble (locked in `utils/preregistered_constants.py`) |
| Input | `data/hetionet-v1.0-edges.tsv`, `data/hetionet-v1.0-nodes.tsv` |
| Output | `artifacts/predictions/sealed_test_metrics.json` |
| Verification | `python -m utils.sealed_test_set --verify` |
| Bootstrap CI | `scripts/run_bootstrap_ci.py` → `artifacts/predictions/bootstrap_ci.json` |

### 3.2 Supplementary sensitivity analyses

| Analysis | Script | Output |
|----------|--------|--------|
| Pauli vs ZZ feature map | `scripts/run_optimized_pipeline.py --qsvc_feature_map_type ZZ` | `docs/results/sensitivity_pauli_vs_zz.md` |
| Stacking vs weighted ensemble | `scripts/run_optimized_pipeline.py --ensemble_method weighted` | `docs/results/sensitivity_stacking_vs_weighted.md` |
| ZNE (linear vs Richardson) | `scripts/self_test_mitigation.py` | `docs/results/zne_implementation_audit.md` |
| Negative controls | `benchmarking/negative_controls.py` | `artifacts/evaluations/negative_controls.json` |

### 3.3 New omics layers (Sprints 2–6)

| Layer | Driver script / module | Artifact |
|-------|------------------------|---------|
| Single-cell ingest + QC | `single_cell_layer/loaders.py`, `qc.py` | `artifacts/single_cell/qc/qc_report.md` |
| Disease signatures | `single_cell_layer/disease_signature.py` | `artifacts/signatures/disease_signature.json` |
| Cell-type stratified signatures | `single_cell_layer/cell_type_signature.py` | `artifacts/signatures/cell_type_signatures.json` |
| Reversal scores | `perturbation_layer/reversal_score.py` | `artifacts/perturbations/reversal_scores.csv` |
| Evidence fusion (KG-only) | `scripts/run_full_repurposing_pipeline.py --mode kg-only` | `artifacts/predictions/top_candidates.csv` |
| Evidence fusion (KG+omics) | `scripts/run_full_repurposing_pipeline.py --mode kg+omics` | `artifacts/predictions/top_candidates.csv` |
| Clinical validation | `scripts/run_full_repurposing_pipeline.py --validate` | `artifacts/predictions/run_summary.json` (clinical_evidence_score populated) |

---

## 4. End-to-end reproduction recipe

```bash
# 1. Clone and create environment
git clone https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc.git
cd hybrid-qml-kg-poc
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-omics.txt

# 2. Verify environment
bash scripts/dgx/check_environment.sh

# 3. Smoke test (synthetic data, < 1 min)
bash scripts/dgx/run_smoke_test.sh

# 4. Headline result (reproduces PR-AUC 0.7987)
python scripts/run_optimized_pipeline.py \
    --relation CtD --embedding_method RotatE --negative_sampling hard

# 5. Bootstrap CI (paired comparison)
python scripts/run_bootstrap_ci.py --n_bootstrap 1000

# 6. Full repurposing pipeline (KG-only — baseline preserved)
python scripts/run_full_repurposing_pipeline.py --mode kg-only

# 7. Full repurposing pipeline (KG + omics)
python scripts/run_full_repurposing_pipeline.py --mode kg+omics --validate
```

Each step writes to `artifacts/` so subsequent runs can verify deterministic
output. Run hashes are recorded by `scripts/record_hetionet_hash.py` (data)
and emitted to `artifacts/pipeline_run_hash.txt` (code + config).

---

## 5. Deterministic seeds

All randomness goes through `utils.reproducibility.set_global_seed(...)`,
which seeds numpy, Python `random`, PyTorch (if installed), and Qiskit's
backend RNGs. Default seed: 42 (locked in `utils/preregistered_constants.py`).

The bootstrap CI driver uses a separate seed (declared in `osf_preregistration_v1.md`)
so the headline result and the confidence interval are reproducible
independently.

---

## 6. Known sources of non-determinism

| Source | Mitigation |
|--------|-----------|
| GPU-accelerated kernel computation (Qiskit Aer GPU) | Statevector simulation is used in CI; GPU statevector matches CPU bit-for-bit |
| RAPIDS `leiden` clustering | Documented in `ASSUMPTIONS_AND_DEVIATIONS.md`; falls back to scanpy Leiden if unavailable |
| ClinicalTrials.gov API responses | `query_clinical_trials` is replayed from cached responses in CI when `CT_API_CACHE=1` |
| Threading (`OMP_NUM_THREADS`) | Pinned to 1 in `scripts/dgx/run_smoke_test.sh` |

---

## 7. CI gates

Each PR runs:
1. `scripts/dgx/run_smoke_test.sh` — must exit 0
2. Sealed test set verification — PR-AUC must equal locked value to 4 dp
3. ZNE extrapolation tests — `tests/test_zne_extrapolations.py`
4. Synthetic KG fixture test — `tests/test_synthetic_fixture.py`

Manual gate at each milestone (M2, M3, M4, M5, M6, M7): full pipeline
reproduction recorded in `docs/reproducibility/milestone_M{n}_run.md`.

---

## 8. Open items (resolved at M5)

- [ ] Pin exact Hetionet snapshot URL and SHA256 in `hetionet_snapshot.md`
- [ ] Cache ClinicalTrials.gov + PubMed responses for hermetic reruns
- [ ] Record GPU vs CPU PR-AUC delta (currently expected: 0)
- [ ] Add `pipeline_run_hash.txt` emission to `run_full_repurposing_pipeline.py`
- [ ] Container image (Docker) tagged with the preprint DOI

---

## 9. References

- `docs/paper_alignment/PAPER_IMPLEMENTATION_MAP.md` — paper section ↔ code map
- `docs/paper_alignment/METHOD_REPRODUCTION_CHECKLIST.md` — per-method status
- `docs/paper_alignment/ASSUMPTIONS_AND_DEVIATIONS.md` — deviations from paper
- `docs/paper_alignment/FIGURE_REPRODUCTION_PLAN.md` — per-figure scripts
- `preregistration/osf_preregistration_v1.md` — pre-registered hypotheses
