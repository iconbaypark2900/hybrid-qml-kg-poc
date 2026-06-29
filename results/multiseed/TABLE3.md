# Table 3 — Tier 1 CtD results (DGX Spark, cached embeddings)

Generated from `results/multiseed/`, `results/moa_benchmark/`, and `results/cpd_run/`
after `./scripts/run_tier1_cached_sequential.sh`.

## Multi-seed CtD (5 seeds: 42, 7, 13, 99, 2026)

Config: RotatE cached full-graph (`rotate_256d_*` symlinked as `rotate_128d_*`),
Pauli QSVC, Nyström `m=200`, stacking ensemble, no MoA features.

| Model | Test PR-AUC (mean ± std) | Per-seed values |
|-------|--------------------------|-----------------|
| **HistGBDT** | **0.7051 ± 0.0368** | 0.7446, 0.7221, 0.7106, 0.7025, 0.6457 |
| Ensemble-QC-stacking | 0.5987 ± 0.0204 | 0.5781, 0.5832, 0.5924, 0.6144, 0.6254 |
| ExtraTrees-Optimized | 0.5047 ± 0.0208 | 0.4719, 0.5199, 0.5239, 0.4981, 0.5095 |
| QSVC-Optimized | 0.5006 ± 0.0107 | 0.5086, 0.4969, 0.4912, 0.5151, 0.4913 |
| RandomForest-Optimized | 0.4401 ± 0.0151 | 0.4207, 0.4462, 0.4285, 0.4479, 0.4574 |

Machine-readable summary: `results/multiseed/summary.json` (RF, ET, QSVC, Ensemble).

## Improved multi-seed CtD + MoA (`results/rerun_improved/`)

Config: same cached embeddings + MoA features, full classical suite (no `fast_mode`),
`--optimize_feature_map_reps`, `--skip_vqc` / `--skip_svm_rbf`, Nyström `m=200`.
Script: `./scripts/run_improved_rerun.sh`.

| Model | Test PR-AUC (mean ± std) | Per-seed values |
|-------|--------------------------|-----------------|
| **HistGBDT** | **0.7393 ± 0.0373** | 0.7784, 0.7452, 0.7614, 0.7310, 0.6807 |
| Ensemble-QC-stacking | 0.6298 ± 0.0177 | 0.6103, 0.6290, 0.6269, 0.6242, 0.6587 |
| ExtraTrees-Optimized | 0.5366 ± 0.0151 | 0.5506, 0.5413, 0.5440, 0.5112, 0.5361 |
| QSVC-Optimized | 0.5006 ± 0.0107 | 0.5086, 0.4969, 0.4912, 0.5151, 0.4913 |
| RandomForest-Optimized | 0.4701 ± 0.0058 | 0.4692, 0.4711, 0.4717, 0.4612, 0.4772 |

Machine-readable summary: `results/rerun_improved/summary.json`.

**Δ vs Tier 1 multiseed (no MoA):** HistGBDT **+0.034** mean PR-AUC (0.705 → 0.739).

> **Note:** Ensemble rows above used the pre-fix stacking path (LR picked first). See
> **256D + MoA with ensemble fix** below for corrected stacking.

## 256D + MoA + full classical (`results/rerun_256d_moa/`)

Config: cached 256D RotatE full-graph, MoA features, `--embedding_dim 256`,
full classical (no `fast_mode`), `--tune_classical`, stacking with **best PR-AUC
classical** for meta-learner (HistGBDT on all seeds), `--skip_vqc` / `--skip_svm_rbf`,
Nyström `m=200`. Script: `./scripts/run_256d_moa_multiseed.sh`.

| Model | Test PR-AUC (mean ± std) | Per-seed values |
|-------|--------------------------|-----------------|
| **Ensemble-QC-stacking** | **0.7398 ± 0.0378** | 0.7805, 0.7452, 0.7614, 0.7313, 0.6807 |
| HistGBDT | 0.7393 ± 0.0373 | 0.7784, 0.7452, 0.7614, 0.7310, 0.6807 |
| ExtraTrees-Optimized | 0.5367 ± 0.0152 | 0.5506, 0.5416, 0.5440, 0.5112, 0.5361 |
| QSVC-Optimized | 0.5006 ± 0.0107 | 0.5086, 0.4969, 0.4912, 0.5151, 0.4913 |
| RandomForest-Optimized | 0.4701 ± 0.0058 | 0.4692, 0.4711, 0.4717, 0.4612, 0.4772 |

Machine-readable summary: `results/rerun_256d_moa/summary.json`.

**Best single run (seed 42):** Ensemble-QC-stacking **0.7805** (HistGBDT 0.7784; +0.0021
from stacking). Beats prior MoA benchmark HistGBDT 0.7784.

**Δ vs broken ensemble (improved rerun block):** stacking mean **+0.110** (0.630 → 0.740)
after fixing classical model selection.

## Single-run extensions (seed 42 config, same cached pipeline)

| Run | Relation | Best model | Test PR-AUC |
|-----|----------|------------|-------------|
| MoA benchmark | CtD + MoA | HistGBDT | **0.7784** |
| CpD relation | CpD | HistGBDT | **0.7627** |
| Multi-seed seed 42 | CtD (no MoA) | HistGBDT | 0.7446 |

## Notes for Methods / paper text

- **Cached embeddings:** Tier 1 queue used `--use_cached_embeddings` with 256D RotatE
  artifacts symlinked to 128D paths (complex real+imag concatenation).
- **Nyström:** Multiseed runs used `--qsvc_nystrom_m 200` (document if full kernel
  differs from headline 0.7987 ensemble run).
- **VQC:** Failed on all seeds (`Pauli` feature map; VQC accepts only `ZZ`/`Z`).
  QSVC rows above are valid; exclude VQC from Table 3 or re-run with `--skip_vqc`.
- **Quantum vs classical:** QSVC ≈ 0.50 (chance on balanced CtD). With ensemble fix,
  stacking matches or slightly beats HistGBDT (seed 42: +0.0021); quantum adds no lift
  on other seeds when HistGBDT dominates.
- **Qiskit Aer:** `gpu_simulator` fell back to CPU on DGX Spark ARM64; PyTorch CUDA
  used for embedding load / classical training.

## Baselines (same CtD split: seed 42, hard negatives, 80/20)

Computed with `scripts/degree_heuristic_baseline.py` (`negative_sampling=hard`).

| Baseline | Test PR-AUC | Notes |
|----------|-------------|-------|
| Random (expected) | **0.5000** | Balanced test set (151 pos / 151 neg) |
| Degree-heuristic | **0.4749** | `compound_degree × disease_degree` on training positives only |

Artifact: `results/degree_heuristic_baseline.json`.

**Paper stance (2026-06):** Report **0.7805** (best single-run ensemble, 256D+MoA seed 42) and
**0.7398 ± 0.038** (5-seed ensemble mean). Historical headline **0.7987** / RF **0.7838**
are not reproduced on the current protocol (see `results/headline_repro_fresh/INVESTIGATION.md`:
`fast_mode`, HistGBDT vs RF, feature/MoA growth). Do not chase 0.7987 without pinned artifacts.

## Still needed for arXiv v2 (roadmap Tier 1)

1. ~~Random baseline row~~ — done (above).
2. ~~Degree-heuristic baseline~~ — done (above).
3. LaTeX citation key reconciliation.
4. Commit rendered figures `figures/fig{1,2,3}_*.pdf` (PDFs exist under `figures/`; verify vs current numbers before commit).
