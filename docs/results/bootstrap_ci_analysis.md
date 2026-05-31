# Bootstrap CI Analysis — H1 and H1b decision rules

**Run date:** 2026-05-04
**Git commit:** `3a9b9a0f68d235dcf57d192b78f1c765a399befe`
**Hetionet snapshot:** [`docs/reproducibility/hetionet_snapshot.md`](../docs/reproducibility/hetionet_snapshot.md) (edges sha256 `4b47bad290881ed468f2be9d11bf9adb7aaa2c5a1296de7ba09cdca058565a4e`)
**Configuration:** PauliFeatureMap reps=2, RotatE 128D pair features (concat + diff + Hadamard + scalars), hard negatives 1:1, 5-fold stratified CV
**Quantum backend:** config/quantum_config_gpu.yaml
**Bootstrap:** 500 resamples, seed `20260504`, 95% confidence

**Reproducibility (per preregistration §9.4):**
- Environment hash (SHA-256 of `pip freeze`): `eda19cfe0cda29908fa8346d51334afe14dfa68855dbe04d00aab9bdbfe15b63`
- Environment artifact: `results/cv_predictions/env.txt`
- OS / Python: Linux-6.11.0-1016-nvidia-aarch64-with-glibc2.39 | python 3.12.3
- Run metadata captured: 2026-05-05T01:17:08Z

## OOF point estimates (PR-AUC)

| Model | OOF PR-AUC |
|---|---|
| LR | 0.6574 |
| RF | 0.6952 |
| ET | 0.6985 |
| Ensemble | 0.6373 |

## H1 — QSVC alone vs each baseline

_QSVC was skipped in this run; H1 not computed. Re-run without `--skip_qsvc`._

## H1b — Stacking ensemble vs each baseline

| Baseline | Point (Ens − base) | 95% CI | Supports H1b (lo > 0)? |
|---|---|---|---|
| LR | -0.0201 | [-0.0914, +0.0445] | ✗ |
| RF | -0.0579 | [-0.1443, +0.0432] | ✗ |
| ET | -0.0612 | [-0.1250, +0.0119] | ✗ |

**Conjunction:** 0 of 3 support H1b.
**H1b supported:** **NO**.

## Notes

- Forward-looking baselines R-GCN and TransE are not yet implemented; H1/H1b are reported against the currently-implemented classical baselines (LR, RF, ET) and will be re-run after R-GCN and TransE are added (per preregistration §6.2).
- Per-fold per-model predictions cached at `results/cv_predictions/fold_{0..4}.npz` for reproducibility.
- Bootstrap is on OOF predictions (each instance contributes once); per-fold paired bootstrap is left as future work per preregistration §8.1's literal wording.
