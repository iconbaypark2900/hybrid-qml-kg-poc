# ZNE implementation audit — preregistration §5.5/§5.7 vs current code

**Audit date:** 2026-05-04
**Branch:** `roc/preregistration-followups`
**Scope:** Compare the existing Zero-Noise Extrapolation implementation in `quantum_layer/` against the Pauli Path ZNE commitments locked by [`preregistration/osf_preregistration_v1.md`](../../preregistration/osf_preregistration_v1.md) §5.4, §5.5, §5.7, §8.6.
**Purpose:** Determine whether hardware experiments (preregistration §5.4) require new ZNE code or can run from configuration. Surface specific gaps with concrete remediation so the implementation can be brought into alignment before §5.4 launches.

This is a **code review** producing a gap list — no implementation changes are made by this document.

## A. Preregistration commitments

Quoted directly from the preregistration:

1. **Hardware experiment scope** (§5.4): "3 problem sizes (10, 15, 20 dimensions) × 3 backend snapshots × 100 evaluation samples per condition = 900 hardware-evaluated samples."
2. **Pauli Path ZNE per-kernel-entry workflow** (§5.7), 6 steps:
   1. Compute at noise scaling 1× (raw)
   2. Compute at noise scaling 3× (fold-based circuit folding)
   3. Compute at noise scaling 5× (fold-based circuit folding)
   4. Extrapolate to zero-noise via linear regression
   5. Extrapolate to zero-noise via Richardson extrapolation
   6. Report both extrapolations and raw unmitigated result
3. **Sensitivity analysis pre-committed** (§8.6): "ZNE extrapolation method (linear vs Richardson reported separately)."

## B. Current implementation

| File | Public surface | What it does |
|---|---|---|
| `quantum_layer/advanced_error_mitigation.py` (320 lines) | `PauliPathZNE` (lines 17–186), `AdaptiveErrorMitigation` (189–244), `DynamicalDecouplingEnhanced` (247–295) | `PauliPathZNE.fit_noise_model()` fits an analytical 3-parameter Pauli-path model `(C0, H_bar, sigma)` to per-scale measurements via Bayesian priors; returns the fitted `C0` zero-noise value. |
| `quantum_layer/quantum_error_mitigation.py` (669 lines) | `ZeroNoiseExtrapolation` (29–169), `ProbabilisticErrorCancellation` (172–285), `CliffordDataRegression` (288–430), `CompositeErrorMitigation` (433–562) | `ZeroNoiseExtrapolation` supports `extrapolation_method ∈ {"polynomial", "exponential", "linear"}` and uses **gate-repetition** noise amplification (line 70 `amplify_circuit_noise()`), not unitary folding. |
| `quantum_layer/qml_trainer.py` (lines 871–1185, ZNE invocation only) | `qsvc_with_precomputed_kernel()` invokes `PauliPathZNE` in the simulator-noisy path | Reads scales from YAML config (line 878: `scales = zne_cfg.get("scales", [1.0, 1.5, 2.0])`). Builds depolarizing noise model, amplifies via `p_s = base_prob * float(s)` (line 1104). Logs single `zne_kernel_posneg_mean_C0` extrapolation per stream (lines 1153–1170). |
| `scripts/self_test_mitigation.py` (small) | Validates ZNE+readout column presence in observables | Checks for `obs_zne_enabled`, `obs_zne_kernel_posneg_mean_C0`, etc. (lines 59–64). Does not assert on noise-factor values or linear/Richardson distinction. |

## C. Match matrix

| # | Commitment (§ ref) | Implementation | Status |
|---|---|---|---|
| 1 | Noise scaling at 1×, 3×, 5× (§5.7 steps 1–3) | YAML default `[1.0, 1.5, 2.0]` (`qml_trainer.py:878`); configurable but NOT preregistration-locked | ⚠️ partial |
| 2 | Fold-based circuit folding (§5.7 steps 2–3) | Gate-repetition via `amplify_circuit_noise()` (`quantum_error_mitigation.py:50–95`); NOT unitary folding | ⚠️ partial |
| 3 | Linear extrapolation to zero-noise (§5.7 step 4) | `ZeroNoiseExtrapolation` supports `extrapolation_method='linear'` (`quantum_error_mitigation.py:40`); but the active path uses `PauliPathZNE.fit_noise_model()` (analytical only). Linear polyfit not surfaced separately. | ❌ missing |
| 4 | Richardson extrapolation to zero-noise (§5.7 step 5) | Not implemented; only the analytical Pauli-path model produces a zero-noise value | ❌ missing |
| 5 | Report both extrapolations separately (§5.7 step 6 + §8.6) | Single `zne_kernel_posneg_mean_C0` column per stream (raw / readout-mitigated); no `_linear` or `_richardson` suffix variants | ❌ missing |
| 6 | Report raw unmitigated alongside (§5.7 step 6) | `zne_kernel_posneg_mean_C0_raw` (`qml_trainer.py:1162`) plus `kernel_posneg_mean_explicit_raw_lambda1` at λ=1.0 | ✅ matches |
| 7 | Hardware experiment scope: 9 conditions × 100 samples = 900 (§5.4) | No hardware execution path in the ZNE code; simulator-only invocation | ❓ unclear / forward-looking |
| 8 | Per-kernel-entry workflow with optional readout mitigation | Readout mitigation is integrated and applied alongside ZNE (`qml_trainer.py:1053–1089`) | ✅ matches |
| 9 | Pauli Path analytical model (§5.5 reference) | `PauliPathZNE.model_function()` (`advanced_error_mitigation.py:31–56`) implements error function + exponential damping | ✅ matches |
| 10 | Backend snapshot variability sensitivity (§8.6) | No backend-snapshot or calibration-cycle metadata captured in ZNE observables | ❓ unclear |

**Summary:** 3 ✅ matches, 2 ⚠️ partial, 3 ❌ missing, 2 ❓ unclear (deferred or hardware-phase work).

## D. Gaps and remediation

### Gap 1 — Noise factors do not default to `[1.0, 3.0, 5.0]`

- **Where:** `quantum_layer/qml_trainer.py:878` defaults to `[1.0, 1.5, 2.0]`; the preregistration locks `[1, 3, 5]` per §5.7.
- **Fix:** Change the default to `[1.0, 3.0, 5.0]` and add `ZNE_NOISE_SCALES = (1, 3, 5)` to [`utils/preregistered_constants.py`](../../utils/preregistered_constants.py) (already present — just wire `qml_trainer.py` to read from it). Update any committed YAML configs with the same values.
- **Effort:** 30 minutes (1-line default + import wiring + config update). 1 line, 1 small refactor — light-touch.

### Gap 2 — Gate-repetition is not fold-based circuit folding

- **Where:** `quantum_layer/quantum_error_mitigation.py:50–95` `amplify_circuit_noise()` repeats gates an "odd factor" number of times; preregistration §5.7 commits to **fold-based circuit folding** (canonical form: `U → U U† U` repeating to amplify noise without changing the ideal output).
- **Why it matters:** Mathematically distinct techniques. Gate-repetition can produce a different effective-noise scaling and may not match Pauli-path theory. For *simulator* validation the difference is tolerable; for *hardware* §5.4 runs reviewers will expect canonical folding.
- **Fix:** Either (a) use the `mitiq` library's `mitiq.zne.scaling.fold_global` or `fold_gates_at_random` to do unitary folding, or (b) implement a small in-tree `fold_circuit(qc, scale)` helper that conjugate-folds gates to match `scale ∈ {1, 3, 5}`. Library route is faster and battle-tested.
- **Effort:** ~1 day if using `mitiq` (already a possible dep — `mitiq>=0.32` was in the original tar's optional `quantum` extra, not in `requirements-full.txt` here). Add to `requirements-gpu.txt` or a new `requirements-mitiq.txt`.

### Gap 3 — No explicit linear polyfit extrapolation

- **Where:** `PauliPathZNE.fit_noise_model()` returns a single analytical `C0`. Preregistration §5.7 step 4 commits to a **linear regression** extrapolation alongside.
- **Fix:** In `advanced_error_mitigation.py`, after the analytical fit, also compute `C0_linear = np.polyfit(noise_scales, measurements, deg=1)` and return the intercept. Update the return signature to a dict `{analytical, linear, richardson, params}`.
- **Effort:** 1–2 hours.

### Gap 4 — No Richardson extrapolation

- **Where:** Same as Gap 3.
- **Fix:** Add Richardson extrapolation. Standard form for 3 noise scales `λ_1, λ_2, λ_3 = 1, 3, 5`: solve the system that eliminates first- and second-order error terms to extrapolate to λ=0. With evenly spaced scales this collapses to a closed-form weighted combination of the three measurements. Library option: `mitiq.zne.inference.RichardsonFactory`.
- **Effort:** 1–2 hours (closed-form), or merged with Gap 2 if adopting `mitiq`.

### Gap 5 — No separate logging of linear vs Richardson vs analytical

- **Where:** `qml_trainer.py:1153–1170` emits a single `zne_kernel_posneg_mean_C0` column.
- **Fix:** Once Gap 3 and Gap 4 are addressed, expand the observable schema to emit `..._C0_analytical`, `..._C0_linear`, `..._C0_richardson`. Update `scripts/self_test_mitigation.py` to assert all three columns are populated. The §8.6 sensitivity analysis writeup will then have its raw inputs.
- **Effort:** 2 hours (schema + downstream parsing + test assertions).

### Gap 6 — No hardware execution path

- **Where:** Forward-looking per preregistration §5.4 ("Hardware experiments forward-looking"). Simulator-only is not a current bug.
- **Fix:** Implement `QuantumExecutor.run_zne_job()` with IBM backend routing for §5.4's 900-sample hardware run. Aggregate 100 samples per `(problem_size, backend_snapshot)` condition. Log per-job IBM job IDs to satisfy §9.3 reproducibility commitment.
- **Effort:** 2–3 days (IBM job submission, retry/error handling, results aggregation, observable logging). Calendar-bound by IBM Torino queue time on top of effort.

### Gap 7 — No backend-snapshot metadata in observables

- **Where:** §8.6 commits to "Backend snapshot variability (different calibration cycles)" as a sensitivity. The current ZNE observables do not capture backend snapshot / calibration-cycle ID.
- **Fix:** Extend the observables emitted by `qml_trainer.py` to include `zne_backend_name` and `zne_backend_snapshot_id` (read from IBM job metadata at execution time). Group by snapshot in the §8.6 analysis notebook.
- **Effort:** 1–2 hours plus implementation alongside Gap 6.

## E. Open editorial questions

These need an author decision before the gap-filling work proceeds:

1. **Lock noise factors at `[1.0, 3.0, 5.0]`?** Currently configurable via YAML; preregistration says `[1, 3, 5]`. Locking would mean removing the YAML override (or marking it as a deviation requiring a §12 amendment).
2. **Reporting strategy for the three extrapolations.** Three separate observable columns (`..._analytical`, `..._linear`, `..._richardson`) — verbose but explicit — or a single JSON column packing all three? §8.6 implies separate reporting; verbose probably wins.
3. **Adopt `mitiq` as a dependency?** Cleaner and well-tested for unitary folding + Richardson, but adds a transitive dependency. Alternative: in-tree closed-form implementations (small but correct).
4. **Is fold-based circuit folding mandatory for simulator runs**, or only for hardware §5.4? Reviewers may not pull on simulator runs as long as the hardware run uses the canonical method.
5. **Readout-mitigation × ZNE interaction.** The current code reuses the λ=1.0 readout calibration matrix at all noise scales (`qml_trainer.py:1110`). Is that the intended workflow, or should readout calibration be re-derived per noise scale?

## F. Recommended next moves

In dependency order:

| Priority | Item | Effort | Blocks |
|---|---|---|---|
| P0 | Decide questions E.1, E.2, E.3, E.5 | minutes (author input) | All gap fixes |
| P1 | Gap 1 — change noise factor defaults to `[1, 3, 5]` | 30 min | Honest match for §5.7 steps 1–3 |
| P2 | Gap 3 + Gap 4 — add linear and Richardson alongside the analytical fit | half-day combined | §5.7 steps 4–5, §8.6 |
| P3 | Gap 5 — emit three separate observable columns; update self-test | 2 hours | §8.6 sensitivity write-up |
| P4 | Gap 2 — switch to canonical unitary folding (probably via `mitiq`) | 1 day | §5.7 steps 2–3 fidelity (mandatory for hardware) |
| P5 | Gap 6 + Gap 7 — hardware execution path on IBM Torino with snapshot metadata | 2–3 days + IBM queue time | §5.4, §5.6, §9.3, §8.6 |

P1 alone is a 30-minute commit that closes one of the partial-status rows. P2 + P3 together (~1 day) close the linear / Richardson reporting gaps and unblock the §8.6 sensitivity write-up. P4 is the one item with non-trivial scope that ought to land before any §5.4 hardware runs.

## G. References

- `quantum_layer/advanced_error_mitigation.py`:17–186 — `PauliPathZNE` analytical model and zero-noise fit
- `quantum_layer/quantum_error_mitigation.py`:29–169 — `ZeroNoiseExtrapolation` (gate-repetition variant, three extrapolation modes including linear)
- `quantum_layer/qml_trainer.py`:871–1185 — ZNE invocation, observable schema, readout-mitigation interaction
- `scripts/self_test_mitigation.py`:59–64 — current observable-presence assertions
- `preregistration/osf_preregistration_v1.md`:§5.4, §5.5, §5.7, §8.6
- `utils/preregistered_constants.py` — `ZNE_NOISE_SCALES = (1, 3, 5)` already defined; not yet wired into `qml_trainer.py`
