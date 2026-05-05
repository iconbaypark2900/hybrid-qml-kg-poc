# Sensitivity analysis — Stacking vs weighted-average ensemble (supplementary)

**Status:** Partial — qualitative observations from the existing experiment log; the formal weighted-average run is deferred to the headline GPU run on the DGX.
**Preregistration anchor:** [`preregistration/osf_preregistration_v1.md`](../../preregistration/osf_preregistration_v1.md) §5.4 (stacking is the headline ensemble), §8.6 ("Ensemble configuration — stacking (primary) vs weighted-average (sensitivity)").

This supplementary section addresses the reviewer question: *does the choice of ensemble combiner matter, given that the project ships a stacking ensemble as headline?*

## Setup

The hybrid ensemble combines four base learners — QSVC, tuned Logistic Regression, tuned Random Forest, tuned Extra Trees — each trained on the same RotatE 128D pair features with PauliFeatureMap reps=2 for the quantum learner. Two combiner methods are documented in `quantum_layer/quantum_classical_ensemble.py` ([source](../../quantum_layer/quantum_classical_ensemble.py)):

- **Stacking (primary).** A meta-learner (logistic regression by default) is trained on the out-of-fold predictions of the base learners. The meta-learner discovers the optimal weights from the data — including the relative weight of QSVC versus the classical learners.
- **Weighted-average (sensitivity).** Base learner probabilities are linearly combined with manually-set weights. The current default is `{"quantum": 0.5, "classical": 0.5}` per `QuantumClassicalEnsemble.__init__()`. Manual weights can be overridden via `--ensemble_quantum_weight`.

## Existing observations

Source: [`README.md`](../../README.md) "Experiment Log" key findings (Pauli primary configuration unless noted).

| Variant | Ensemble PR-AUC | Note |
|---|---|---|
| Stacking (best-tuned classical, Pauli reps=2) | **0.7987** | Headline |
| Stacking + manually-set `ensemble_quantum_weight=0.4` | 0.7408 | "Stacking learns weights" — the manual weight has no additional effect over stacking's learned weights |
| Stacking with QSVC-C variant (`qsvc_C=0.05` vs 0.1) | 0.7408 | Same as `C=0.1` baseline |

The README's Key Findings note this directly:

> Stacking ensemble learns optimal classical/quantum weights automatically; manually setting `ensemble_quantum_weight` has no additional effect.

In other words: when the meta-learner is in charge, the manual quantum-weight knob is redundant. This is the expected behavior of a stacking architecture — the meta-learner subsumes any fixed-weight scheme that's expressible as a linear combination of base outputs. The manually-weighted regime is therefore a *strictly weaker* family unless the meta-learner is misspecified.

## What's missing

No apples-to-apples weighted-average run with `ensemble_method='weighted_average'` is documented in the experiment log. The numbers above show a stacking ensemble whose manual quantum-weight override was ignored — they don't show the *pure* weighted-average configuration (no meta-learner, only fixed weights).

To complete §8.6's "stacking primary, weighted-average sensitivity" comparison, the headline GPU run should produce:

- **Run A (already in the headline driver):** `ensemble_method='stacking'` → `Ensemble (stacking)` row in [`docs/results/bootstrap_ci_analysis.md`](bootstrap_ci_analysis.md).
- **Run B (sensitivity):** `ensemble_method='weighted_average'` with `weights={"quantum": 0.5, "classical": 0.5}` → `Ensemble (weighted_average)` row in the same doc.

`scripts/run_bootstrap_ci.py` currently implements stacking via `cross_val_predict` on per-base OOF predictions. Adding a weighted-average path is a small extension: skip the meta-learner training, average the four base OOF score vectors with fixed weights, and treat the result as another "model" through the bootstrap helper.

## Expected outcome (qualitative)

- Stacking ≥ weighted-average in expectation, since stacking can recover any weighted-average configuration plus more.
- The gap is informative for §8.6: a small gap means the dataset's structure makes the optimal weights nearly equal, which is itself useful methodology evidence; a large gap argues for stacking as a meaningful methodology choice rather than a default.
- If stacking fails to materially beat weighted-average, the manuscript's headline claim weakens: "the meta-learner is doing real work" becomes harder to defend.

## What to do once the GPU run lands

1. Add a `--ensemble_method` flag (or `--include_weighted_average`) to `scripts/run_bootstrap_ci.py` that runs the weighted-average path alongside stacking and emits a second OOF ensemble vector.
2. Add a row to [`docs/results/bootstrap_ci_analysis.md`](bootstrap_ci_analysis.md) under a "Sensitivity (§8.6): ensemble configuration" section: `Ensemble (weighted_average)` with its own OOF PR-AUC and a paired-bootstrap CI for the difference Stacking − Weighted-average.
3. Replace the qualitative claims in this doc with the actual numbers and bootstrap bounds.

## Files referenced

- [`preregistration/osf_preregistration_v1.md`](../../preregistration/osf_preregistration_v1.md) §5.4, §6.4, §8.6
- [`README.md`](../../README.md) "Experiment Log" — source of the qualitative observations
- [`quantum_layer/quantum_classical_ensemble.py`](../../quantum_layer/quantum_classical_ensemble.py) — both ensemble methods are in code
- [`scripts/run_bootstrap_ci.py`](../../scripts/run_bootstrap_ci.py) — currently stacking-only; weighted-average path is the work item for the next iteration
- [`utils/preregistered_constants.py`](../../utils/preregistered_constants.py) `ENSEMBLE_METHOD = "stacking"` (locked primary)
