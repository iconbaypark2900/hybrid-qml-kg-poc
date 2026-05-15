import { buildScientificQualityEvidence } from "./v2-quality-data";
import type { LatestRunResponse } from "./api";

const latestRun = {
  status: "ok",
  results_dir: "results",
  latest_csv: {
    path: "results/latest_run.csv",
    mtime_epoch: 1,
    row: {
      qml_random_state: "42",
      obs_kernel_offdiag_std: "0.12",
      obs_kernel_gap: "0.04",
      negative_sampling: "random",
    },
  },
  latest_json: {
    path: "results/optimized_results.json",
    mtime_epoch: 1,
    ranking: [],
    config: {
      calibration_method: "isotonic",
      random_state: 42,
      negative_sampling: "random",
    },
  },
  provenance: [],
  message: null,
} satisfies LatestRunResponse;

const quality = buildScientificQualityEvidence({
  latestRun,
  analysis: null,
});

if (quality.variance.status !== "artifact_only") {
  throw new Error("Kernel variance should be marked as artifact-only evidence.");
}

if (quality.confidenceInterval.status !== "not_recorded") {
  throw new Error("Confidence intervals must not be inferred from single-run artifacts.");
}

if (quality.calibrationCurve.status !== "not_recorded") {
  throw new Error("Calibration method config alone is not a calibration curve.");
}
