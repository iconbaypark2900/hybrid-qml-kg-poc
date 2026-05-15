import type { AnalysisSummaryResponse, LatestRunResponse } from "./api";
import type { EvidenceSource, V2CircuitEvidence, V2GraphEvidence, V2MoleculeEvidence } from "./v2-live-data";

export type QualityStatus = "recorded" | "not_recorded" | "artifact_only" | "fallback";
export type VisualOverlayStatus = "live" | "fallback" | "not_recorded";

export interface QualityControlItem {
  key: string;
  label: string;
  status: QualityStatus;
  value?: string;
  detail: string;
  source?: string;
  nextAction?: string;
}

export interface ScientificQualityEvidence {
  multiSeed: QualityControlItem;
  randomBaseline: QualityControlItem;
  degreeBaseline: QualityControlItem;
  confidenceInterval: QualityControlItem;
  variance: QualityControlItem;
  calibrationCurve: QualityControlItem;
  uncertainty: QualityControlItem;
  warnings: string[];
}

export interface VisualEvidenceOverlay {
  label: string;
  value: string;
  status: VisualOverlayStatus;
  explanation: string;
}

export function buildScientificQualityEvidence({
  latestRun,
  analysis,
}: {
  latestRun: LatestRunResponse | null;
  analysis: AnalysisSummaryResponse | null;
}): ScientificQualityEvidence {
  const csvRow = latestRun?.latest_csv?.row ?? {};
  const config = latestRun?.latest_json?.config ?? {};
  const warnings: string[] = [];
  const seed = stringValue(csvRow.qml_random_state) ?? stringValue(config.random_state);
  const kernelStd = stringValue(csvRow.obs_kernel_offdiag_std);
  const kernelGap = stringValue(csvRow.obs_kernel_gap);
  const calibrationMethod = stringValue(config.calibration_method);
  const negativeSampling =
    stringValue(csvRow.negative_sampling) ?? stringValue(config.negative_sampling);
  const ranking = analysis?.ranking ?? latestRun?.latest_json?.ranking ?? [];

  warnings.push(
    "Confidence intervals, calibration curves, and random/degree baselines are shown only when recorded in artifacts.",
  );
  if (!latestRun) {
    warnings.push("No latest run artifact is available; quality controls are fallback/not recorded.");
  }
  if (ranking.length === 0) {
    warnings.push("No model ranking artifact is available for quality-control context.");
  }

  return {
    multiSeed: {
      key: "multiSeed",
      label: "Multi-seed results",
      status: "not_recorded",
      value: seed ? `single seed ${seed}` : "not recorded",
      detail:
        "Current artifacts expose at most a single random seed for this v2 view; no multi-seed aggregate was recorded.",
      source: latestRun?.latest_json?.path,
      nextAction: "Run the multi-seed preset before claiming stability.",
    },
    randomBaseline: {
      key: "randomBaseline",
      label: "Random baseline",
      status: hasBaseline(ranking, "random") ? "recorded" : "not_recorded",
      value: hasBaseline(ranking, "random") ? "recorded" : "not recorded",
      detail: negativeSampling
        ? `Negative sampling recorded as ${negativeSampling}; this is task construction context, not a random baseline row.`
        : "No explicit random classifier baseline row was found in the current artifacts.",
      nextAction: "Add random baseline artifact row or run a baseline preset.",
    },
    degreeBaseline: {
      key: "degreeBaseline",
      label: "Degree heuristic baseline",
      status: hasBaseline(ranking, "degree") ? "recorded" : "not_recorded",
      value: hasBaseline(ranking, "degree") ? "recorded" : "not recorded",
      detail:
        "No explicit degree-heuristic baseline row was found in current artifacts.",
      nextAction: "Run or ingest degree baseline before comparing graph-hub effects.",
    },
    confidenceInterval: {
      key: "confidenceInterval",
      label: "Confidence intervals",
      status: hasAnyField(ranking, ["ci", "confidence_interval", "pr_auc_ci"])
        ? "recorded"
        : "not_recorded",
      value: hasAnyField(ranking, ["ci", "confidence_interval", "pr_auc_ci"])
        ? "recorded"
        : "not recorded",
      detail:
        "No confidence interval fields were found; single-run PR-AUC should not be presented as a stability estimate.",
      nextAction: "Generate multi-seed or bootstrap artifacts with CI fields.",
    },
    variance: {
      key: "variance",
      label: "Variance / kernel spread",
      status: kernelStd ? "artifact_only" : "not_recorded",
      value: kernelStd ? `kernel offdiag std ${kernelStd}` : "not recorded",
      detail: kernelStd
        ? `Kernel spread was recorded${kernelGap ? ` with gap ${kernelGap}` : ""}; this is kernel-statistical evidence, not a model confidence interval.`
        : "No variance or kernel spread field was found.",
      source: latestRun?.latest_csv?.path,
    },
    calibrationCurve: {
      key: "calibrationCurve",
      label: "Calibration curve",
      status: hasAnyField(ranking, ["calibration_curve", "calibration_bins"])
        ? "recorded"
        : "not_recorded",
      value: calibrationMethod ? `method config: ${calibrationMethod}` : "not recorded",
      detail: calibrationMethod
        ? "A calibration method is configured, but no calibration curve/bin artifact is present."
        : "No calibration method or curve artifact is present.",
      nextAction: "Persist calibration bins/curve data before showing reliability diagrams.",
    },
    uncertainty: {
      key: "uncertainty",
      label: "Uncertainty",
      status: kernelStd ? "artifact_only" : "not_recorded",
      value: kernelStd ? "kernel-derived only" : "not recorded",
      detail:
        "Uncertainty is limited to recorded kernel statistics in this artifact-first view; predictive uncertainty was not recorded.",
      nextAction: "Add uncertainty estimates or calibrated prediction intervals to artifacts.",
    },
    warnings,
  };
}

export function buildMoleculeOverlays(evidence: V2MoleculeEvidence): VisualEvidenceOverlay[] {
  return [
    {
      label: "Molecule source",
      value: evidence.source,
      status: sourceToOverlayStatus(evidence.source),
      explanation:
        evidence.source === "live"
          ? "3D coordinates were loaded from the molecule API."
          : "The molecule panel is using a fallback visual; do not interpret geometry.",
    },
    {
      label: "Atoms / bonds",
      value: `${evidence.atoms.length} atoms / ${evidence.bonds.length} bonds`,
      status: evidence.atoms.length > 0 ? "live" : "not_recorded",
      explanation: "Counts come from the rendered molecular coordinate payload.",
    },
  ];
}

export function buildGraphOverlays(evidence: V2GraphEvidence): VisualEvidenceOverlay[] {
  return [
    {
      label: "KG subgraph",
      value: `${evidence.nodes.length} nodes / ${evidence.links.length} edges`,
      status: evidence.nodes.length > 0 ? "live" : sourceToOverlayStatus(evidence.source),
      explanation: evidence.centerEntity
        ? `Centered on ${evidence.centerEntity}.`
        : "No center entity was recorded.",
    },
    {
      label: "Candidate edge check",
      value: evidence.links.length > 0 ? "inspect graph" : "not recorded",
      status: evidence.links.length > 0 ? "live" : "not_recorded",
      explanation:
        "Graph presence does not prove a direct candidate-specific edge unless the relation row is visible.",
    },
  ];
}

export function buildCircuitOverlays(evidence: V2CircuitEvidence): VisualEvidenceOverlay[] {
  const params = evidence.params;
  return [
    {
      label: "Feature map",
      value: params ? `${params.feature_map}, ${params.n_qubits} qubits` : "not recorded",
      status: params ? sourceToOverlayStatus(evidence.source) : "not_recorded",
      explanation: "Circuit labels are runtime/config evidence, not a new hardware validation.",
    },
    {
      label: "Execution mode",
      value: params?.execution_mode ?? "not recorded",
      status: params?.execution_mode ? "live" : "not_recorded",
      explanation:
        "Execution mode describes configured runtime context and is separate from model quality.",
    },
  ];
}

function sourceToOverlayStatus(source: EvidenceSource): VisualOverlayStatus {
  return source === "live" ? "live" : source === "fallback" ? "fallback" : "not_recorded";
}

function stringValue(value: unknown): string | undefined {
  if (value === null || value === undefined || value === "") return undefined;
  return String(value);
}

function hasBaseline(ranking: Array<Record<string, unknown>>, keyword: string): boolean {
  return ranking.some((row) => {
    const name = String(row.name ?? row.method ?? "").toLowerCase();
    const type = String(row.type ?? "").toLowerCase();
    return (
      type === "baseline" &&
      (name.includes(keyword) || name.includes(`${keyword} classifier`))
    );
  });
}

function hasAnyField(
  ranking: Array<Record<string, unknown>>,
  keys: string[],
): boolean {
  return ranking.some((row) => keys.some((key) => row[key] !== undefined));
}
