import {
  fetchAnalysisSummary,
  fetchJobs,
  fetchLatestRun,
  fetchStatus,
  fetchVizCircuitParams,
  fetchVizKGSubgraph,
  fetchVizModelMetrics,
  fetchVizMolecule,
  fetchVizRunPredictions,
  type AnalysisSummaryResponse,
  type CandidateEvidenceLink,
  type EvidenceProvenance,
  type JobResponse,
  type LatestRunResponse,
  type StatusResponse,
  type VizAtom,
  type VizBond,
  type VizCircuitResponse,
  type VizKGLink,
  type VizKGNode,
  type VizModelMetric,
  type VizRunPrediction,
} from "./api";
import { candidates, methodRows, type V2Candidate, type V2Session } from "./v2-data";

export type EvidenceSource = "live" | "fallback" | "empty" | "error";

export interface EvidenceState<T> {
  source: EvidenceSource;
  data: T;
  message: string | null;
  provenance: EvidenceProvenance[];
  evidenceLinks: CandidateEvidenceLink[];
}

export interface V2MethodRow {
  method: string;
  type: "Classical" | "Hybrid" | "Quantum";
  prAuc: string;
  meaning: string;
}

export interface V2RunJob {
  id: string;
  setup: string;
  status: string;
  duration: string;
  next: string;
}

export interface V2ExperimentEvidence {
  status: EvidenceState<StatusResponse | null>;
  latestRun: EvidenceState<LatestRunResponse | null>;
  analysis: EvidenceState<AnalysisSummaryResponse | null>;
  methods: EvidenceState<V2MethodRow[]>;
  predictions: EvidenceState<V2Candidate[]>;
  jobs: EvidenceState<V2RunJob[]>;
}

export interface V2MoleculeEvidence {
  source: EvidenceSource;
  compoundName: string;
  atoms: VizAtom[];
  bonds: VizBond[];
  message: string | null;
  provenance: EvidenceProvenance[];
}

export interface V2GraphEvidence {
  source: EvidenceSource;
  centerEntity: string | null;
  nodes: VizKGNode[];
  links: VizKGLink[];
  message: string | null;
  provenance: EvidenceProvenance[];
}

export interface V2CircuitEvidence {
  source: EvidenceSource;
  params: VizCircuitResponse | null;
  message: string | null;
  provenance: EvidenceProvenance[];
}

export interface V2VisualEvidence {
  molecule: V2MoleculeEvidence;
  graph: V2GraphEvidence;
  circuit: V2CircuitEvidence;
}

export interface V2EvidenceSnapshot {
  version: number;
  generatedAt: string;
  session: {
    sessionId?: string;
    selectedEntity: V2Session["selectedEntity"];
    selectedCandidate: V2Session["selectedCandidate"];
    runMode: string;
    scoreThreshold: string;
    mechanismWeight: string;
  };
  experiment: {
    statusSource: EvidenceSource;
    methodsSource: EvidenceSource;
    predictionsSource: EvidenceSource;
    jobSource: EvidenceSource;
    methodCount: number;
    predictionCount: number;
  };
  visual?: {
    moleculeSource: EvidenceSource;
    graphSource: EvidenceSource;
    circuitSource: EvidenceSource;
    moleculeAtomCount: number;
    graphNodeCount: number;
  };
  provenance: EvidenceProvenance[];
  candidateEvidenceLinks: CandidateEvidenceLink[];
}

export async function loadV2ExperimentEvidence(
  session: V2Session,
): Promise<V2ExperimentEvidence> {
  const [status, latestRun, analysis, modelMetrics, predictions, jobs] =
    await Promise.all([
      safeCall(fetchStatus),
      safeCall(fetchLatestRun),
      safeCall(fetchAnalysisSummary),
      safeCall(fetchVizModelMetrics),
      safeCall(() => fetchVizRunPredictions(6)),
      safeCall(fetchJobs),
    ]);

  return {
    status: toState(status, null),
    latestRun: toState(latestRun, null),
    analysis: toState(analysis, null),
    methods: buildMethodState(modelMetrics),
    predictions: buildPredictionState(predictions, session),
    jobs: buildJobsState(jobs, session),
  };
}

export async function loadV2VisualEvidence(
  session: V2Session,
): Promise<V2VisualEvidence> {
  const [molecule, graph, circuit] = await Promise.all([
    safeCall(() => fetchVizMolecule(session.selectedCandidate.candidate)),
    safeCall(() =>
      fetchVizKGSubgraph(session.selectedCandidate.disease, 24, 1),
    ),
    safeCall(fetchVizCircuitParams),
  ]);

  return {
    molecule: buildMoleculeEvidence(molecule, session),
    graph: buildGraphEvidence(graph, session),
    circuit: buildCircuitEvidence(circuit),
  };
}

export function buildV2EvidenceSnapshot(
  session: V2Session,
  experiment: V2ExperimentEvidence,
  visual?: V2VisualEvidence,
): V2EvidenceSnapshot {
  const provenance = [
    ...experiment.status.provenance,
    ...experiment.latestRun.provenance,
    ...experiment.analysis.provenance,
    ...experiment.methods.provenance,
    ...experiment.predictions.provenance,
    ...(visual?.molecule.provenance ?? []),
    ...(visual?.graph.provenance ?? []),
    ...(visual?.circuit.provenance ?? []),
  ];

  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    session: {
      sessionId: session.sessionId,
      selectedEntity: session.selectedEntity,
      selectedCandidate: session.selectedCandidate,
      runMode: session.runMode,
      scoreThreshold: session.scoreThreshold,
      mechanismWeight: session.mechanismWeight,
    },
    experiment: {
      statusSource: experiment.status.source,
      methodsSource: experiment.methods.source,
      predictionsSource: experiment.predictions.source,
      jobSource: experiment.jobs.source,
      methodCount: experiment.methods.data.length,
      predictionCount: experiment.predictions.data.length,
    },
    visual: visual
      ? {
          moleculeSource: visual.molecule.source,
          graphSource: visual.graph.source,
          circuitSource: visual.circuit.source,
          moleculeAtomCount: visual.molecule.atoms.length,
          graphNodeCount: visual.graph.nodes.length,
        }
      : undefined,
    provenance,
    candidateEvidenceLinks: experiment.predictions.evidenceLinks,
  };
}

function buildMethodState(
  result: SettledResult<Awaited<ReturnType<typeof fetchVizModelMetrics>>>,
): EvidenceState<V2MethodRow[]> {
  if (result.ok && result.value.status === "ok" && result.value.models.length > 0) {
    return {
      source: "live",
      data: result.value.models.map(mapMetricToMethodRow),
      message: result.value.run_timestamp
        ? `Loaded model metrics from run ${result.value.run_timestamp}.`
        : "Loaded model metrics from latest run artifacts.",
      provenance: result.value.provenance ?? [],
      evidenceLinks: [],
    };
  }

  return {
    source: result.ok ? "fallback" : "error",
    data: methodRows,
    message: result.ok
      ? result.value.message ?? "No live model metrics found; using QCE26 paper-aligned fallback."
      : result.error.message,
    provenance: [
      fallbackProvenance("/viz/model-metrics", "QCE26 paper-aligned model metrics fallback."),
    ],
    evidenceLinks: [],
  };
}

function buildPredictionState(
  result: SettledResult<Awaited<ReturnType<typeof fetchVizRunPredictions>>>,
  session: V2Session,
): EvidenceState<V2Candidate[]> {
  if (result.ok && result.value.status === "ok" && result.value.predictions.length > 0) {
    return {
      source: "live",
      data: mapPredictionsToCandidates(result.value.predictions, session),
      message: result.value.source_file
        ? `Loaded candidates from ${result.value.source_file}.`
        : "Loaded candidates from latest prediction artifacts.",
      provenance: result.value.provenance ?? [],
      evidenceLinks: buildCandidateEvidenceLinks(session, result.value.provenance ?? []),
    };
  }

  return {
    source: result.ok ? "fallback" : "error",
    data: [session.selectedCandidate, ...candidates.filter((candidate) => candidate !== session.selectedCandidate)],
    message: result.ok
      ? result.value.message ?? "No live predictions found; using QCE26 paper-aligned candidates."
      : result.error.message,
    provenance: [
      fallbackProvenance("/viz/run-predictions", "QCE26 paper-aligned candidate fallback."),
    ],
    evidenceLinks: buildCandidateEvidenceLinks(session),
  };
}

function buildJobsState(
  result: SettledResult<JobResponse[]>,
  session: V2Session,
): EvidenceState<V2RunJob[]> {
  if (result.ok && result.value.length > 0) {
    return {
      source: "live",
      data: result.value.slice(0, 3).map(mapJob),
      message: "Loaded recent pipeline jobs.",
      provenance: [],
      evidenceLinks: [],
    };
  }

  return {
    source: result.ok ? "fallback" : "error",
    data: [
      {
        id: "job_qce26_001",
        setup: `${session.runMode}, score ${session.scoreThreshold}`,
        status: "ready",
        duration: "18m 12s",
        next: "Validate",
      },
      {
        id: "job_qce26_002",
        setup: "Mechanism rerank",
        status: "queued",
        duration: "0m 00s",
        next: "Monitor",
      },
      {
        id: "job_qce26_003",
        setup: "Quantum branch",
        status: "unavailable",
        duration: "0m 00s",
        next: "Inspect",
      },
    ],
    message: result.ok
      ? "No pipeline jobs found; showing representative QCE26 workflow states."
      : result.error.message,
    provenance: [
      fallbackProvenance("/jobs", "Representative job states for v2 provenance workflow."),
    ],
    evidenceLinks: [],
  };
}

function buildMoleculeEvidence(
  result: SettledResult<Awaited<ReturnType<typeof fetchVizMolecule>>>,
  session: V2Session,
): V2MoleculeEvidence {
  if (result.ok && result.value.status === "ok" && result.value.atoms.length > 0) {
    return {
      source: "live",
      compoundName: result.value.compound_name ?? session.selectedCandidate.candidate,
      atoms: result.value.atoms,
      bonds: result.value.bonds,
      message: "Loaded molecule coordinates from the visualization API.",
      provenance: result.value.provenance ?? [],
    };
  }

  return {
    source: result.ok ? "fallback" : "error",
    compoundName: session.selectedCandidate.candidate,
    atoms: [],
    bonds: [],
    message: result.ok
      ? result.value.message ?? "No molecule coordinates available; showing structured fallback."
      : result.error.message,
    provenance: result.ok
      ? result.value.provenance ?? [fallbackProvenance("/viz/molecule", "Molecule fallback panel.")]
      : [fallbackProvenance("/viz/molecule", result.error.message)],
  };
}

function buildGraphEvidence(
  result: SettledResult<Awaited<ReturnType<typeof fetchVizKGSubgraph>>>,
  session: V2Session,
): V2GraphEvidence {
  if (result.ok && result.value.status === "ok" && result.value.nodes.length > 0) {
    return {
      source: "live",
      centerEntity: result.value.center_entity ?? session.selectedCandidate.disease,
      nodes: result.value.nodes,
      links: result.value.links,
      message: "Loaded KG subgraph evidence.",
      provenance: result.value.provenance ?? [],
    };
  }

  return {
    source: result.ok ? "fallback" : "error",
    centerEntity: session.selectedCandidate.disease,
    nodes: [],
    links: [],
    message: result.ok
      ? result.value.message ?? "No KG subgraph available; showing relation evidence fallback."
      : result.error.message,
    provenance: result.ok
      ? result.value.provenance ?? [fallbackProvenance("/viz/kg-subgraph", "KG fallback relation evidence.")]
      : [fallbackProvenance("/viz/kg-subgraph", result.error.message)],
  };
}

function buildCircuitEvidence(
  result: SettledResult<Awaited<ReturnType<typeof fetchVizCircuitParams>>>,
): V2CircuitEvidence {
  if (result.ok && result.value.status === "ok") {
    return {
      source: "live",
      params: result.value,
      message: "Loaded circuit parameters from API.",
      provenance: result.value.provenance ?? [],
    };
  }

  return {
    source: result.ok ? "fallback" : "error",
    params: null,
    message: result.ok
      ? "No circuit params available; using QCE26 16Q Pauli framing."
      : result.error.message,
    provenance: result.ok
      ? result.value.provenance ?? [fallbackProvenance("/viz/circuit-params", "QCE26 circuit fallback.")]
      : [fallbackProvenance("/viz/circuit-params", result.error.message)],
  };
}

function mapMetricToMethodRow(metric: VizModelMetric): V2MethodRow {
  return {
    method: metric.name,
    type: inferMethodType(metric.name),
    prAuc: metric.pr_auc.toFixed(4),
    meaning:
      metric.type === "quantum"
        ? "Quantum branch metric from latest run artifacts"
        : metric.type === "ensemble"
          ? "Stacked or ensemble metric from latest run artifacts"
          : "Classical baseline metric from latest run artifacts",
  };
}

function mapPredictionsToCandidates(
  predictions: VizRunPrediction[],
  session: V2Session,
): V2Candidate[] {
  const mapped = predictions.slice(0, 6).map((prediction) => ({
    ...session.selectedCandidate,
    candidate: prediction.compound_name,
    disease: prediction.disease_name,
    score: prediction.score.toFixed(3),
    reason: `${prediction.confidence} confidence artifact-backed prediction`,
    next: "Validate",
  }));

  return mapped.length > 0 ? mapped : session.candidates;
}

function mapJob(job: JobResponse): V2RunJob {
  return {
    id: job.id,
    setup: String(job.flags?.relation ?? job.experiment_metadata?.note ?? "Pipeline run"),
    status: job.status,
    duration: formatDuration(job.started_at, job.finished_at),
    next: job.status === "completed" ? "Review" : "Monitor",
  };
}

function inferMethodType(name: string): V2MethodRow["type"] {
  const normalized = name.toLowerCase();
  if (normalized.includes("qsvc") || normalized.includes("vqc") || normalized.includes("quantum")) {
    return "Quantum";
  }
  if (normalized.includes("ensemble") || normalized.includes("stack")) {
    return "Hybrid";
  }
  return "Classical";
}

function formatDuration(startedAt: number | null, finishedAt: number | null): string {
  if (!startedAt) return "not started";
  const end = finishedAt ?? Date.now() / 1000;
  const totalSeconds = Math.max(0, Math.round(end - startedAt));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function toState<T>(
  result: SettledResult<T>,
  fallback: T,
): EvidenceState<T> {
  if (result.ok) {
    return {
      source: result.value == null ? "empty" : "live",
      data: result.value ?? fallback,
      message: null,
      provenance: extractProvenance(result.value),
      evidenceLinks: [],
    };
  }

  return {
    source: "error",
    data: fallback,
    message: result.error.message,
    provenance: [fallbackProvenance("unknown", result.error.message)],
    evidenceLinks: [],
  };
}

function extractProvenance(value: unknown): EvidenceProvenance[] {
  if (
    value &&
    typeof value === "object" &&
    "provenance" in value &&
    Array.isArray((value as { provenance?: unknown }).provenance)
  ) {
    return (value as { provenance: EvidenceProvenance[] }).provenance;
  }
  return [];
}

function fallbackProvenance(endpoint: string, note: string): EvidenceProvenance {
  return {
    endpoint,
    source_kind: "fallback",
    notes: [note],
  };
}

function buildCandidateEvidenceLinks(
  session: V2Session,
  provenance: EvidenceProvenance[] = [
    fallbackProvenance("/viz/run-predictions", "QCE26 paper-aligned candidate evidence."),
  ],
): CandidateEvidenceLink[] {
  const candidate = session.selectedCandidate;
  const baseProvenance = provenance[0] ?? fallbackProvenance("/viz/run-predictions", "Candidate evidence fallback.");
  const clinicalSource = candidate.decisionDetail.includes("No direct")
    ? "ClinicalTrials.gov search returned no direct trial support"
    : "ClinicalTrials.gov validation summary";

  return [
    {
      kind: "model_score",
      label: `${candidate.candidate} to ${candidate.disease} score ${candidate.score}`,
      source: "Hybrid QML-KG run artifacts",
      relation: "CtD",
      score: Number(candidate.score),
      provenance: baseProvenance,
    },
    {
      kind: "clinical_trial",
      label: candidate.decisionDetail,
      source: clinicalSource,
      relation: "clinical-validation",
      provenance: {
        endpoint: "/v2/validation",
        source_kind: "fallback",
        notes: [candidate.supportItems[0]?.value ?? "Paper-aligned clinical validation statement."],
      },
    },
    {
      kind: "mechanism",
      label: candidate.supportItems[1]?.label ?? "Mechanism evidence",
      source: candidate.supportItems[1]?.value ?? candidate.reason,
      relation: "mechanism-of-action",
      provenance: {
        endpoint: "/v2/validation",
        source_kind: "fallback",
        notes: [candidate.workingConclusion],
      },
    },
    {
      kind: "kg_edge",
      label: candidate.relationEvidence[0]?.text ?? "CtD relation evidence",
      source: "Hetionet relation evidence",
      relation: candidate.relationEvidence[0]?.code ?? "CtD",
      provenance: {
        endpoint: "/viz/kg-subgraph",
        source_kind: "dataset",
        relation: candidate.relationEvidence[0]?.code ?? "CtD",
        notes: ["Candidate-level KG edge support should be inspected in the Visual graph panel."],
      },
    },
  ];
}

type SettledResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: Error };

async function safeCall<T>(fn: () => Promise<T>): Promise<SettledResult<T>> {
  try {
    return { ok: true, value: await fn() };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error : new Error("Request failed"),
    };
  }
}
