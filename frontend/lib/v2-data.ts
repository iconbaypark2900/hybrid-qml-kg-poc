export const entitySuggestions = [
  {
    name: "Atherosclerosis",
    type: "Disease",
    route: "Disease route",
    description:
      "Rank compounds and mechanisms connected to the vascular disease neighborhood.",
  },
  {
    name: "Liver cancer",
    type: "Disease",
    route: "Disease route",
    description:
      "Inspect compound hypotheses with clinical and graph evidence around liver cancer.",
  },
  {
    name: "Gout",
    type: "Disease",
    route: "Disease route",
    description:
      "Surface biologically plausible candidates even when direct trial support is absent.",
  },
  {
    name: "Losartan",
    type: "Compound",
    route: "Compound route",
    description:
      "Look outward from the compound to disease hypotheses and vascular pathway evidence.",
  },
  {
    name: "Mitomycin",
    type: "Compound",
    route: "Compound route",
    description:
      "Trace the compound toward cancer hypotheses and supporting treatment context.",
  },
  {
    name: "Ezetimibe",
    type: "Compound",
    route: "Compound route",
    description:
      "Inspect cholesterol and urate-transport biology around a novel gout hypothesis.",
  },
  {
    name: "NPC1L1",
    type: "Gene / protein",
    route: "Target route",
    description:
      "Start from target biology and inspect nearby compounds, diseases, and mechanisms.",
  },
];

export type V2EntitySuggestion = (typeof entitySuggestions)[number];
export type V2EntityType = V2EntitySuggestion["type"];
export type V2RunMode = "Classical" | "Hybrid" | "Quantum hardware";
export type V2ModelType = "Classical" | "Hybrid" | "Quantum";
export type V2Tone = "success" | "warning" | "danger" | "quantum";

export interface V2Candidate {
  candidate: string;
  disease: string;
  score: string;
  reason: string;
  next: string;
  evidencePosture: string;
  decision: string;
  decisionDetail: string;
  supportItems: Array<{ label: string; value: string; tone: V2Tone }>;
  weakItems: string[];
  workingConclusion: string;
  pathway: string[];
  relationEvidence: Array<{ code: string; text: string; strength: "strong" | "medium" | "check" }>;
}

export const methodRows = [
  {
    method: "Ensemble-QC (Pauli)",
    type: "Hybrid" as V2ModelType,
    prAuc: "0.7987",
    meaning: "Paper headline: stacked classical learners plus Pauli QSVC",
  },
  {
    method: "RandomForest-Opt",
    type: "Classical" as V2ModelType,
    prAuc: "0.7838",
    meaning: "Best classical baseline in the primary QCE26 result",
  },
  {
    method: "ExtraTrees-Opt",
    type: "Classical" as V2ModelType,
    prAuc: "0.7807",
    meaning: "Strong tree baseline with different split behavior",
  },
  {
    method: "QSVC Pauli",
    type: "Quantum" as V2ModelType,
    prAuc: "0.6343",
    meaning: "Individually weaker but useful for ensemble decorrelation",
  },
  {
    method: "Ensemble-QC (ZZ)",
    type: "Hybrid" as V2ModelType,
    prAuc: "0.7408",
    meaning: "Feature-map comparator that misses the Pauli ensemble gain",
  },
];

const paperCandidates = [
  {
    candidate: "Losartan",
    disease: "Atherosclerosis",
    score: "0.528",
    reason: "Clinically supported vascular hypothesis with 7+ Phase-4 trials",
    next: "Validate",
    evidencePosture: "Validated",
    decision: "Keep",
    decisionDetail: "Supported by trials",
    supportItems: [
      {
        label: "Clinical support",
        value: "Seven or more ClinicalTrials.gov registrations support the Losartan to atherosclerosis hypothesis.",
        tone: "success" as V2Tone,
      },
      {
        label: "Mechanism context",
        value: "Vascular and cardiometabolic neighborhoods make the lower model score worth retaining.",
        tone: "success" as V2Tone,
      },
      {
        label: "Score-validity check",
        value: "This example demonstrates why score alone is not the final decision rule.",
        tone: "warning" as V2Tone,
      },
    ],
    weakItems: [
      "The score is lower than the top graph-artifact prediction.",
      "Trial support should still be checked for indication, phase, and endpoint match.",
      "Mechanism evidence should travel with the candidate before follow-up.",
    ],
    workingConclusion:
      "Keep Losartan to atherosclerosis as a validated lead because clinical support survives the score-validity inversion check.",
    pathway: ["Compound", "Vascular target context", "Cardiometabolic pathway", "Atherosclerosis"],
    relationEvidence: [
      ["CtD", "Candidate appears in the compound-treats-disease evaluation set", "strong"],
      ["CbG", "Compound-target context supports vascular mechanism review", "medium"],
      ["GpPW", "Target and pathway context connect to vascular biology", "strong"],
      ["DaG", "Disease-gene neighborhood should be checked before follow-up", "check"],
    ].map(([code, text, strength]) => ({
      code,
      text,
      strength: strength as "strong" | "medium" | "check",
    })),
  },
  {
    candidate: "Mitomycin",
    disease: "Liver cancer",
    score: "0.525",
    reason: "TACE treatment context with seven clinical trial registrations",
    next: "Compare",
    evidencePosture: "Validated",
    decision: "Keep",
    decisionDetail: "Supported by trials",
    supportItems: [
      {
        label: "Clinical support",
        value: "Seven trial registrations around TACE support the Mitomycin to liver cancer case.",
        tone: "success" as V2Tone,
      },
      {
        label: "Treatment context",
        value: "Cancer treatment context makes this a plausible retained candidate despite moderate model score.",
        tone: "success" as V2Tone,
      },
      {
        label: "Scope check",
        value: "The dashboard should avoid implying broad oncology generalization from this single relation.",
        tone: "warning" as V2Tone,
      },
    ],
    weakItems: [
      "Clinical context may be procedure-specific rather than a general treatment signal.",
      "The model score is moderate and should not be presented as proof.",
      "Mechanism evidence should distinguish liver cancer context from general cancer proximity.",
    ],
    workingConclusion:
      "Keep Mitomycin to liver cancer as clinically supported, but annotate that the evidence is tied to TACE context.",
    pathway: ["Compound", "Cancer treatment context", "Procedure evidence", "Liver cancer"],
    relationEvidence: [
      ["CtD", "Compound-disease relation is the active prediction target", "strong"],
      ["CbG", "Target context requires deeper review", "check"],
      ["GpPW", "Cancer pathway evidence should be inspected", "medium"],
      ["DaG", "Disease association evidence anchors the liver cancer neighborhood", "strong"],
    ].map(([code, text, strength]) => ({
      code,
      text,
      strength: strength as "strong" | "medium" | "check",
    })),
  },
  {
    candidate: "Ezetimibe",
    disease: "Gout",
    score: "0.693",
    reason: "Novel hypothesis with NPC1L1 and urate-transport plausibility",
    next: "Open 3D",
    evidencePosture: "Plausible",
    decision: "Review",
    decisionDetail: "No direct trial support",
    supportItems: [
      {
        label: "Target biology",
        value: "NPC1L1 expression and urate-transport context make the gout hypothesis biologically plausible.",
        tone: "success" as V2Tone,
      },
      {
        label: "Analogical support",
        value: "Anti-inflammatory and cholesterol-urate evidence provide indirect mechanism support.",
        tone: "warning" as V2Tone,
      },
      {
        label: "Novelty posture",
        value: "The dashboard should label this as a hypothesis, not as a validated clinical finding.",
        tone: "warning" as V2Tone,
      },
    ],
    weakItems: [
      "No direct trial support was found for Ezetimibe to gout.",
      "The supporting evidence is mechanistic and analogical rather than clinical.",
      "The candidate should be reviewed before adding to a validated shortlist.",
    ],
    workingConclusion:
      "Keep Ezetimibe to gout as a novel hypothesis only if the NPC1L1 and urate-transport evidence remains visible.",
    pathway: ["Compound", "NPC1L1 target context", "Urate transport", "Gout"],
    relationEvidence: [
      ["CtD", "Candidate is a novel compound-disease hypothesis", "medium"],
      ["CbG", "NPC1L1 target context is the key mechanism hook", "strong"],
      ["GpPW", "Pathway context connects cholesterol and urate transport", "medium"],
      ["DaG", "Disease-gene evidence needs validation", "check"],
    ].map(([code, text, strength]) => ({
      code,
      text,
      strength: strength as "strong" | "medium" | "check",
    })),
  },
];

export const candidates: V2Candidate[] = paperCandidates;

const candidateByEntity: Record<string, V2Candidate> = {
  Atherosclerosis: paperCandidates[0],
  Losartan: paperCandidates[0],
  "Liver cancer": paperCandidates[1],
  Mitomycin: paperCandidates[1],
  Gout: paperCandidates[2],
  Ezetimibe: paperCandidates[2],
  NPC1L1: paperCandidates[2],
};

export interface V2Session {
  sessionId?: string;
  selectedEntity: V2EntitySuggestion;
  runMode: V2RunMode;
  scoreThreshold: string;
  mechanismWeight: string;
  toggles: string[];
  selectedCandidate: V2Candidate;
  candidates: V2Candidate[];
}

export type V2SearchParams =
  | URLSearchParams
  | Pick<URLSearchParams, "get">
  | Record<string, string | string[] | undefined>;

export const v2Session: V2Session = {
  selectedEntity: entitySuggestions[0],
  runMode: "Hybrid" as V2RunMode,
  scoreThreshold: "0.65",
  mechanismWeight: "High",
  toggles: ["Include protein targets", "Show pathways", "Hide weak links"],
  selectedCandidate: candidates[0],
  candidates,
};

export function parseV2Session(params?: V2SearchParams): V2Session {
  const sessionId = readParam(params, "session_id");
  const entityName = readParam(params, "entity");
  const runMode = readParam(params, "runMode");
  const candidateDisease = readParam(params, "candidate");

  const selectedEntity =
    entitySuggestions.find((entity) => entity.name === entityName) ??
    v2Session.selectedEntity;
  const selectedCandidate =
    candidates.find(
      (candidate) =>
        candidate.disease === candidateDisease ||
        `${candidate.candidate} to ${candidate.disease}` === candidateDisease,
    ) ?? getV2CandidateForEntity(selectedEntity.name);

  return {
    ...v2Session,
    sessionId,
    selectedEntity,
    selectedCandidate,
    runMode: isV2RunMode(runMode) ? runMode : v2Session.runMode,
  };
}

export function getV2CandidateForEntity(entityName: string): V2Candidate {
  return candidateByEntity[entityName] ?? candidates[0];
}

export function buildV2Params(session: V2Session): string {
  const params = new URLSearchParams({
    entity: session.selectedEntity.name,
    runMode: session.runMode,
    candidate: session.selectedCandidate.disease,
  });
  if (session.sessionId) {
    params.set("session_id", session.sessionId);
  }

  return `?${params.toString()}`;
}

export function buildV2ParamsFromSessionId(sessionId: string): string {
  return `?${new URLSearchParams({ session_id: sessionId }).toString()}`;
}

function readParam(params: V2SearchParams | undefined, key: string): string | undefined {
  if (!params) return undefined;
  if (hasParamGetter(params)) return params.get(key) ?? undefined;
  const value = params[key];
  return Array.isArray(value) ? value[0] : value;
}

function hasParamGetter(
  params: V2SearchParams,
): params is URLSearchParams | Pick<URLSearchParams, "get"> {
  return typeof (params as Pick<URLSearchParams, "get">).get === "function";
}

function isV2RunMode(value: string | undefined): value is V2RunMode {
  return value === "Classical" || value === "Hybrid" || value === "Quantum hardware";
}

