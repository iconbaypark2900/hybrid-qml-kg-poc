export function getApiBaseUrl(): string {
  if (
    typeof window !== "undefined" &&
    process.env.NEXT_PUBLIC_USE_API_PROXY !== "0"
  ) {
    return "/api/proxy";
  }

  return (
    (typeof window !== "undefined"
      ? process.env.NEXT_PUBLIC_API_URL
      : process.env.NEXT_PUBLIC_API_URL) ?? "http://127.0.0.1:8780"
  );
}

const DEFAULT_TIMEOUT_MS = Number(process.env.NEXT_PUBLIC_API_TIMEOUT_MS ?? 15000);
const DEFAULT_RETRY_COUNT = Number(process.env.NEXT_PUBLIC_API_RETRY_COUNT ?? 1);
const GET_CACHE_TTL_MS = Number(process.env.NEXT_PUBLIC_API_CACHE_TTL_MS ?? 15000);
const getCache = new Map<string, { expiresAt: number; value: unknown }>();

export class QGGApiError extends Error {
  status: number | null;
  path: string;
  requestId: string;
  retryable: boolean;

  constructor({
    message,
    status,
    path,
    requestId,
    retryable,
  }: {
    message: string;
    status: number | null;
    path: string;
    requestId: string;
    retryable: boolean;
  }) {
    super(message);
    this.name = "QGGApiError";
    this.status = status;
    this.path = path;
    this.requestId = requestId;
    this.retryable = retryable;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const cacheKey = `${method}:${path}`;
  const now = Date.now();
  if (method === "GET" && isCacheableGet(path)) {
    const cached = getCache.get(cacheKey);
    if (cached && cached.expiresAt > now) {
      return cached.value as T;
    }
  }

  const attempts = method === "GET" ? DEFAULT_RETRY_COUNT + 1 : 1;
  let lastError: QGGApiError | null = null;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const value = await apiFetchOnce<T>(path, init, method);
      if (method === "GET" && isCacheableGet(path)) {
        getCache.set(cacheKey, {
          expiresAt: Date.now() + GET_CACHE_TTL_MS,
          value,
        });
      }
      return value;
    } catch (error) {
      if (!(error instanceof QGGApiError)) throw error;
      lastError = error;
      if (!error.retryable || attempt === attempts - 1) {
        throw error;
      }
      await delay(200 * (attempt + 1));
    }
  }

  throw lastError ?? new Error("API request failed");
}

async function apiFetchOnce<T>(
  path: string,
  init: RequestInit | undefined,
  method: string,
): Promise<T> {
  const url = `${getApiBaseUrl()}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  const requestId = makeRequestId();
  const headers = new Headers(init?.headers);
  headers.set("x-qgg-request-id", requestId);

  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      method,
      headers,
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeout);
    const aborted = error instanceof DOMException && error.name === "AbortError";
    throw new QGGApiError({
      message: aborted
        ? `API timeout after ${DEFAULT_TIMEOUT_MS}ms`
        : error instanceof Error
          ? error.message
          : "API request failed",
      status: null,
      path,
      requestId,
      retryable: method === "GET",
    });
  }
  clearTimeout(timeout);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new QGGApiError({
      message: body?.detail ?? `API error ${res.status}: ${res.statusText}`,
      status: res.status,
      path,
      requestId,
      retryable: method === "GET" && res.status >= 500,
    });
  }
  return res.json() as Promise<T>;
}

function isCacheableGet(path: string): boolean {
  if (path.startsWith("/jobs") || path.startsWith("/research-sessions")) return false;
  return (
    path.startsWith("/status") ||
    path.startsWith("/runs/latest") ||
    path.startsWith("/analysis") ||
    path.startsWith("/viz") ||
    path.startsWith("/kg") ||
    path.startsWith("/repurposing") ||
    path.startsWith("/quantum/config")
  );
}

function makeRequestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `req_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function tenantHeaders(tenantId?: string): HeadersInit {
  const headers: Record<string, string> = {};
  const cleaned = tenantId?.trim();
  if (cleaned) headers["X-Tenant-Id"] = cleaned;
  return headers;
}

function withJsonHeaders(headers: HeadersInit = {}): HeadersInit {
  return {
    ...(headers as Record<string, string>),
    "Content-Type": "application/json",
  };
}

// ---------- /status ----------

export interface StatusResponse {
  status: string;
  orchestrator_ready: boolean;
  classical_model_loaded: boolean;
  quantum_model_loaded: boolean;
  entity_count: number;
  supported_relations: string[];
}

export function fetchStatus(): Promise<StatusResponse> {
  return apiFetch<StatusResponse>("/status");
}

// ---------- /ops ----------

export interface OpsHealthResponse {
  status: string;
  uptime_seconds: number;
  request_counts: Record<string, number>;
  request_failures: Record<string, number>;
  recent_failure_count: number;
  orchestrator_ready: boolean;
  classical_model_loaded: boolean;
  quantum_model_loaded: boolean;
  cors_allowed_origins: string[];
  internal_auth_enabled: boolean;
  environment: string;
}

export interface OpsFailureResponse {
  request_id: string;
  method: string;
  path: string;
  status_code: number;
  duration_ms: number;
  timestamp: number;
}

export function fetchOpsHealth(): Promise<OpsHealthResponse> {
  return apiFetch<OpsHealthResponse>("/ops/health");
}

export function fetchOpsErrors(limit?: number): Promise<OpsFailureResponse[]> {
  const query = limit ? `?limit=${limit}` : "";
  return apiFetch<OpsFailureResponse[]>(`/ops/errors${query}`);
}

// ---------- /runs/latest ----------

export type EvidenceSourceKind =
  | "run_artifact"
  | "api"
  | "dataset"
  | "external"
  | "config"
  | "fallback";

export interface EvidenceProvenance {
  endpoint: string;
  source_kind: EvidenceSourceKind;
  artifact_path?: string | null;
  artifact_name?: string | null;
  mtime_epoch?: number | null;
  run_timestamp?: string | null;
  relation?: string | null;
  model_name?: string | null;
  model_type?: string | null;
  embedding_method?: string | null;
  embedding_dim?: number | null;
  seed?: number | null;
  config?: Record<string, unknown> | null;
  notes?: string[];
}

export interface CandidateEvidenceLink {
  kind: "kg_edge" | "clinical_trial" | "mechanism" | "model_score";
  label: string;
  source: string;
  relation?: string | null;
  url?: string | null;
  score?: number | null;
  provenance?: EvidenceProvenance | null;
}

export interface LatestCsvArtifact {
  path: string;
  mtime_epoch: number;
  row: Record<string, string | null> | null;
}

export interface LatestJsonArtifact {
  path: string;
  mtime_epoch: number;
  ranking: Array<Record<string, unknown>>;
  relation?: string | null;
  timestamp?: string | null;
  config?: Record<string, unknown> | null;
  embedding_method?: string | null;
  embedding_dim?: number | null;
  qml_dim?: number | null;
  qml_feature_map?: string | null;
  seed?: number | null;
  job_id?: string | null;
  run_id?: string | null;
}

export interface LatestRunResponse {
  status: string;
  results_dir: string;
  latest_csv: LatestCsvArtifact | null;
  latest_json: LatestJsonArtifact | null;
  provenance: EvidenceProvenance[];
  message: string | null;
}

export function fetchLatestRun(): Promise<LatestRunResponse> {
  return apiFetch<LatestRunResponse>("/runs/latest");
}

// ---------- /repurposing ----------

export interface RepurposingDisease {
  id: string;
  name: string;
  cohort: string;
  source: string;
  sample_count: number;
  smallest_class_count: number;
  evidence_status: string;
  notes: string[];
}

export interface RepurposingEvidenceComponent {
  label: string;
  value: string;
  status: string;
  detail: string;
}

export interface RepurposingStructureEvidence {
  status: string;
  target_count: number;
  available_target_count: number;
  missing_rate: number;
  target_ids: string[];
  provenance: EvidenceProvenance[];
}

export interface RepurposingStructureTargets {
  mapping_status: string;
  target_source?: string | null;
  compound_kg_id?: string | null;
  disease_kg_id?: string | null;
  target_ids: string[];
  target_count: number;
  structure_artifact_target_count: number;
  parsed_structure_count: number;
  missing_structure_target_ids: string[];
  structure_feature_score: number;
  notes?: string | null;
}

export interface RepurposingProteinStructureEvidence {
  target_id: string;
  target_name: string;
  display_name: string;
  artifact_available: boolean;
  parse_success: boolean;
  confidence: Record<string, unknown>;
  feature_summary: Record<string, unknown>;
  viewer: {
    kind: string;
    supports_3d: boolean;
    preferred_viewer: string;
    artifact_path?: string | null;
    artifact_format?: string | null;
  };
  claim_policy: string;
}

export interface RepurposingAudit {
  status: string;
  claim_policy: string;
  warnings: string[];
  quantum_advantage_claim_allowed: boolean;
  clinical_claim_allowed: boolean;
}

export interface RepurposingCandidate {
  compound_id: string;
  compound_name: string;
  disease_id: string;
  disease_name: string;
  hypothesis_score: number;
  scoring_mode: string;
  rank: number;
  summary: string;
  evidence_components: RepurposingEvidenceComponent[];
  kg_paths: string[];
  rnaseq_signature: Record<string, unknown>;
  structure: RepurposingStructureEvidence;
  structure_targets: RepurposingStructureTargets;
  protein_structures: RepurposingProteinStructureEvidence[];
  classical_ml: Record<string, unknown>;
  quantum_benchmark: Record<string, unknown>;
  audit: RepurposingAudit;
}

export interface RepurposingDiseasesResponse {
  status: string;
  diseases: RepurposingDisease[];
  provenance: EvidenceProvenance[];
}

export interface RepurposingCandidatesResponse {
  status: string;
  disease: RepurposingDisease;
  candidates: RepurposingCandidate[];
  scoring_modes: string[];
  manifest: Record<string, unknown>;
  provenance: EvidenceProvenance[];
  message?: string | null;
}

export function fetchRepurposingDiseases(): Promise<RepurposingDiseasesResponse> {
  return apiFetch<RepurposingDiseasesResponse>("/repurposing/diseases");
}

export function fetchRepurposingCandidates(
  diseaseId = "brca_external_validation",
): Promise<RepurposingCandidatesResponse> {
  return apiFetch<RepurposingCandidatesResponse>(
    `/repurposing/candidates?disease_id=${encodeURIComponent(diseaseId)}`,
  );
}


export function getRepurposingEvidenceBundleUrl(
  diseaseId = "brca_external_validation",
  format: "json" | "markdown" = "json",
): string {
  return `${getApiBaseUrl()}/repurposing/evidence-bundle?disease_id=${encodeURIComponent(diseaseId)}&format=${format}`;
}

export function getStructureArtifactUrl(artifactPath: string): string {
  return `${getApiBaseUrl()}/repurposing/structure-artifact?path=${encodeURIComponent(artifactPath)}`;
}

// ---------- /predict-link ----------

export interface PredictionRequest {
  drug: string;
  disease: string;
  method?: string;
}

export interface PredictionResponse {
  drug: string;
  disease: string;
  drug_id: string;
  disease_id: string;
  link_probability: number;
  model_used: string;
  status: string;
  error_message: string | null;
}

export function predictLink(
  req: PredictionRequest,
): Promise<PredictionResponse> {
  return apiFetch<PredictionResponse>("/predict-link", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

// ---------- /batch-predict ----------

export function batchPredict(
  reqs: PredictionRequest[],
): Promise<PredictionResponse[]> {
  return apiFetch<PredictionResponse[]>("/batch-predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(reqs),
  });
}

// ---------- /ranked-mechanisms ----------

export interface RankedMechanismsRequest {
  hypothesis_id: string;
  disease_id: string;
  top_k?: number;
}

export interface RankedCandidate {
  compound_id: string;
  compound_name: string;
  score: number;
  mechanism_summary: string;
}

export interface RankedMechanismsResponse {
  ranked_candidates: RankedCandidate[];
  model_used: string;
  hypothesis_id: string;
  status?: string;
  error_message?: string | null;
}

export function rankedMechanisms(
  req: RankedMechanismsRequest,
): Promise<RankedMechanismsResponse> {
  return apiFetch<RankedMechanismsResponse>("/ranked-mechanisms", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

// ---------- /hypotheses ----------

export type HypothesisStatus = "draft" | "active" | "tested";

export interface Hypothesis {
  id: string;
  name: string;
  description: string;
  disease_focus?: string | null;
  mechanism_type?: string | null;
  notes?: string | null;
  status: HypothesisStatus;
  created_at: number;
  updated_at: number;
  last_tested_run_id?: string | null;
}

export interface HypothesisCreateRequest {
  name: string;
  description: string;
  disease_focus?: string;
  mechanism_type?: string;
  notes?: string;
  status?: HypothesisStatus;
}

export interface HypothesisUpdateRequest {
  name?: string;
  description?: string;
  disease_focus?: string | null;
  mechanism_type?: string | null;
  notes?: string | null;
  status?: HypothesisStatus;
  last_tested_run_id?: string | null;
}

export function fetchHypotheses(): Promise<Hypothesis[]> {
  return apiFetch<Hypothesis[]>("/hypotheses");
}

export function fetchHypothesis(hypothesisId: string): Promise<Hypothesis> {
  return apiFetch<Hypothesis>(`/hypotheses/${encodeURIComponent(hypothesisId)}`);
}

export function createHypothesis(
  req: HypothesisCreateRequest,
): Promise<Hypothesis> {
  return apiFetch<Hypothesis>("/hypotheses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function updateHypothesis(
  hypothesisId: string,
  req: HypothesisUpdateRequest,
): Promise<Hypothesis> {
  return apiFetch<Hypothesis>(`/hypotheses/${encodeURIComponent(hypothesisId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

// ---------- /research-sessions ----------

export interface ResearchSession {
  id: string;
  hypothesis_id?: string | null;
  title: string;
  reviewer_name?: string | null;
  reviewer_email?: string | null;
  selected_entity: Record<string, unknown>;
  selected_candidate: Record<string, unknown>;
  run_mode: string;
  score_threshold?: string | null;
  mechanism_weight?: string | null;
  decision: string;
  notes?: string | null;
  evidence_state: Record<string, unknown>;
  provenance: Array<Record<string, unknown>>;
  created_at: number;
  updated_at: number;
  exported_at?: number | null;
}

export interface ResearchSessionCreateRequest {
  title: string;
  reviewer_name?: string | null;
  reviewer_email?: string | null;
  selected_entity: Record<string, unknown>;
  selected_candidate: Record<string, unknown>;
  run_mode: string;
  score_threshold?: string | null;
  mechanism_weight?: string | null;
  decision: string;
  notes?: string | null;
  evidence_state: Record<string, unknown>;
  provenance: Array<Record<string, unknown>>;
  hypothesis_id?: string | null;
}

export type ResearchSessionUpdateRequest = Partial<ResearchSessionCreateRequest>;

export interface EvidencePacketResponse {
  evidence_packet_version: number;
  session: ResearchSession;
  research_context: Record<string, unknown>;
  decision: Record<string, unknown>;
  evidence_state: Record<string, unknown>;
  provenance: Array<Record<string, unknown>>;
}

export function fetchResearchSessions(params?: {
  hypothesisId?: string;
  reviewerEmail?: string;
}): Promise<ResearchSession[]> {
  const search = new URLSearchParams();
  if (params?.hypothesisId) search.set("hypothesis_id", params.hypothesisId);
  if (params?.reviewerEmail) search.set("reviewer_email", params.reviewerEmail);
  const qs = search.toString();
  return apiFetch<ResearchSession[]>(`/research-sessions${qs ? `?${qs}` : ""}`);
}

export function fetchResearchSession(sessionId: string): Promise<ResearchSession> {
  return apiFetch<ResearchSession>(
    `/research-sessions/${encodeURIComponent(sessionId)}`,
  );
}

export function createResearchSession(
  req: ResearchSessionCreateRequest,
): Promise<ResearchSession> {
  return apiFetch<ResearchSession>("/research-sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function updateResearchSession(
  sessionId: string,
  req: ResearchSessionUpdateRequest,
): Promise<ResearchSession> {
  return apiFetch<ResearchSession>(
    `/research-sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    },
  );
}

export function fetchEvidencePacket(
  sessionId: string,
): Promise<EvidencePacketResponse> {
  return apiFetch<EvidencePacketResponse>(
    `/research-sessions/${encodeURIComponent(sessionId)}/export`,
  );
}

export function exportEvidencePacketUrl(sessionId: string): string {
  return `${getApiBaseUrl()}/research-sessions/${encodeURIComponent(sessionId)}/export`;
}

// ---------- /kg ----------

export interface KGStatsResponse {
  status: string;
  entity_count: number;
  edge_count: number;
  relation_types: string[];
  embedding_dim: number | null;
  qml_dim: number | null;
  sample_entities: string[];
  sample_edges: Array<Record<string, string>>;
}

export function fetchKGStats(): Promise<KGStatsResponse> {
  return apiFetch<KGStatsResponse>("/kg/stats");
}

// ---------- /quantum ----------

export interface QuantumConfigResponse {
  status: string;
  execution_mode: string | null;
  backend: string | null;
  shots: number | null;
  quantum_model_loaded: boolean;
  config: Record<string, unknown> | null;
}

export function fetchQuantumConfig(): Promise<QuantumConfigResponse> {
  return apiFetch<QuantumConfigResponse>("/quantum/config");
}

// ---------- /quantum/runtime/verify (BYOK — ephemeral, never stored server-side) ----------

export interface QuantumRuntimeVerifyRequest {
  api_token: string;
  instance_crn?: string;
  channel?: string;
}

export interface QuantumRuntimeVerifyResponse {
  status: string;
  message: string;
  backend_count: number;
  hardware_backend_names: string[];
  simulator_count: number;
  instances_count: number | null;
}

export function verifyQuantumRuntime(
  req: QuantumRuntimeVerifyRequest,
): Promise<QuantumRuntimeVerifyResponse> {
  return apiFetch<QuantumRuntimeVerifyResponse>("/quantum/runtime/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_token: req.api_token,
      instance_crn: req.instance_crn?.trim() || undefined,
      channel: req.channel?.trim() || "ibm_quantum_platform",
    }),
  });
}

// ---------- /config/ibm-quantum ----------

export interface IBMQuantumConfigResponse {
  configured: boolean;
  tenant_id: string;
  provider: string;
  instance: string | null;
  channel: string;
  token_preview: string | null;
  secret_storage: string | null;
  created_at: number | null;
  updated_at: number | null;
  last_verified_at: number | null;
  message: string | null;
}

export interface IBMQuantumConfigSaveRequest {
  token: string;
  instance?: string;
  channel?: string;
  tenantId?: string;
}

export interface IBMQuantumConfigVerifyRequest {
  token?: string;
  instance?: string;
  channel?: string;
  tenantId?: string;
}

export interface IBMQuantumConfigVerifyResponse
  extends QuantumRuntimeVerifyResponse {
  tenant_id: string;
  used_stored_credentials: boolean;
}

export function fetchIBMQuantumConfig(
  tenantId?: string,
): Promise<IBMQuantumConfigResponse> {
  return apiFetch<IBMQuantumConfigResponse>("/config/ibm-quantum", {
    headers: tenantHeaders(tenantId),
  });
}

export function saveIBMQuantumConfig(
  req: IBMQuantumConfigSaveRequest,
): Promise<IBMQuantumConfigResponse> {
  return apiFetch<IBMQuantumConfigResponse>("/config/ibm-quantum", {
    method: "POST",
    headers: withJsonHeaders(tenantHeaders(req.tenantId)),
    body: JSON.stringify({
      token: req.token,
      instance: req.instance?.trim() || undefined,
      channel: req.channel?.trim() || "ibm_quantum_platform",
    }),
  });
}

export function verifyIBMQuantumConfig(
  req: IBMQuantumConfigVerifyRequest,
): Promise<IBMQuantumConfigVerifyResponse> {
  return apiFetch<IBMQuantumConfigVerifyResponse>("/config/ibm-quantum/verify", {
    method: "POST",
    headers: withJsonHeaders(tenantHeaders(req.tenantId)),
    body: JSON.stringify({
      token: req.token?.trim() || undefined,
      instance: req.instance?.trim() || undefined,
      channel: req.channel?.trim() || "ibm_quantum_platform",
    }),
  });
}

// ---------- /analysis ----------

export interface AnalysisSummaryResponse {
  status: string;
  best_model: string | null;
  best_pr_auc: number | null;
  model_count: number;
  classical_count: number;
  quantum_count: number;
  ensemble_count: number;
  ranking: Array<Record<string, unknown>> | null;
  relation: string | null;
  run_timestamp: string | null;
  provenance: EvidenceProvenance[];
  message: string | null;
}

export function fetchAnalysisSummary(): Promise<AnalysisSummaryResponse> {
  return apiFetch<AnalysisSummaryResponse>("/analysis/summary");
}

// ---------- /exports ----------

export interface ExportFileInfo {
  name: string;
  size_bytes: number;
  modified: number;
}

export interface ExportListResponse {
  status: string;
  files: ExportFileInfo[];
}

export function fetchExports(): Promise<ExportListResponse> {
  return apiFetch<ExportListResponse>("/exports");
}

export function exportDownloadUrl(filename: string): string {
  return `${getApiBaseUrl()}/exports/${encodeURIComponent(filename)}`;
}

// ---------- /viz ----------

export interface VizAtom {
  element: string;
  x: number;
  y: number;
  z: number;
}

export interface VizBond {
  source: number; // atom index
  target: number;
  order: number;
}

export interface VizMoleculeResponse {
  status: string;
  compound_id?: string | null;
  compound_name?: string | null;
  atoms: VizAtom[];
  bonds: VizBond[];
  provenance?: EvidenceProvenance[];
  message?: string | null;
}

export function fetchVizMolecule(
  compound: string,
): Promise<VizMoleculeResponse> {
  return apiFetch<VizMoleculeResponse>(
    `/viz/molecule?compound=${encodeURIComponent(compound)}`,
  );
}

export interface VizEmbNode {
  id: string;
  label: string;
  type: "compound" | "disease";
  x: number;
  y: number;
  z: number;
}

export interface VizEmbEdge {
  source: string;
  target: string;
}

export interface VizEmbeddingsResponse {
  status: string;
  nodes: VizEmbNode[];
  edges: VizEmbEdge[];
  model_name?: string | null;
  available_models: string[];
  compound_count: number;
  disease_count: number;
  edge_count: number;
  variance_explained?: number[] | null;
  projection?: string | null;
  projection_note?: string | null;
  message?: string | null;
}

export type EmbeddingProjection = "pca" | "pca_stretch" | "tsne";

export function fetchVizEmbeddings(
  model?: string,
  projection?: EmbeddingProjection,
): Promise<VizEmbeddingsResponse> {
  const params = new URLSearchParams();
  if (model) params.set("model", model);
  if (projection) params.set("projection", projection);
  const qs = params.toString();
  return apiFetch<VizEmbeddingsResponse>(
    `/viz/embeddings${qs ? `?${qs}` : ""}`,
  );
}

export interface VizKGNode {
  id: string;
  label: string;
  entity_type: string;
  score?: number | null;
}

export interface VizKGLink {
  source: string;
  target: string;
  relation: string;
}

export interface VizKGSubgraphResponse {
  status: string;
  nodes: VizKGNode[];
  links: VizKGLink[];
  center_entity?: string | null;
  provenance?: EvidenceProvenance[];
  message?: string | null;
}

export function fetchVizKGSubgraph(
  entity: string,
  maxNodes?: number,
  hops?: number,
  relationFilter?: string,
): Promise<VizKGSubgraphResponse> {
  const params = new URLSearchParams({ entity });
  if (maxNodes) params.set("max_nodes", String(maxNodes));
  if (hops) params.set("hops", String(hops));
  if (relationFilter) params.set("relation_filter", relationFilter);
  return apiFetch<VizKGSubgraphResponse>(`/viz/kg-subgraph?${params}`);
}

export interface KGSearchResult {
  id: string;
  name: string;
  kind: string;
}

export function searchKGEntities(
  q: string,
  limit?: number,
): Promise<{ results: KGSearchResult[] }> {
  const params = new URLSearchParams({ q });
  if (limit) params.set("limit", String(limit));
  return apiFetch<{ results: KGSearchResult[] }>(`/viz/kg-search?${params}`);
}

export interface VizPrediction {
  compound_id: string;
  compound_name: string;
  disease_id: string;
  disease_name: string;
  score: number;
  confidence: "High" | "Medium" | "Low";
  model_used: string;
}

export interface VizPredictionsResponse {
  status: string;
  predictions: VizPrediction[];
  message?: string | null;
}

export function fetchVizPredictions(
  topK?: number,
): Promise<VizPredictionsResponse> {
  const params = topK ? `?top_k=${topK}` : "";
  return apiFetch<VizPredictionsResponse>(`/viz/predictions${params}`);
}

export interface VizRunPrediction {
  compound_id: string;
  compound_name: string;
  disease_id: string;
  disease_name: string;
  y_true: number;
  score_classical?: number | null;
  score_quantum?: number | null;
  score: number;
  confidence: "High" | "Medium" | "Low";
}

export interface VizRunPredictionsResponse {
  status: string;
  predictions: VizRunPrediction[];
  run_timestamp?: string | null;
  source_file?: string | null;
  available_runs: string[];
  provenance?: EvidenceProvenance[];
  message?: string | null;
}

export function fetchVizRunPredictions(
  topK?: number,
  run?: string,
): Promise<VizRunPredictionsResponse> {
  const params = new URLSearchParams();
  if (topK) params.set("top_k", String(topK));
  if (run) params.set("run", run);
  const qs = params.toString();
  return apiFetch<VizRunPredictionsResponse>(
    `/viz/run-predictions${qs ? `?${qs}` : ""}`,
  );
}

export interface VizModelMetric {
  name: string;
  type: string;
  pr_auc: number;
  accuracy: number;
  fit_time: number;
}

export interface VizModelMetricsResponse {
  status: string;
  models: VizModelMetric[];
  ablation?: Record<string, number> | null;
  relation?: string | null;
  run_timestamp?: string | null;
  provenance?: EvidenceProvenance[];
  message?: string | null;
}

export function fetchVizModelMetrics(): Promise<VizModelMetricsResponse> {
  return apiFetch<VizModelMetricsResponse>("/viz/model-metrics");
}

export interface EmbeddingVectorResponse {
  status: string;
  drug: string;
  disease: string;
  drug_id: string;
  disease_id: string;
  drug_name: string;
  disease_name: string;
  drug_embedding: number[];
  disease_embedding: number[];
  abs_diff: number[];
  hadamard_product: number[];
  qml_dim: number;
  in_training_set: { drug: boolean; disease: boolean };
  message?: string | null;
}

export function fetchVizEmbeddingVector(
  drug: string,
  disease: string,
): Promise<EmbeddingVectorResponse> {
  const params = new URLSearchParams({ drug, disease });
  return apiFetch<EmbeddingVectorResponse>(`/viz/embedding-vector?${params}`);
}

export interface VizCircuitResponse {
  status: string;
  n_qubits: number;
  n_reps: number;
  entanglement: string;
  feature_map: string;
  execution_mode?: string | null;
  backend?: string | null;
  shots?: number | null;
  provenance?: EvidenceProvenance[];
}

export function fetchVizCircuitParams(): Promise<VizCircuitResponse> {
  return apiFetch<VizCircuitResponse>("/viz/circuit-params");
}

// ---------- /jobs ----------

export interface JobCreateRequest {
  relation?: string;
  embedding_method?: string;
  embedding_dim?: number;
  embedding_epochs?: number;
  qml_dim?: number;
  qml_feature_map?: string;
  qml_feature_map_reps?: number;
  fast_mode?: boolean;
  skip_quantum?: boolean;
  run_ensemble?: boolean;
  ensemble_method?: string;
  tune_classical?: boolean;
  results_dir?: string;
  quantum_config_path?: string;
  hypothesis_id?: string;
  experiment_note?: string;
  experiment_tags?: string[];
}

export interface JobResponse {
  id: string;
  status: string;
  created_at: number;
  hypothesis_id?: string | null;
  experiment_metadata?: Record<string, unknown>;
  flags?: Record<string, unknown>;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  error: string | null;
  log_tail: string[] | null;
}

export function createPipelineJob(
  req: JobCreateRequest,
): Promise<JobResponse> {
  return apiFetch<JobResponse>("/jobs/pipeline", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function fetchJob(jobId: string): Promise<JobResponse> {
  return apiFetch<JobResponse>(`/jobs/${jobId}`);
}

export function fetchJobs(): Promise<JobResponse[]> {
  return apiFetch<JobResponse[]>("/jobs");
}

export interface ExperimentHistory {
  job_id: string;
  hypothesis_id?: string | null;
  relation?: string | null;
  config: Record<string, unknown>;
  metadata: Record<string, unknown>;
  status: string;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  exit_code?: number | null;
  error?: string | null;
  run_timestamp?: string | null;
}

export function fetchExperimentsHistory(
  hypothesisId?: string,
): Promise<ExperimentHistory[]> {
  const params = new URLSearchParams();
  if (hypothesisId) params.set("hypothesis_id", hypothesisId);
  const qs = params.toString();
  return apiFetch<ExperimentHistory[]>(
    `/experiments/history${qs ? `?${qs}` : ""}`,
  );
}

export function fetchHypothesisExperiments(
  hypothesisId: string,
): Promise<ExperimentHistory[]> {
  return apiFetch<ExperimentHistory[]>(
    `/hypotheses/${encodeURIComponent(hypothesisId)}/experiments`,
  );
}
