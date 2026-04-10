export function getApiBaseUrl(): string {
  return (
    (typeof window !== "undefined"
      ? process.env.NEXT_PUBLIC_API_URL
      : process.env.NEXT_PUBLIC_API_URL) ?? "http://localhost:8000"
  );
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${getApiBaseUrl()}${path}`;
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(
      body?.detail ?? `API error ${res.status}: ${res.statusText}`,
    );
  }
  return res.json() as Promise<T>;
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

// ---------- /runs/latest ----------

export interface LatestRunResponse {
  status: string;
  results_dir: string;
  latest_csv: Record<string, unknown> | null;
  latest_json: Record<string, unknown> | null;
  message: string | null;
}

export function fetchLatestRun(): Promise<LatestRunResponse> {
  return apiFetch<LatestRunResponse>("/runs/latest");
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
  message?: string | null;
}

export function fetchVizModelMetrics(): Promise<VizModelMetricsResponse> {
  return apiFetch<VizModelMetricsResponse>("/viz/model-metrics");
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
}

export interface JobResponse {
  id: string;
  status: string;
  created_at: number;
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
