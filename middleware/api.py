# middleware/api.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import logging
import os
import numpy as np
from .orchestrator import LinkPredictionOrchestrator
from utils.latest_run import get_latest_run_snapshot
from .job_manager import job_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid QML-KG Biomedical Link Predictor",
    description="""
    A hybrid quantum-classical system for predicting drug-disease treatment relationships
    using the Hetionet knowledge graph.
    
    - **Classical Baseline**: Logistic Regression on KG embeddings
    - **Quantum Model**: Variational Quantum Classifier (VQC)
    - **Use Case**: Drug repurposing and treatment discovery
    """,
    version="0.1.0",
    contact={
        "name": "Your Name",
        "email": "you@example.com",
    },
)

# Enable CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator (singleton)
try:
    orchestrator = LinkPredictionOrchestrator(use_quantum=False)
    logger.info("Orchestrator initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize orchestrator: {e}")
    orchestrator = None


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    drug: str
    disease: str
    method: Optional[str] = "auto"  # "classical", "quantum", or "auto"


class PredictionResponse(BaseModel):
    drug: str
    disease: str
    drug_id: str
    disease_id: str
    link_probability: float
    model_used: str
    status: str
    error_message: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    orchestrator_ready: bool
    classical_model_loaded: bool
    quantum_model_loaded: bool
    entity_count: int
    supported_relations: List[str] = ["CtD"]  # Compound treats Disease


class RankedMechanismsRequest(BaseModel):
    hypothesis_id: str  # "H-001" | "H-002" | "H-003"
    disease_id: str     # Disease name or Hetionet ID
    top_k: Optional[int] = 50


class RankedCandidate(BaseModel):
    compound_id: str
    compound_name: str
    score: float
    mechanism_summary: str


class RankedMechanismsResponse(BaseModel):
    ranked_candidates: List[RankedCandidate]
    model_used: str
    hypothesis_id: str
    status: Optional[str] = "success"
    error_message: Optional[str] = None


class LatestRunResponse(BaseModel):
    """Latest pipeline artifacts under ``results/`` (see ``utils/latest_run.py``)."""

    status: str
    results_dir: str
    latest_csv: Optional[Dict[str, Any]] = None
    latest_json: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class JobCreateRequest(BaseModel):
    """Subset of ``run_optimized_pipeline.py`` flags exposed to the UI."""
    relation: str = "CtD"
    embedding_method: str = "ComplEx"
    embedding_dim: int = 64
    embedding_epochs: int = 100
    qml_dim: int = 12
    qml_feature_map: str = "ZZ"
    qml_feature_map_reps: int = 2
    fast_mode: bool = True
    skip_quantum: bool = False
    run_ensemble: bool = False
    ensemble_method: str = "weighted_average"
    tune_classical: bool = False
    results_dir: str = "results"
    quantum_config_path: str = "config/quantum_config.yaml"


class JobResponse(BaseModel):
    id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    log_tail: Optional[List[str]] = None


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "Hybrid QML-KG API. Visit /docs for interactive documentation."}


@app.get("/runs/latest", response_model=LatestRunResponse)
async def get_runs_latest():
    """
    Latest run summary: ``latest_run.csv`` row plus newest ``optimized_results_*.json``
    (by mtime), using the same resolution rules as the Streamlit benchmark dashboard.
    """
    payload = get_latest_run_snapshot()
    return LatestRunResponse(**payload)


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system health and readiness status."""
    if orchestrator is None:
        return StatusResponse(
            status="error",
            orchestrator_ready=False,
            classical_model_loaded=False,
            quantum_model_loaded=False,
            entity_count=0
        )
    
    return StatusResponse(
        status="healthy",
        orchestrator_ready=True,
        classical_model_loaded=True,  # Simplified - you could check actual model
        quantum_model_loaded=False,   # Disabled in orchestrator for now
        entity_count=len(orchestrator.embedder.entity_to_id) if orchestrator.embedder else 0
    )


@app.post("/predict-link", response_model=PredictionResponse)
async def predict_link(request: PredictionRequest):
    """
    Predict the probability that a drug treats a disease.
    
    Examples:
    - drug: "DB00945", disease: "DOID_9352" (Aspirin for Diabetes)
    - drug: "Dexamethasone", disease: "DOID_0060048" (COVID-19)
    """
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        result = orchestrator.predict_link_probability(
            drug=request.drug,
            disease=request.disease,
            method=request.method
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error_message"])
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/predict-link", response_model=PredictionResponse)
async def predict_link_get(
    drug: str = Query(..., description="Drug name or ID (e.g., 'Aspirin' or 'DB00945')"),
    disease: str = Query(..., description="Disease name or ID (e.g., 'Diabetes' or 'DOID_9352')"),
    method: str = Query("auto", description="Prediction method: 'classical', 'quantum', or 'auto'")
):
    """
    GET version of predict-link (useful for browser testing).
    """
    request = PredictionRequest(drug=drug, disease=disease, method=method)
    return await predict_link(request)


@app.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_predict(requests: List[PredictionRequest]):
    """
    Predict multiple drug-disease pairs in a single request.
    """
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    results = []
    for req in requests:
        try:
            result = orchestrator.predict_link_probability(
                drug=req.drug,
                disease=req.disease,
                method=req.method
            )
            results.append(PredictionResponse(**result))
        except Exception as e:
            # Return error result instead of failing entire batch
            results.append(PredictionResponse(
                drug=req.drug,
                disease=req.disease,
                drug_id="",
                disease_id="",
                link_probability=0.0,
                model_used="error",
                status="error",
                error_message=str(e)
            ))
    
    return results


@app.post("/ranked-mechanisms", response_model=RankedMechanismsResponse)
async def ranked_mechanisms(request: RankedMechanismsRequest):
    """
    Rank intervention candidates for a disease using mechanism-informed hypothesis.

    Uses the mechanism subgraph (H-001, H-002, or H-003) to filter and score
    compound-disease pairs. Returns top-k candidates sorted by link probability.
    """
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        result = orchestrator.rank_mechanism_candidates(
            hypothesis_id=request.hypothesis_id,
            disease_id=request.disease_id,
            top_k=request.top_k or 50,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error_message", "Ranking failed"))
        return RankedMechanismsResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ranked mechanisms failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class KGStatsResponse(BaseModel):
    """Knowledge graph summary statistics."""
    status: str
    entity_count: int = 0
    edge_count: int = 0
    relation_types: List[str] = []
    embedding_dim: Optional[int] = None
    qml_dim: Optional[int] = None
    sample_entities: List[str] = []
    sample_edges: List[Dict[str, str]] = []


class QuantumConfigResponse(BaseModel):
    """Quantum execution configuration and model status."""
    status: str
    execution_mode: Optional[str] = None
    backend: Optional[str] = None
    shots: Optional[int] = None
    quantum_model_loaded: bool = False
    config: Optional[Dict[str, Any]] = None


@app.get("/kg/stats", response_model=KGStatsResponse)
async def kg_stats():
    """Knowledge graph statistics from the loaded embedder."""
    if orchestrator is None or orchestrator.embedder is None:
        return KGStatsResponse(status="unavailable")

    emb = orchestrator.embedder
    entity_count = len(emb.entity_to_id) if emb.entity_to_id else 0
    sample_entities = list(emb.entity_to_id.keys())[:20] if emb.entity_to_id else []

    edge_count = 0
    relation_types: List[str] = []
    sample_edges: List[Dict[str, str]] = []
    try:
        from kg_layer.kg_loader import load_hetionet_edges
        df = load_hetionet_edges(data_dir=orchestrator.data_dir)
        edge_count = len(df)
        relation_types = sorted(df["metaedge"].unique().tolist())
        sample_edges = df.head(10).to_dict("records")
    except Exception as exc:
        logger.warning(f"Could not load edges for stats: {exc}")

    return KGStatsResponse(
        status="ok",
        entity_count=entity_count,
        edge_count=edge_count,
        relation_types=relation_types,
        embedding_dim=emb.embedding_dim,
        qml_dim=emb.qml_dim,
        sample_entities=sample_entities,
        sample_edges=sample_edges,
    )


@app.get("/quantum/config", response_model=QuantumConfigResponse)
async def quantum_config():
    """Current quantum execution configuration."""
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "quantum_config.yaml")
    config_data: Optional[Dict[str, Any]] = None
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning(f"Could not load quantum config: {exc}")

    q = (config_data or {}).get("quantum", {})
    execution_mode = q.get("execution_mode")
    backend = None
    shots = None
    if execution_mode == "heron":
        backend = q.get("heron", {}).get("backend")
        shots = q.get("heron", {}).get("shots")
    elif execution_mode == "simulator":
        shots = q.get("simulator", {}).get("shots")

    quantum_loaded = False
    if orchestrator is not None:
        quantum_loaded = getattr(orchestrator, "use_quantum", False)

    return QuantumConfigResponse(
        status="ok",
        execution_mode=execution_mode,
        backend=backend,
        shots=shots,
        quantum_model_loaded=quantum_loaded,
        config=config_data,
    )


class AnalysisSummaryResponse(BaseModel):
    """Aggregated metrics derived from the latest pipeline run."""
    status: str
    best_model: Optional[str] = None
    best_pr_auc: Optional[float] = None
    model_count: int = 0
    classical_count: int = 0
    quantum_count: int = 0
    ensemble_count: int = 0
    ranking: Optional[List[Dict[str, Any]]] = None
    relation: Optional[str] = None
    run_timestamp: Optional[str] = None
    message: Optional[str] = None


class ExportFileInfo(BaseModel):
    name: str
    size_bytes: int
    modified: float


class ExportListResponse(BaseModel):
    status: str
    files: List[ExportFileInfo]


@app.get("/analysis/summary", response_model=AnalysisSummaryResponse)
async def analysis_summary():
    """Aggregated metrics from the latest pipeline run for analysis views."""
    snapshot = get_latest_run_snapshot()
    if snapshot["status"] == "empty":
        return AnalysisSummaryResponse(
            status="no_results",
            message="No pipeline results found.",
        )
    json_blob = snapshot.get("latest_json") or {}
    ranking = json_blob.get("ranking") or []
    best = ranking[0] if ranking else {}
    return AnalysisSummaryResponse(
        status="ok",
        best_model=best.get("name"),
        best_pr_auc=best.get("pr_auc"),
        model_count=len(ranking),
        classical_count=sum(1 for r in ranking if r.get("type") == "classical"),
        quantum_count=sum(1 for r in ranking if r.get("type") == "quantum"),
        ensemble_count=sum(1 for r in ranking if r.get("type") == "ensemble"),
        ranking=ranking,
        relation=json_blob.get("relation"),
        run_timestamp=json_blob.get("timestamp"),
    )


ALLOWED_EXPORT_EXTENSIONS = {".json", ".csv", ".txt", ".log"}


@app.get("/exports", response_model=ExportListResponse)
async def list_exports():
    """List downloadable artifacts under ``results/`` (allowlisted extensions only)."""
    from utils.latest_run import get_results_dir

    rd = get_results_dir()
    files: List[ExportFileInfo] = []
    if rd.exists():
        for p in sorted(rd.iterdir()):
            if p.is_file() and p.suffix in ALLOWED_EXPORT_EXTENSIONS:
                try:
                    st = p.stat()
                    files.append(ExportFileInfo(
                        name=p.name,
                        size_bytes=st.st_size,
                        modified=st.st_mtime,
                    ))
                except OSError:
                    pass
    return ExportListResponse(status="ok", files=files)


@app.get("/exports/{filename}")
async def download_export(filename: str):
    """Download a single results file (path traversal protected)."""
    from fastapi.responses import FileResponse
    from utils.latest_run import get_results_dir
    import re

    if not re.match(r'^[\w\-. ]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    rd = get_results_dir()
    path = (rd / filename).resolve()
    if not str(path).startswith(str(rd.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if path.suffix not in ALLOWED_EXPORT_EXTENSIONS:
        raise HTTPException(status_code=403, detail="File type not allowed")
    return FileResponse(path, filename=filename)


@app.post("/jobs/pipeline", response_model=JobResponse)
async def create_pipeline_job(request: JobCreateRequest):
    """
    Start ``run_optimized_pipeline.py`` as a background job.
    Returns immediately with a job id that can be polled via ``GET /jobs/{id}``.
    """
    flags = request.model_dump()
    job = job_manager.create(flags)
    return JobResponse(**job.to_dict())


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Poll a pipeline job for status and log output."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**job.to_dict())


@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs():
    """List all pipeline jobs (most recent first)."""
    jobs = sorted(job_manager.list_jobs(), key=lambda j: j.created_at, reverse=True)
    return [JobResponse(**j.to_dict()) for j in jobs]


# ---------------------------------------------------------------------------
# Visualization endpoints (/viz/*)
# ---------------------------------------------------------------------------


class VizAtom(BaseModel):
    element: str
    x: float
    y: float
    z: float


class VizBond(BaseModel):
    source: int  # atom index
    target: int
    order: int  # 1=single, 2=double, 3=triple


class VizMoleculeResponse(BaseModel):
    status: str
    compound_id: Optional[str] = None
    compound_name: Optional[str] = None
    atoms: List[VizAtom] = []
    bonds: List[VizBond] = []
    message: Optional[str] = None


class VizEmbNode(BaseModel):
    id: str
    label: str
    type: str  # "compound" or "disease"
    x: float
    y: float
    z: float


class VizEmbEdge(BaseModel):
    source: str
    target: str


class VizEmbeddingsResponse(BaseModel):
    status: str
    nodes: List[VizEmbNode] = []
    edges: List[VizEmbEdge] = []
    model_name: Optional[str] = None
    available_models: List[str] = []
    compound_count: int = 0
    disease_count: int = 0
    edge_count: int = 0
    variance_explained: Optional[List[float]] = None
    projection: Optional[str] = None
    projection_note: Optional[str] = None
    message: Optional[str] = None


class VizKGNode(BaseModel):
    id: str
    label: str
    entity_type: str
    score: Optional[float] = None


class VizKGLink(BaseModel):
    source: str
    target: str
    relation: str


class VizKGSubgraphResponse(BaseModel):
    status: str
    nodes: List[VizKGNode] = []
    links: List[VizKGLink] = []
    center_entity: Optional[str] = None
    message: Optional[str] = None


class VizPrediction(BaseModel):
    compound_id: str
    compound_name: str
    disease_id: str
    disease_name: str
    score: float
    confidence: str  # "High" / "Medium" / "Low"
    model_used: str


class VizPredictionsResponse(BaseModel):
    status: str
    predictions: List[VizPrediction] = []
    message: Optional[str] = None


class VizModelMetric(BaseModel):
    name: str
    type: str  # "classical" / "quantum" / "ensemble"
    pr_auc: float
    accuracy: float
    fit_time: float


class VizModelMetricsResponse(BaseModel):
    status: str
    models: List[VizModelMetric] = []
    ablation: Optional[Dict[str, float]] = None
    relation: Optional[str] = None
    run_timestamp: Optional[str] = None
    message: Optional[str] = None


class VizCircuitResponse(BaseModel):
    status: str
    n_qubits: int = 6
    n_reps: int = 2
    entanglement: str = "full"
    feature_map: str = "Pauli"
    execution_mode: Optional[str] = None
    backend: Optional[str] = None
    shots: Optional[int] = None


class VizRunPrediction(BaseModel):
    compound_id: str
    compound_name: str
    disease_id: str
    disease_name: str
    y_true: int
    score_classical: Optional[float] = None
    score_quantum: Optional[float] = None
    score: float  # best available score
    confidence: str  # "High" / "Medium" / "Low"


class VizRunPredictionsResponse(BaseModel):
    status: str
    predictions: List[VizRunPrediction] = []
    run_timestamp: Optional[str] = None
    source_file: Optional[str] = None
    available_runs: List[str] = []
    message: Optional[str] = None


# ---- Cached embedding viz (keyed by "model:projection") ----
_emb_cache: Dict[str, VizEmbeddingsResponse] = {}

# Known embedding model prefixes
_EMBEDDING_MODELS = [
    "rotate_128d", "rotate_256d", "rotate_512d",
    "complex_128d", "complex_256d",
]


def _list_available_embedding_models() -> List[str]:
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data")
    available = []
    for prefix in _EMBEDDING_MODELS:
        emb_file = os.path.join(data_dir, f"{prefix}_entity_embeddings.npy")
        id_file = os.path.join(data_dir, f"{prefix}_entity_ids.json")
        if os.path.exists(emb_file) and os.path.exists(id_file):
            available.append(prefix)
    # Also check default (no prefix)
    if os.path.exists(os.path.join(data_dir, "entity_embeddings.npy")) and \
       os.path.exists(os.path.join(data_dir, "entity_ids.json")):
        available.append("default")
    return available


def _coords_to_box(coords: np.ndarray, per_axis: bool) -> np.ndarray:
    """Map coords to ~[-5, 5]. per_axis=True stretches each dimension (fills the view)."""
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    out = np.array(coords, dtype=np.float64, copy=True)
    if per_axis:
        for i in range(3):
            col = out[:, i]
            mx = np.abs(col).max() or 1.0
            out[:, i] = col / mx * 5.0
    else:
        mx = np.abs(out).max() or 1.0
        out = out / mx * 5.0
    return out


def _compute_embedding_viz(
    model: Optional[str] = None,
    projection: str = "pca_stretch",
) -> VizEmbeddingsResponse:
    """3D projection of entity embeddings: PCA, variance-stretched PCA, or t-SNE."""
    allowed = {"pca", "pca_stretch", "tsne"}
    if projection not in allowed:
        projection = "pca_stretch"

    available = _list_available_embedding_models()
    if not available:
        return VizEmbeddingsResponse(status="error", message="No embeddings found",
                                     available_models=[])

    chosen = model if model and model in available else available[0]
    cache_key = f"{chosen}:{projection}"
    if cache_key in _emb_cache:
        return _emb_cache[cache_key]

    import json
    from sklearn.decomposition import PCA
    import pandas as pd

    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data")

    if chosen == "default":
        emb_file = os.path.join(data_dir, "entity_embeddings.npy")
        id_file = os.path.join(data_dir, "entity_ids.json")
    else:
        emb_file = os.path.join(data_dir, f"{chosen}_entity_embeddings.npy")
        id_file = os.path.join(data_dir, f"{chosen}_entity_ids.json")

    embs = np.load(emb_file)
    with open(id_file) as f:
        entity_ids = json.load(f)

    nodes_tsv = os.path.join(data_dir, "hetionet-v1.0-nodes.tsv")
    name_map: Dict[str, str] = {}
    if os.path.exists(nodes_tsv):
        ndf = pd.read_csv(nodes_tsv, sep="\t")
        name_map = dict(zip(ndf["id"], ndf["name"]))

    indices, filtered_ids = [], []
    for i, eid in enumerate(entity_ids):
        if eid.startswith("Compound::") or eid.startswith("Disease::"):
            indices.append(i)
            filtered_ids.append(eid)
    if not indices:
        return VizEmbeddingsResponse(status="error", message="No compound/disease embeddings",
                                     available_models=available)

    sub_embs = embs[indices]
    n_samples, n_feat = sub_embs.shape

    variance: Optional[List[float]] = None
    proj_note: Optional[str] = None
    coords: np.ndarray

    if projection == "tsne":
        from sklearn.manifold import TSNE

        n_pre = max(2, min(50, n_feat, n_samples - 1))
        pre = PCA(n_components=n_pre, random_state=42).fit_transform(sub_embs)
        perp = int(min(30, max(5, (n_samples - 1) // 4)))
        try:
            coords = TSNE(
                n_components=3,
                perplexity=perp,
                random_state=42,
                max_iter=1000,
                init="pca",
            ).fit_transform(pre)
        except Exception as exc:
            logger.warning("t-SNE failed (%s); falling back to stretched PCA", exc)
            projection = "pca_stretch"
            n_components = min(3, n_samples, n_feat)
            pca = PCA(n_components=n_components)
            coords = pca.fit_transform(sub_embs)
            variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
            coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)
            coords = _coords_to_box(coords, per_axis=True)
            proj_note = "Stretched PCA (t-SNE unavailable)"
        else:
            coords = _coords_to_box(coords, per_axis=True)
            variance = None
            proj_note = (
                "t-SNE emphasizes local neighborhoods; distances between clusters are not meaningful."
            )
    else:
        n_components = min(3, n_samples, n_feat)
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(sub_embs)
        variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        if projection == "pca_stretch":
            # Z-score each PC axis then scale independently → fills 3D view when PC1–3 explain little variance.
            coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)
            coords = _coords_to_box(coords, per_axis=True)
            proj_note = (
                "Each PCA axis scaled separately so the view uses the full depth "
                "(does not change relative ordering along each axis)."
            )
        else:
            coords = _coords_to_box(coords, per_axis=False)
            proj_note = (
                "Raw PCA into 3D; when variance per axis is low, points look like one tight cloud."
            )

    nodes: List[VizEmbNode] = []
    n_compounds = 0
    n_diseases = 0
    for idx, eid in enumerate(filtered_ids):
        etype = "compound" if eid.startswith("Compound::") else "disease"
        if etype == "compound":
            n_compounds += 1
        else:
            n_diseases += 1
        label = name_map.get(eid, eid.split("::")[-1])
        nodes.append(VizEmbNode(
            id=eid, label=label, type=etype,
            x=round(float(coords[idx, 0]), 3),
            y=round(float(coords[idx, 1]), 3),
            z=round(float(coords[idx, 2]), 3),
        ))

    edges: List[VizEmbEdge] = []
    embedded_set = set(filtered_ids)
    edges_sif = os.path.join(data_dir, "hetionet-v1.0-edges.sif")
    if os.path.exists(edges_sif):
        edf = pd.read_csv(edges_sif, sep="\t")
        ctd = edf[edf["metaedge"] == "CtD"]
        for _, row in ctd.iterrows():
            s, t = row["source"], row["target"]
            if s in embedded_set and t in embedded_set:
                edges.append(VizEmbEdge(source=s, target=t))

    result = VizEmbeddingsResponse(
        status="ok", nodes=nodes, edges=edges,
        model_name=chosen, available_models=available,
        compound_count=n_compounds, disease_count=n_diseases,
        edge_count=len(edges), variance_explained=variance,
        projection=projection,
        projection_note=proj_note,
    )
    _emb_cache[cache_key] = result
    return result


@app.get("/viz/embeddings", response_model=VizEmbeddingsResponse)
async def viz_embeddings(
    model: Optional[str] = Query(None, description="Embedding model (e.g. rotate_128d, rotate_512d, complex_256d)"),
    projection: str = Query(
        "pca_stretch",
        description="pca = raw 3D PCA; pca_stretch = per-axis scale (clearer layout); tsne = nonlinear 3D (slower, cached)",
    ),
):
    """Projected entity embeddings for 3D scatter (PCA, stretched PCA, or t-SNE)."""
    try:
        return _compute_embedding_viz(model, projection=projection)
    except Exception as e:
        logger.error(f"Embedding PCA failed: {e}")
        return VizEmbeddingsResponse(status="error", message=str(e))


# ---- Cached Hetionet dataframes (loaded once, shared across endpoints) ----
_het_edf: Optional[Any] = None  # pd.DataFrame
_het_name_map: Optional[Dict[str, str]] = None
_het_kind_map: Optional[Dict[str, str]] = None


def _load_hetionet():
    """Load and cache Hetionet nodes + edges. Returns (edf, name_map, kind_map)."""
    global _het_edf, _het_name_map, _het_kind_map
    if _het_edf is not None:
        return _het_edf, _het_name_map, _het_kind_map

    import pandas as pd
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data")

    nodes_tsv = os.path.join(data_dir, "hetionet-v1.0-nodes.tsv")
    edges_sif = os.path.join(data_dir, "hetionet-v1.0-edges.sif")

    name_map: Dict[str, str] = {}
    kind_map: Dict[str, str] = {}
    if os.path.exists(nodes_tsv):
        ndf = pd.read_csv(nodes_tsv, sep="\t")
        name_map = dict(zip(ndf["id"], ndf["name"]))
        kind_map = dict(zip(ndf["id"], ndf["kind"]))

    # Auto-download edges if missing (uses kg_loader's multi-mirror logic)
    if not os.path.exists(edges_sif):
        try:
            from kg_layer.kg_loader import download_hetionet_if_missing
            edges_sif = download_hetionet_if_missing(data_dir)
        except Exception as _dl_err:
            logger.warning(f"Could not auto-download Hetionet edges: {_dl_err}")

    edf = None
    if os.path.exists(edges_sif):
        edf = pd.read_csv(edges_sif, sep="\t")

    _het_edf = edf
    _het_name_map = name_map
    _het_kind_map = kind_map
    return edf, name_map, kind_map


@app.get("/viz/kg-search")
async def viz_kg_search(
    q: str = Query(..., description="Search term (name fragment)"),
    limit: int = Query(20, description="Max results"),
):
    """Search Hetionet entities by name for autocomplete."""
    _, name_map, kind_map = _load_hetionet()
    if not name_map:
        return {"results": []}

    term = q.lower()
    results = []
    for eid, name in name_map.items():
        if term in name.lower() or term in eid.lower():
            results.append({
                "id": eid,
                "name": name,
                "kind": kind_map.get(eid, eid.split("::")[0]),
            })
            if len(results) >= limit:
                break
    return {"results": results}


@app.get("/viz/kg-subgraph", response_model=VizKGSubgraphResponse)
async def viz_kg_subgraph(
    entity: str = Query(..., description="Center entity ID (e.g. Compound::DB00635)"),
    max_nodes: int = Query(120, description="Max nodes in subgraph"),
    hops: int = Query(1, description="Number of hops from center (1-3)", ge=1, le=3),
    relation_filter: Optional[str] = Query(None, description="Comma-separated relation types to include (e.g. CtD,CbG,DaG)"),
):
    """Extract a multi-hop subgraph from Hetionet centered on a given entity."""
    import pandas as pd
    edf, name_map, kind_map = _load_hetionet()
    if edf is None:
        return VizKGSubgraphResponse(status="error", message="Edge file not found")

    # Optional relation filter
    allowed_rels = None
    if relation_filter:
        allowed_rels = set(r.strip() for r in relation_filter.split(","))

    working_edf = edf
    if allowed_rels:
        working_edf = edf[edf["metaedge"].isin(allowed_rels)]

    # Biologically relevant edge types (for priority sorting)
    priority = ["CtD", "CbG", "CpD", "CuG", "CdG", "DaG", "DuG", "DdG", "DpS", "DlA", "DrD"]
    priority_set = set(priority)

    # Multi-hop BFS
    frontier = {entity}
    all_entity_ids = {entity}
    all_links: List[VizKGLink] = []
    seen_edges: set = set()

    for hop in range(hops):
        if not frontier:
            break
        # Find edges touching the frontier
        mask = working_edf["source"].isin(frontier) | working_edf["target"].isin(frontier)
        hop_edges = working_edf[mask]

        # Priority sort
        hop_priority = hop_edges[hop_edges["metaedge"].isin(priority_set)]
        hop_other = hop_edges[~hop_edges["metaedge"].isin(priority_set)]
        hop_sorted = pd.concat([hop_priority, hop_other])

        next_frontier: set = set()
        for _, row in hop_sorted.iterrows():
            s, t, rel = row["source"], row["target"], row["metaedge"]
            edge_key = (s, t, rel)
            if edge_key in seen_edges:
                continue

            # Check if adding new nodes would exceed limit
            new_nodes = set()
            if s not in all_entity_ids:
                new_nodes.add(s)
            if t not in all_entity_ids:
                new_nodes.add(t)
            if len(all_entity_ids) + len(new_nodes) > max_nodes:
                continue

            seen_edges.add(edge_key)
            all_links.append(VizKGLink(source=s, target=t, relation=rel))
            for nid in new_nodes:
                all_entity_ids.add(nid)
                next_frontier.add(nid)

        frontier = next_frontier

    nodes: List[VizKGNode] = []
    for nid in all_entity_ids:
        kind = kind_map.get(nid, nid.split("::")[0]) if kind_map else nid.split("::")[0]
        label = name_map.get(nid, nid.split("::")[-1]) if name_map else nid.split("::")[-1]
        nodes.append(VizKGNode(id=nid, label=label, entity_type=kind))

    return VizKGSubgraphResponse(
        status="ok",
        nodes=nodes,
        links=all_links,
        center_entity=entity,
    )


@app.get("/viz/predictions", response_model=VizPredictionsResponse)
async def viz_predictions(top_k: int = Query(30, description="Number of predictions")):
    """
    Top compound-disease predictions from known CtD edges scored by the loaded model.
    Returns a ranked table of predictions for the dashboard.
    """
    if orchestrator is None:
        return VizPredictionsResponse(status="error", message="Orchestrator not ready")

    import pandas as pd

    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data")

    nodes_tsv = os.path.join(data_dir, "hetionet-v1.0-nodes.tsv")
    edges_sif = os.path.join(data_dir, "hetionet-v1.0-edges.sif")
    name_map: Dict[str, str] = {}
    if os.path.exists(nodes_tsv):
        ndf = pd.read_csv(nodes_tsv, sep="\t")
        name_map = dict(zip(ndf["id"], ndf["name"]))

    # Collect known CtD pairs
    pairs: List[tuple] = []
    if os.path.exists(edges_sif):
        edf = pd.read_csv(edges_sif, sep="\t")
        ctd = edf[edf["metaedge"] == "CtD"]
        embedded = set(orchestrator.embedder.entity_to_id.keys()) if orchestrator.embedder else set()
        for _, row in ctd.iterrows():
            if row["source"] in embedded and row["target"] in embedded:
                pairs.append((row["source"], row["target"]))

    # Score a sample (cap to avoid long waits)
    import random
    sample = pairs[:top_k * 3] if len(pairs) <= top_k * 3 else random.sample(pairs, top_k * 3)
    scored: List[VizPrediction] = []
    for drug_id, disease_id in sample:
        try:
            result = orchestrator.predict_link_probability(drug=drug_id, disease=disease_id)
            if result["status"] != "success":
                continue
            prob = result["link_probability"]
            conf = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
            scored.append(VizPrediction(
                compound_id=drug_id,
                compound_name=name_map.get(drug_id, drug_id.split("::")[-1]),
                disease_id=disease_id,
                disease_name=name_map.get(disease_id, disease_id.split("::")[-1]),
                score=round(prob, 4),
                confidence=conf,
                model_used=result["model_used"],
            ))
        except Exception:
            continue

    scored.sort(key=lambda p: p.score, reverse=True)
    return VizPredictionsResponse(status="ok", predictions=scored[:top_k])


@app.get("/viz/model-metrics", response_model=VizModelMetricsResponse)
async def viz_model_metrics():
    """Model comparison metrics from the latest pipeline run."""
    snapshot = get_latest_run_snapshot()
    if snapshot["status"] == "empty":
        return VizModelMetricsResponse(status="no_results", message="No pipeline results found")

    json_blob = snapshot.get("latest_json") or {}
    ranking = json_blob.get("ranking") or []

    models = [
        VizModelMetric(
            name=r["name"],
            type=r.get("type", "classical"),
            pr_auc=r.get("pr_auc", 0),
            accuracy=r.get("accuracy", 0),
            fit_time=r.get("fit_time", 0),
        )
        for r in ranking
        if isinstance(r, dict) and r.get("name")
    ]

    # Compute ablation-style comparison: drop from best PR-AUC
    ablation: Dict[str, float] = {}
    if models:
        best = max(m.pr_auc for m in models)
        ablation["Full ensemble"] = round(best, 4)
        classical_only = [m for m in models if m.type == "classical"]
        quantum_only = [m for m in models if m.type == "quantum"]
        if classical_only:
            ablation["Best classical only"] = round(max(m.pr_auc for m in classical_only), 4)
        if quantum_only:
            ablation["Best quantum only"] = round(max(m.pr_auc for m in quantum_only), 4)

    return VizModelMetricsResponse(
        status="ok",
        models=models,
        ablation=ablation,
        relation=json_blob.get("relation"),
        run_timestamp=json_blob.get("timestamp"),
    )


@app.get("/viz/circuit-params", response_model=VizCircuitResponse)
async def viz_circuit_params():
    """Quantum circuit parameters for the circuit diagram."""
    import yaml

    base = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base, "config", "quantum_config.yaml")
    config_data: Dict[str, Any] = {}
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}
    except Exception:
        pass

    q = config_data.get("quantum", {})
    # Also read latest pipeline results for actual qml_dim / feature_map_reps
    snapshot = get_latest_run_snapshot()
    cfg = {}
    if snapshot.get("latest_json"):
        cfg = snapshot["latest_json"].get("config", {})

    n_qubits = cfg.get("qml_dim", 12)
    n_reps = cfg.get("qml_feature_map_reps", 2)
    feature_map = cfg.get("qml_feature_map", "Pauli")
    entanglement = cfg.get("qml_entanglement", "full")

    execution_mode = q.get("execution_mode")
    backend = None
    shots = None
    if execution_mode == "heron":
        backend = q.get("heron", {}).get("backend")
        shots = q.get("heron", {}).get("shots")
    elif execution_mode == "simulator":
        shots = q.get("simulator", {}).get("shots")

    return VizCircuitResponse(
        status="ok",
        n_qubits=n_qubits,
        n_reps=n_reps,
        entanglement=entanglement,
        feature_map=feature_map,
        execution_mode=execution_mode,
        backend=backend,
        shots=shots,
    )


# ---- Molecule 3D via PubChem ----
_mol_cache: Dict[str, VizMoleculeResponse] = {}

# PubChem element number → symbol
_ELEMENT_MAP = {
    1: "H", 2: "He", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 12: "Mg", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K",
    20: "Ca", 26: "Fe", 29: "Cu", 30: "Zn", 35: "Br", 53: "I",
}


@app.get("/viz/molecule", response_model=VizMoleculeResponse)
async def viz_molecule(
    compound: str = Query(..., description="Compound name or Hetionet ID (e.g. Cyclosporine or Compound::DB00091)"),
):
    """Fetch 3D molecular structure from PubChem for a given compound."""
    import urllib.request
    import json as _json
    import pandas as pd

    # Resolve name
    name = compound
    if compound.startswith("Compound::"):
        base = os.path.dirname(os.path.dirname(__file__))
        nodes_tsv = os.path.join(base, "data", "hetionet-v1.0-nodes.tsv")
        if os.path.exists(nodes_tsv):
            ndf = pd.read_csv(nodes_tsv, sep="\t")
            row = ndf[ndf["id"] == compound]
            if not row.empty:
                name = row.iloc[0]["name"]

    cache_key = name.lower().strip()
    if cache_key in _mol_cache:
        return _mol_cache[cache_key]

    # Fetch from PubChem
    try:
        encoded = urllib.request.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/JSON?record_type=3d"
        req = urllib.request.Request(url, headers={"User-Agent": "HybridQMLKG/0.1"})
        raw = urllib.request.urlopen(req, timeout=15).read()
        data = _json.loads(raw)
    except Exception as exc:
        return VizMoleculeResponse(
            status="error",
            compound_name=name,
            message=f"PubChem lookup failed for '{name}': {exc}",
        )

    try:
        pc = data["PC_Compounds"][0]
        elements = pc["atoms"]["element"]
        conf = pc["coords"][0]["conformers"][0]
        xs, ys, zs = conf["x"], conf["y"], conf["z"]

        atoms = [
            VizAtom(
                element=_ELEMENT_MAP.get(elements[i], "X"),
                x=round(xs[i], 4),
                y=round(ys[i], 4),
                z=round(zs[i], 4),
            )
            for i in range(len(elements))
        ]

        bonds_data = pc.get("bonds", {})
        aid1 = bonds_data.get("aid1", [])
        aid2 = bonds_data.get("aid2", [])
        orders = bonds_data.get("order", [1] * len(aid1))
        bonds = [
            VizBond(source=aid1[i] - 1, target=aid2[i] - 1, order=orders[i])
            for i in range(len(aid1))
        ]

        resp = VizMoleculeResponse(
            status="ok",
            compound_id=compound,
            compound_name=name,
            atoms=atoms,
            bonds=bonds,
        )
        _mol_cache[cache_key] = resp
        return resp
    except Exception as exc:
        return VizMoleculeResponse(
            status="error",
            compound_name=name,
            message=f"Failed to parse PubChem response: {exc}",
        )


# ---- Run predictions (from pipeline CSV output) ----


def _list_available_runs() -> List[str]:
    """Return timestamps of available optimized_results files."""
    from utils.latest_run import get_results_dir
    import re
    rd = get_results_dir()
    timestamps = []
    if rd.exists():
        for p in sorted(rd.iterdir(), reverse=True):
            m = re.match(r"optimized_results_(\d{8}-\d{6})\.json", p.name)
            if m:
                timestamps.append(m.group(1))
    return timestamps


def _load_name_map() -> Dict[str, str]:
    """Load Hetionet node ID → human name mapping."""
    import pandas as pd
    base = os.path.dirname(os.path.dirname(__file__))
    nodes_tsv = os.path.join(base, "data", "hetionet-v1.0-nodes.tsv")
    if os.path.exists(nodes_tsv):
        ndf = pd.read_csv(nodes_tsv, sep="\t")
        return dict(zip(ndf["id"], ndf["name"]))
    return {}


@app.get("/viz/run-predictions", response_model=VizRunPredictionsResponse)
async def viz_run_predictions(
    run: Optional[str] = Query(None, description="Run timestamp (e.g. 20260325-223245). Default: latest."),
    top_k: int = Query(50, description="Number of top predictions to return"),
):
    """
    Compound-disease predictions from a pipeline run's CSV output.

    Reads ``predictions_compare.csv`` (classical + quantum scores) or
    ``predictions_latest.csv`` from ``results/``. Resolves entity IDs to
    human-readable names via Hetionet nodes.
    """
    import pandas as pd
    import json as _json
    from utils.latest_run import get_results_dir

    base = os.path.dirname(os.path.dirname(__file__))
    rd = get_results_dir()
    available = _list_available_runs()

    if not rd.exists():
        return VizRunPredictionsResponse(
            status="error", message="Results directory not found", available_runs=available,
        )

    name_map = _load_name_map()

    # Load entity ID list for resolving numeric target_id
    entity_ids: List[str] = []
    for prefix in ("rotate_128d", "rotate_256d", "complex_128d", ""):
        id_file = os.path.join(base, "data", f"{prefix}_entity_ids.json" if prefix else "entity_ids.json")
        if os.path.exists(id_file):
            with open(id_file) as f:
                candidate = _json.load(f)
            if len(candidate) > len(entity_ids):
                entity_ids = candidate

    # Determine which files to read
    run_ts = run or (available[0] if available else None)
    source_file: Optional[str] = None

    # Try predictions_compare first (has both classical + quantum scores)
    compare_path = rd / "predictions_compare.csv"
    latest_path = rd / "predictions_latest.csv"
    # If a specific run is requested, look for timestamped files
    if run_ts:
        ts_compare = rd / f"predictions_compare_{run_ts}.csv"
        ts_qsvc = rd / f"predictions_QSVC_{run_ts}.csv"
        if ts_compare.exists():
            compare_path = ts_compare
        elif ts_qsvc.exists():
            latest_path = ts_qsvc

    df: Optional[pd.DataFrame] = None

    if compare_path.exists():
        df = pd.read_csv(compare_path)
        source_file = compare_path.name
        # Schema: source, target, y_true, y_score_classical, y_score_quantum
        if "y_score_classical" not in df.columns:
            df = None  # wrong format, fall through

    if df is None and latest_path.exists():
        df = pd.read_csv(latest_path)
        source_file = latest_path.name
        # Schema: split, source_id, target_id, source, target, label, y_true, y_pred, y_score

    if df is None:
        return VizRunPredictionsResponse(
            status="error", message="No prediction CSV found in results/",
            available_runs=available,
        )

    predictions: List[VizRunPrediction] = []

    if "y_score_classical" in df.columns:
        # predictions_compare.csv format
        df = df.sort_values("y_score_classical", ascending=False)
        for _, row in df.head(top_k).iterrows():
            comp_id = str(row.get("source", ""))
            dis_id = str(row.get("target", ""))
            sc = float(row.get("y_score_classical", 0))
            sq = float(row.get("y_score_quantum", 0))
            best = max(sc, sq)
            conf = "High" if best > 0.7 else ("Medium" if best > 0.4 else "Low")
            predictions.append(VizRunPrediction(
                compound_id=comp_id,
                compound_name=name_map.get(comp_id, comp_id.split("::")[-1] if "::" in comp_id else comp_id),
                disease_id=dis_id,
                disease_name=name_map.get(dis_id, dis_id.split("::")[-1] if "::" in dis_id else dis_id),
                y_true=int(row.get("y_true", 0)),
                score_classical=round(sc, 6),
                score_quantum=round(sq, 6),
                score=round(best, 6),
                confidence=conf,
            ))
    else:
        # predictions_latest.csv format
        # Resolve NaN targets via entity_ids + target_id
        def resolve_target(row):
            t = row.get("target")
            if pd.notna(t) and str(t).strip():
                return str(t)
            tid = row.get("target_id")
            if pd.notna(tid):
                tid_int = int(tid)
                if 0 <= tid_int < len(entity_ids):
                    return entity_ids[tid_int]
            return ""

        df["target_resolved"] = df.apply(resolve_target, axis=1)

        # Prefer test split, fallback to all
        test = df[df.get("split", pd.Series()) == "test"] if "split" in df.columns else df
        if test.empty:
            test = df
        test = test.sort_values("y_score", ascending=False)

        for _, row in test.head(top_k).iterrows():
            comp_id = str(row.get("source", ""))
            dis_id = str(row.get("target_resolved", ""))
            sc = float(row.get("y_score", 0))
            conf = "High" if sc > 0.7 else ("Medium" if sc > 0.4 else "Low")
            predictions.append(VizRunPrediction(
                compound_id=comp_id,
                compound_name=name_map.get(comp_id, comp_id.split("::")[-1] if "::" in comp_id else comp_id),
                disease_id=dis_id,
                disease_name=name_map.get(dis_id, dis_id.split("::")[-1] if "::" in dis_id else dis_id),
                y_true=int(row.get("y_true", 0)),
                score_classical=round(sc, 6) if "y_score_classical" not in df.columns else None,
                score_quantum=None,
                score=round(sc, 6),
                confidence=conf,
            ))

    return VizRunPredictionsResponse(
        status="ok",
        predictions=predictions,
        run_timestamp=run_ts,
        source_file=source_file,
        available_runs=available,
    )


# Example usage (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)