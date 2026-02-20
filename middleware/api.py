# middleware/api.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
from .orchestrator import LinkPredictionOrchestrator

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


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "Hybrid QML-KG API. Visit /docs for interactive documentation."}


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


# Example usage (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)