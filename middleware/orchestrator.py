# middleware/orchestrator.py

import os
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from kg_layer.kg_embedder import HetionetEmbedder
from kg_layer.kg_loader import load_hetionet_edges
from middleware.ranked_mechanisms import rank_mechanism_candidates as _rank_mechanism_candidates
from classical_baseline.train_baseline import ClassicalLinkPredictor
from quantum_layer.qml_model import QMLLinkPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Brand / common name → Hetionet ID.
# Hetionet uses INN/WHO systematic names; these aliases bridge the most common
# brand names that users are likely to type. The metadata CSV overrides these
# for any key it also covers (via setdefault), so they are safe fallbacks.
COMMON_NAME_ALIASES: Dict[str, str] = {
    # Compounds — brand / common names
    "aspirin":              "Compound::DB00945",  # acetylsalicylic acid
    "tylenol":              "Compound::DB00316",  # acetaminophen / paracetamol
    "acetaminophen":        "Compound::DB00316",
    "paracetamol":          "Compound::DB00316",
    "advil":                "Compound::DB01050",  # ibuprofen
    "motrin":               "Compound::DB01050",
    "ibuprofen":            "Compound::DB01050",
    "metformin":            "Compound::DB00331",
    "glucophage":           "Compound::DB00331",
    "atorvastatin":         "Compound::DB01076",
    "lipitor":              "Compound::DB01076",
    "dexamethasone":        "Compound::DB01234",
    "prednisone":           "Compound::DB00635",
    "warfarin":             "Compound::DB00682",
    "coumadin":             "Compound::DB00682",
    "lisinopril":           "Compound::DB00722",
    "metoprolol":           "Compound::DB00264",
    "amlodipine":           "Compound::DB00381",
    "omeprazole":           "Compound::DB00338",
    "prilosec":             "Compound::DB00338",
    "simvastatin":          "Compound::DB00641",
    "zocor":                "Compound::DB00641",
    "losartan":             "Compound::DB00678",
    "cozaar":               "Compound::DB00678",
    "levothyroxine":        "Compound::DB00451",
    "synthroid":            "Compound::DB00451",
    "albuterol":            "Compound::DB01001",
    "salbutamol":           "Compound::DB01001",
    "amoxicillin":          "Compound::DB01060",
    "ciprofloxacin":        "Compound::DB00537",
    "pindolol":             "Compound::DB00960",
    "gabapentin":           "Compound::DB00996",
    "neurontin":            "Compound::DB00996",
    "sertraline":           "Compound::DB01104",
    "zoloft":               "Compound::DB01104",
    "fluoxetine":           "Compound::DB00472",
    "prozac":               "Compound::DB00472",
    "celecoxib":            "Compound::DB00482",
    "celebrex":             "Compound::DB00482",
    "montelukast":          "Compound::DB00471",
    "singulair":            "Compound::DB00471",
    # Diseases — common names Hetionet may use a longer form for
    "diabetes":             "Disease::DOID:9351",   # type 1 diabetes mellitus
    "type 2 diabetes":      "Disease::DOID:9352",
    "hypertension":         "Disease::DOID:10763",
    "high blood pressure":  "Disease::DOID:10763",
    "cancer":               "Disease::DOID:162",
    "breast cancer":        "Disease::DOID:1612",
    "lung cancer":          "Disease::DOID:1324",
    "asthma":               "Disease::DOID:2841",
    "alzheimer":            "Disease::DOID:10652",
    "alzheimer's disease":  "Disease::DOID:10652",
    "alzheimers disease":   "Disease::DOID:10652",
    "parkinson":            "Disease::DOID:14330",
    "parkinson's disease":  "Disease::DOID:14330",
    "parkinsons disease":   "Disease::DOID:14330",
    "depression":           "Disease::DOID:1596",
    "rheumatoid arthritis": "Disease::DOID:7148",
    "multiple sclerosis":   "Disease::DOID:2377",
    "lupus":                "Disease::DOID:9074",
    "crohn":                "Disease::DOID:8778",
    "crohn's disease":      "Disease::DOID:8778",
    "heart failure":        "Disease::DOID:6000",
    "schizophrenia":        "Disease::DOID:5419",
}


class LinkPredictionOrchestrator:
    """
    Orchestrates end-to-end link prediction for biomedical knowledge graphs.
    
    Handles:
      - Entity name resolution (e.g., "Aspirin" -> Compound::DB00945)
      - Embedding retrieval
      - Classical or quantum prediction
      - Fallback to classical if quantum fails
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        results_dir: str = "results",
        use_quantum: bool = True,
        qml_config_path: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.use_quantum = use_quantum
        
        # Load entity mappings
        self._load_entity_mappings()

        # Load classical model first so we can infer the expected feature/qml dim
        self.classical_predictor = ClassicalLinkPredictor()
        if not self.classical_predictor.load_model():
            logger.warning("Classical model not found — orchestrator running in degraded mode.")
            self.classical_predictor = None

        # Infer qml_dim from trained model (4-feature scheme: [h, t, |h-t|, h*t])
        qml_dim = 12  # default from last training run
        if self.classical_predictor and self.classical_predictor.model:
            try:
                qml_dim = self.classical_predictor.model.n_features_in_ // 4
            except Exception:
                pass

        # Initialize embedder with matching qml_dim and reduce dimensions
        self.embedder = HetionetEmbedder(qml_dim=qml_dim)
        if not self.embedder.load_saved_embeddings():
            logger.warning("Embeddings not found — orchestrator running in degraded mode.")
            self.embedder = None
        else:
            # Sync embedding_dim to the actual loaded shape so _deterministic_vec
            # produces vectors of the right size for PCA transform.
            self.embedder.embedding_dim = int(self.embedder.entity_embeddings.shape[1])
            self.embedder.reduce_to_qml_dim()
        
        # Load quantum model (if requested)
        self.quantum_predictor = None
        if self.use_quantum:
            try:
                self._load_quantum_model(qml_config_path)
            except Exception as e:
                logger.warning(f"Failed to load quantum model: {e}. Falling back to classical only.")
                self.use_quantum = False
    
    def _load_entity_mappings(self) -> None:
        """Load entity name to Hetionet ID mappings.

        Reads data/hetionet_nodes_metadata.csv (node_id, name, namespace, external_url).
        Indexes on three keys per node so users can type any of:
          - full ID:          "Compound::DB00945"
          - ID suffix:        "DB00945"
          - human-readable:   "aspirin"  (case-insensitive)
        """
        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}

        # Seed with brand / common-name aliases so users can type "aspirin", "ibuprofen", etc.
        # The metadata CSV uses setdefault below, so CSV entries will not override these.
        self.name_to_id.update(COMMON_NAME_ALIASES)

        map_path = os.path.join(self.data_dir, "hetionet_nodes_metadata.csv")
        if not os.path.exists(map_path):
            logger.warning(f"Node metadata not found at {map_path}. Entity name resolution disabled.")
            return

        df = pd.read_csv(map_path)
        for _, row in df.iterrows():
            het_id = str(row["node_id"])
            human_name = str(row.get("name", "")).strip()

            # Always register the full ID as a passthrough key
            self.name_to_id[het_id] = het_id

            # Register the suffix after "::" (e.g. "DB00945", "DOID:9352")
            if "::" in het_id:
                suffix = het_id.split("::", 1)[-1]
                self.name_to_id.setdefault(suffix, het_id)

            # Register the human-readable name (lower-cased for case-insensitive lookup)
            if human_name and human_name.lower() != "nan":
                self.name_to_id.setdefault(human_name.lower(), het_id)
                self.id_to_name[het_id] = human_name

        logger.info(f"Loaded {len(self.name_to_id)} entity name mappings from {map_path}.")
    
    def _resolve_entity(self, name_or_id: str) -> str:
        """Resolve entity name to Hetionet ID."""
        if name_or_id in self.name_to_id:
            return self.name_to_id[name_or_id]
        elif self.embedder is not None and name_or_id in self.embedder.entity_to_id:
            return name_or_id  # Already a Hetionet ID
        else:
            # Fuzzy match or raise error
            matches = [k for k in self.name_to_id.keys() if name_or_id.lower() in k.lower()]
            if matches:
                logger.info(f"Fuzzy match found: {name_or_id} -> {matches[0]}")
                return self.name_to_id[matches[0]]
            else:
                raise ValueError(
                    f"Entity '{name_or_id}' not found in KG. "
                    f"Hetionet uses INN/systematic names — try the INN name or a DrugBank/DOID ID. "
                    f"Examples: 'acetylsalicylic acid' or 'DB00945' for aspirin; "
                    f"'type 2 diabetes mellitus' or 'DOID:9352' for diabetes."
                )
    
    def _prepare_features(self, drug_id: str, disease_id: str) -> np.ndarray:
        """Prepare feature vector for prediction.

        Uses the same 4-feature scheme as training: [h, t, |h-t|, h*t]
        """
        if self.embedder is None:
            raise ValueError("Embedder not loaded — cannot prepare features.")
        drug_emb = self.embedder._get_vec(drug_id, reduced=True)
        disease_emb = self.embedder._get_vec(disease_id, reduced=True)
        features = np.concatenate([
            drug_emb,
            disease_emb,
            np.abs(drug_emb - disease_emb),
            drug_emb * disease_emb,
        ])
        return features.reshape(1, -1)  # Shape: (1, 4*qml_dim)
    
    def predict_link_probability(
        self,
        drug: str,
        disease: str,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Predict probability that drug treats disease.
        
        Args:
            drug: Drug name or Hetionet ID (e.g., "Aspirin" or "Compound::DB00945")
            disease: Disease name or Hetionet ID (e.g., "Diabetes" or "Disease::DOID_9352")
            method: "classical", "quantum", or "auto" (quantum if available, else classical)
        
        Returns:
            Dict with prediction details
        """
        try:
            # Resolve entities
            drug_id = self._resolve_entity(drug)
            disease_id = self._resolve_entity(disease)
            logger.info(f"Resolved: {drug} -> {drug_id}, {disease} -> {disease_id}")
            
            # Prepare features
            X = self._prepare_features(drug_id, disease_id)
            
            # Choose method
            if method == "auto":
                use_quantum = self.use_quantum
            elif method == "quantum":
                if not self.use_quantum:
                    raise RuntimeError("Quantum model not available.")
                use_quantum = True
            else:  # classical
                use_quantum = False
            
            # Predict
            if use_quantum and self.quantum_predictor:
                try:
                    proba = self.quantum_predictor.predict_proba(X)[0, 1]
                    model_used = "quantum"
                except Exception as e:
                    logger.warning(f"Quantum prediction failed: {e}. Falling back to classical.")
                    proba = self.classical_predictor.predict(X)[0]
                    model_used = "classical_fallback"
            else:
                proba = self.classical_predictor.predict(X)[0]
                model_used = "classical"
            
            return {
                "drug": drug,
                "disease": disease,
                "drug_id": drug_id,
                "disease_id": disease_id,
                "link_probability": float(proba),
                "model_used": model_used,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "drug": drug,
                "disease": disease,
                "link_probability": 0.0,
                "model_used": "error",
                "status": "error",
                "error_message": str(e)
            }

    def rank_mechanism_candidates(
        self,
        hypothesis_id: str,
        disease_id: str,
        top_k: int = 50,
        method: str = "auto",
    ) -> Dict[str, Any]:
        """
        Rank compound candidates for a disease using mechanism-informed subgraph.

        Args:
            hypothesis_id: H-001, H-002, or H-003
            disease_id: Disease entity ID (e.g., Disease::DOID:1234)
            top_k: Number of top candidates to return
            method: Prediction method ("classical", "quantum", "auto")

        Returns:
            Dict with ranked_candidates, model_used, hypothesis_id
        """
        try:
            disease_id = self._resolve_entity(disease_id)
        except ValueError:
            return {
                "hypothesis_id": hypothesis_id,
                "ranked_candidates": [],
                "model_used": "error",
                "status": "error",
                "error_message": f"Disease '{disease_id}' not found",
            }

        try:
            df_edges = load_hetionet_edges(data_dir=self.data_dir)
        except Exception as e:
            logger.error(f"Failed to load Hetionet: {e}")
            return {
                "hypothesis_id": hypothesis_id,
                "ranked_candidates": [],
                "model_used": "error",
                "status": "error",
                "error_message": str(e),
            }

        def predictor(drug_id: str, dis_id: str, m: str):
            return self.predict_link_probability(drug=drug_id, disease=dis_id, method=m)

        compound_ids = None
        if self.embedder:
            all_compounds = [
                e for e in self.embedder.entity_to_id
                if isinstance(e, str) and e.startswith("Compound::")
            ]
            compound_ids = all_compounds[:500] if all_compounds else None

        return _rank_mechanism_candidates(
            hypothesis_id=hypothesis_id,
            disease_id=disease_id,
            top_k=top_k,
            df_edges=df_edges,
            predictor=predictor,
            id_to_name=self.id_to_name,
            compound_ids=compound_ids,
            max_compounds=200,
            method=method,
        )

    def _load_quantum_model(self, config_path: Optional[str] = None) -> None:
        """Load pre-trained quantum model (not implemented in this PoC)."""
        # In a full system, you'd save/load QML models
        # For now, we'll retrain on demand or use a cached config
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                qml_config = yaml.safe_load(f)
        else:
            # Default config matching your training
            qml_config = {
                "model_type": "VQC",
                "encoding_method": "feature_map",
                "num_qubits": 5,
                "feature_map_type": "ZZ",
                "feature_map_reps": 2,
                "ansatz_type": "RealAmplitudes",
                "ansatz_reps": 3,
                "optimizer": "COBYLA",
                "max_iter": 50,
                "random_state": 42,
                "quantum_config_path": "config/quantum_config.yaml"
            }
        
        # Create predictor (weights would be loaded in production)
        self.quantum_predictor = QMLLinkPredictor(**qml_config)
        # Note: In this PoC, we don't save QML model weights
        # So quantum predictions will use a randomly initialized model
        # For demo purposes, we'll disable quantum prediction by default
        logger.warning("Quantum model weights not saved. Using classical fallback for predictions.")
        self.use_quantum = False


# Example usage (uncomment to test)
# if __name__ == "__main__":
#     orchestrator = LinkPredictionOrchestrator(use_quantum=False)
#     
#     # Test with known drug-disease pair
#     result = orchestrator.predict_link_probability("DB00945", "DOID_9352")  # Aspirin for Diabetes?
#     print(result)
#     
#     # Test with names (if mapped)
#     # result = orchestrator.predict_link_probability("Aspirin", "Diabetes")
#     # print(result)