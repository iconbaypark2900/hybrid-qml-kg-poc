# middleware/orchestrator.py

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor
from quantum_layer.qml_model import QMLLinkPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # Initialize embedder (read-only mode)
        self.embedder = HetionetEmbedder()
        if not self.embedder.load_saved_embeddings():
            raise RuntimeError("Embeddings not found. Run kg_embedder.py first.")
        
        # Load classical model
        self.classical_predictor = ClassicalLinkPredictor()
        if not self.classical_predictor.load_model():
            raise RuntimeError("Classical model not found. Run classical_baseline/train_baseline.py first.")
        
        # Load quantum model (if requested)
        self.quantum_predictor = None
        if self.use_quantum:
            try:
                self._load_quantum_model(qml_config_path)
            except Exception as e:
                logger.warning(f"Failed to load quantum model: {e}. Falling back to classical only.")
                self.use_quantum = False
    
    def _load_entity_mappings(self) -> None:
        """Load entity name to Hetionet ID mappings."""
        # For PoC, we'll use a simple name-to-ID mapping
        # In production, you'd use a full biomedical name resolver (e.g., MetaMap, SciSpacy)
        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}
        
        # Load from id_to_entity.csv
        map_path = os.path.join(self.data_dir, "id_to_entity.csv")
        if os.path.exists(map_path):
            df = pd.read_csv(map_path)
            for _, row in df.iterrows():
                het_id = row["0"]
                # Extract human-readable name (simplified)
                if het_id.startswith("Compound::"):
                    name = het_id.split("::")[-1]  # e.g., DB00945
                    # In real system, map DB IDs to drug names via DrugBank
                    self.name_to_id[name] = het_id
                    self.id_to_name[het_id] = name
                elif het_id.startswith("Disease::"):
                    name = het_id.split("::")[-1]  # e.g., DOID_9352
                    self.name_to_id[name] = het_id
                    self.id_to_name[het_id] = name
                # Add more entity types as needed
        
        logger.info(f"Loaded {len(self.name_to_id)} entity name mappings.")
    
    def _resolve_entity(self, name_or_id: str) -> str:
        """Resolve entity name to Hetionet ID."""
        if name_or_id in self.name_to_id:
            return self.name_to_id[name_or_id]
        elif name_or_id in self.embedder.entity_to_id:
            return name_or_id  # Already a Hetionet ID
        else:
            # Fuzzy match or raise error
            matches = [k for k in self.name_to_id.keys() if name_or_id.lower() in k.lower()]
            if matches:
                logger.info(f"Fuzzy match found: {name_or_id} -> {matches[0]}")
                return self.name_to_id[matches[0]]
            else:
                raise ValueError(f"Entity '{name_or_id}' not found in KG.")
    
    def _prepare_features(self, drug_id: str, disease_id: str) -> np.ndarray:
        """Prepare feature vector for prediction."""
        # Get embeddings
        try:
            drug_emb = self.embedder.get_entity_embedding(drug_id, reduced=True)
            disease_emb = self.embedder.get_entity_embedding(disease_id, reduced=True)
        except KeyError as e:
            raise ValueError(f"Embedding not found for entity: {e}")
        
        # Concatenate
        features = np.concatenate([drug_emb, disease_emb])
        return features.reshape(1, -1)  # Shape: (1, n_features)
    
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