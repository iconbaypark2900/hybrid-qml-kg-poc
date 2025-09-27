# kg_layer/kg_embedder.py

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import logging

# Optional: Only import PyKEEN if needed
try:
    from pykeen.pipeline import pipeline
    from pykeen.datasets.base import PathDataset
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    logging.warning("PyKEEN not installed. Only precomputed embeddings supported.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HetionetEmbedder:
    """
    Embedding generator for Hetionet subgraphs.
    Supports training via PyKEEN or loading precomputed embeddings.
    Includes dimensionality reduction for QML compatibility.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        embedding_dim: int = 50,        # Original embedding size
        qml_dim: int = 5,               # Reduced size for quantum circuits
        model_name: str = "TransE",     # or "DistMult"
        random_state: int = 42
    ):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self.qml_dim = qml_dim
        self.model_name = model_name
        self.random_state = random_state
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.pca = None
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _create_pykeen_dataset(self, train_edges: pd.DataFrame) -> PathDataset:
        """Convert sampled edges into a PyKEEN-compatible dataset."""
        if not PYKEEN_AVAILABLE:
            raise ImportError("PyKEEN is required to train embeddings. Install with: pip install pykeen")
        
        # Save edges to temporary files
        train_path = os.path.join(self.data_dir, "hetionet_train.tsv")
        train_edges[["source", "metaedge", "target"]].to_csv(
            train_path, sep='\t', index=False, header=False
        )
        
        from pykeen.triples import TriplesFactory
        tf = TriplesFactory.from_path(train_path)
        
        # Wrap as minimal dataset
        class TempDataset(PathDataset):
            def __init__(self):
                self._training = tf
                self._validation = tf  # dummy
                self._testing = tf     # dummy
        
        return TempDataset()
    
    def train_embeddings(self, train_edges: pd.DataFrame) -> None:
        """
        Train KG embeddings using PyKEEN on the provided edges.
        Assumes edges have 'source', 'metaedge', 'target' columns.
        """
        if not PYKEEN_AVAILABLE:
            raise RuntimeError("PyKEEN not available. Use load_precomputed() instead.")
        
        logger.info(f"Training {self.model_name} embeddings (dim={self.embedding_dim})...")
        
        dataset = self._create_pykeen_dataset(train_edges)
        
        result = pipeline(
            dataset=dataset,
            model=self.model_name,
            model_kwargs=dict(embedding_dim=self.embedding_dim),
            training_kwargs=dict(num_epochs=100, batch_size=128),
            random_seed=self.random_state,
            device="cpu"  # Use "cuda" if GPU available
        )
        
        # Extract embeddings
        entity_ids = result.training.entity_ids  # list of entity labels
        entity_embeddings = result.model.entity_embeddings.weight.detach().cpu().numpy()
        
        # Build mapping
        self.entity_to_id = {entity: idx for idx, entity in enumerate(entity_ids)}
        self.id_to_entity = {idx: entity for entity, idx in self.entity_to_id.items()}
        self.embeddings = entity_embeddings
        
        logger.info(f"Trained embeddings for {len(entity_ids)} entities.")
        self._save_embeddings()
    
    def load_precomputed(self, embedding_file: str, entity_file: str) -> None:
        """
        Load precomputed embeddings (e.g., from Zenodo or your own run).
        
        Expected format:
          - embedding_file: CSV or NumPy array (N x D)
          - entity_file: text file with one entity ID per line (order matches embeddings)
        """
        logger.info(f"Loading precomputed embeddings from {embedding_file}...")
        
        # Load entities
        with open(entity_file, 'r') as f:
            entities = [line.strip() for line in f if line.strip()]
        
        # Load embeddings
        if embedding_file.endswith('.npy'):
            embeddings = np.load(embedding_file)
        elif embedding_file.endswith('.csv'):
            embeddings = pd.read_csv(embedding_file, header=None).values
        else:
            raise ValueError("Embedding file must be .npy or .csv")
        
        assert len(entities) == embeddings.shape[0], "Entity count mismatch"
        
        self.entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
        self.id_to_entity = {idx: entity for entity, idx in self.entity_to_id.items()}
        self.embeddings = embeddings
        
        logger.info(f"Loaded embeddings for {len(entities)} entities.")
        self._save_embeddings()
    
    def _save_embeddings(self) -> None:
        """Save embeddings and mappings for reproducibility."""
        np.save(os.path.join(self.data_dir, "embeddings.npy"), self.embeddings)
        pd.Series(self.id_to_entity).to_csv(
            os.path.join(self.data_dir, "id_to_entity.csv"), index_label="id"
        )
        logger.info("Saved embeddings and entity mappings.")
    
    def load_saved_embeddings(self) -> bool:
        """Load previously saved embeddings (if they exist)."""
        embed_path = os.path.join(self.data_dir, "embeddings.npy")
        map_path = os.path.join(self.data_dir, "id_to_entity.csv")
        
        if os.path.exists(embed_path) and os.path.exists(map_path):
            self.embeddings = np.load(embed_path)
            id_to_entity_df = pd.read_csv(map_path, index_col="id")
            self.id_to_entity = id_to_entity_df["0"].to_dict()
            self.entity_to_id = {ent: idx for idx, ent in self.id_to_entity.items()}
            logger.info("Loaded saved embeddings.")
            return True
        return False
    
    def reduce_to_qml_dim(self) -> np.ndarray:
        """
        Reduce embedding dimensionality to qml_dim using PCA.
        Returns normalized embeddings suitable for quantum encoding.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call train_embeddings() or load_precomputed() first.")
        
        if self.qml_dim >= self.embeddings.shape[1]:
            logger.warning("qml_dim >= original dim. Skipping PCA.")
            reduced = self.embeddings
        else:
            logger.info(f"Reducing embeddings from {self.embeddings.shape[1]} to {self.qml_dim} dimensions...")
            self.pca = PCA(n_components=self.qml_dim, random_state=self.random_state)
            reduced = self.pca.fit_transform(self.embeddings)
        
        # Normalize to unit vectors (important for amplitude encoding)
        reduced = normalize(reduced, norm='l2')
        
        # Save reduced embeddings
        np.save(os.path.join(self.data_dir, f"embeddings_qml_{self.qml_dim}d.npy"), reduced)
        logger.info(f"Saved reduced embeddings ({reduced.shape}).")
        return reduced
    
    def get_entity_embedding(self, entity_id: str, reduced: bool = True) -> np.ndarray:
        """Get embedding for a specific entity."""
        if entity_id not in self.entity_to_id:
            raise KeyError(f"Entity {entity_id} not found in KG.")
        
        idx = self.entity_to_id[entity_id]
        if reduced:
            emb_file = os.path.join(self.data_dir, f"embeddings_qml_{self.qml_dim}d.npy")
            if os.path.exists(emb_file):
                reduced_embs = np.load(emb_file)
                return reduced_embs[idx]
            else:
                raise FileNotFoundError("Reduced embeddings not found. Call reduce_to_qml_dim() first.")
        else:
            return self.embeddings[idx]
    
    def prepare_link_features(
        self, 
        edge_df: pd.DataFrame, 
        source_col: str = "source", 
        target_col: str = "target"
    ) -> np.ndarray:
        """
        Prepare feature matrix for link prediction.
        Each row = [source_emb; target_emb] (concatenated).
        """
        reduced_embs = np.load(os.path.join(self.data_dir, f"embeddings_qml_{self.qml_dim}d.npy"))
        
        features = []
        for _, row in edge_df.iterrows():
            src_id = row[source_col]
            tgt_id = row[target_col]
            
            if src_id not in self.entity_to_id or tgt_id not in self.entity_to_id:
                logger.warning(f"Skipping edge ({src_id}, {tgt_id}): entity not in embedding space.")
                continue
                
            src_idx = self.entity_to_id[src_id]
            tgt_idx = self.entity_to_id[tgt_id]
            
            src_emb = reduced_embs[src_idx]
            tgt_emb = reduced_embs[tgt_idx]
            
            # Concatenate embeddings
            features.append(np.concatenate([src_emb, tgt_emb]))
        
        return np.array(features)


# Example usage (uncomment to test)
# if __name__ == "__main__":
#     from kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
#     
#     # Load data
#     df = load_hetionet_edges()
#     task_edges, ent2id, id2ent = extract_task_edges(df, relation_type="CtD", max_entities=500)
#     train_df, test_df = prepare_link_prediction_dataset(task_edges)
#     
#     # Train embeddings
#     embedder = HetionetEmbedder(embedding_dim=32, qml_dim=5)
#     embedder.train_embeddings(train_df)
#     embedder.reduce_to_qml_dim()
#     
#     # Prepare features
#     X_train = embedder.prepare_link_features(train_df)
#     print("Train feature shape:", X_train.shape)