"""
Advanced Knowledge Graph Embedding methods for improved link prediction.

Supports multiple state-of-the-art embedding algorithms:
- ComplEx: Complex-valued embeddings for asymmetric relations
- RotatE: Rotation-based embeddings in complex space
- DistMult: Bilinear diagonal model
- TransE: Translation-based (baseline, already implemented)

Uses PyKEEN library with optimized hyperparameters for biomedical KGs.
"""

import os
import json
import logging
from typing import Dict, Tuple, Optional, Literal
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pykeen.pipeline import pipeline
    from pykeen.datasets.base import PathDataset
    from pykeen.models import ComplEx, RotatE, DistMult, TransE
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    logger.warning("PyKEEN not available. Install with: pip install pykeen")


EmbeddingMethod = Literal['TransE', 'ComplEx', 'RotatE', 'DistMult']


class AdvancedKGEmbedder:
    """
    Advanced Knowledge Graph embedder with multiple algorithm support.

    Recommended settings for biomedical KGs (like Hetionet):
    - ComplEx: Best for complex asymmetric relations (compound-disease)
    - RotatE: Best for hierarchical and diverse relation types
    - DistMult: Fast, good for symmetric relations
    - TransE: Baseline, good for 1-to-1 relations
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        method: EmbeddingMethod = 'ComplEx',
        num_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        regularization_weight: float = 0.01,
        negative_samples: int = 5,
        work_dir: str = "data",
        random_state: int = 42
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings (64 recommended for ComplEx/RotatE)
            method: Embedding algorithm to use
            num_epochs: Training epochs (100-200 for good convergence)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            regularization_weight: L2 regularization strength
            negative_samples: Number of negative samples per positive
            work_dir: Directory for saving embeddings
            random_state: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.method = method
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization_weight = regularization_weight
        self.negative_samples = negative_samples
        self.work_dir = work_dir
        self.random_state = random_state

        # Storage
        self.entity_embeddings: Optional[np.ndarray] = None
        self.relation_embeddings: Optional[np.ndarray] = None
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_relation: Dict[int, str] = {}

        os.makedirs(work_dir, exist_ok=True)

    def _get_save_paths(self) -> Tuple[str, str, str]:
        """Get file paths for saving embeddings."""
        prefix = f"{self.method.lower()}_{self.embedding_dim}d"
        entity_emb = os.path.join(self.work_dir, f"{prefix}_entity_embeddings.npy")
        entity_ids = os.path.join(self.work_dir, f"{prefix}_entity_ids.json")
        relation_emb = os.path.join(self.work_dir, f"{prefix}_relation_embeddings.npy")
        return entity_emb, entity_ids, relation_emb

    def _get_model_kwargs(self) -> Dict:
        """Get model-specific kwargs for PyKEEN."""
        base_kwargs = {
            'embedding_dim': self.embedding_dim,
            'random_seed': self.random_state,
        }

        if self.method == 'ComplEx':
            # ComplEx uses complex-valued embeddings
            return {
                **base_kwargs,
                'entity_initializer': 'xavier_uniform_',
                'relation_initializer': 'xavier_uniform_',
            }
        elif self.method == 'RotatE':
            # RotatE: relations are rotations in complex space
            return {
                **base_kwargs,
                'entity_initializer': 'xavier_uniform_',
            }
        elif self.method == 'DistMult':
            # DistMult: bilinear model
            return {
                **base_kwargs,
                'entity_initializer': 'xavier_uniform_',
                'relation_initializer': 'xavier_uniform_',
            }
        elif self.method == 'TransE':
            # TransE: translation-based
            return {
                **base_kwargs,
                'scoring_fct_norm': 2,  # L2 norm
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def train_embeddings(
        self,
        triples_df: pd.DataFrame,
        validation_triples: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Train embeddings on knowledge graph triples.

        Args:
            triples_df: DataFrame with columns [head/source, relation, tail/target]
            validation_triples: Optional validation set for early stopping

        Returns:
            Dictionary with training results and metrics
        """
        if not PYKEEN_AVAILABLE:
            raise ImportError("PyKEEN is required. Install with: pip install pykeen")

        # Prepare data
        train_path, val_path, test_path = self._prepare_pykeen_dataset(
            triples_df, validation_triples
        )

        dataset = PathDataset(
            training_path=train_path,
            validation_path=val_path,
            testing_path=test_path,
        )

        # Get model configuration
        model_kwargs = self._get_model_kwargs()

        logger.info(f"Training {self.method} embeddings (dim={self.embedding_dim}, epochs={self.num_epochs})...")

        # Training configuration optimized for biomedical KGs
        training_kwargs = {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
        }

        optimizer_kwargs = {
            'lr': self.learning_rate,
            'weight_decay': self.regularization_weight,
        }

        # Negative sampling kwargs
        negative_sampler_kwargs = {
            'num_negs_per_pos': self.negative_samples,
        }

        # Train model
        result = pipeline(
            dataset=dataset,
            model=self.method,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            optimizer='Adam',
            optimizer_kwargs=optimizer_kwargs,
            training_loop='sLCWA',  # stochastic local closed world assumption
            negative_sampler_kwargs=negative_sampler_kwargs,
            stopper='early',
            stopper_kwargs={'patience': 10, 'frequency': 5},
            evaluator='RankBasedEvaluator',
            random_seed=self.random_state,
        )

        # Extract embeddings
        self._extract_embeddings_from_model(result)

        # Save embeddings
        self._save_embeddings()

        # Extract metrics
        metrics = {
            'method': self.method,
            'embedding_dim': self.embedding_dim,
            'num_entities': len(self.entity_to_id),
            'num_relations': len(self.relation_to_id),
            'num_epochs_completed': self.num_epochs,
            'final_loss': float(result.losses[-1]) if result.losses else None,
            'hits_at_10': result.metric_results.get_metric('hits_at_10'),
            'mrr': result.metric_results.get_metric('mean_reciprocal_rank'),
        }

        logger.info(f"Training complete. MRR: {metrics['mrr']:.4f}, Hits@10: {metrics['hits_at_10']:.4f}")

        return metrics

    def _prepare_pykeen_dataset(
        self,
        triples_df: pd.DataFrame,
        validation_triples: Optional[pd.DataFrame] = None
    ) -> Tuple[str, str, str]:
        """Prepare triples in PyKEEN format (TSV files)."""
        # Infer columns
        cols = triples_df.columns
        h_col = next((c for c in cols if c.lower() in ['head', 'source', 'source_id', 'h', 'src']), None)
        r_col = next((c for c in cols if c.lower() in ['relation', 'metaedge', 'rel', 'r']), None)
        t_col = next((c for c in cols if c.lower() in ['tail', 'target', 'target_id', 't', 'dst']), None)

        if h_col is None or t_col is None:
            raise ValueError(f"Could not infer head/tail columns from: {list(cols)}")

        # If no relation column, create synthetic one
        if r_col is None:
            triples_df = triples_df.copy()
            triples_df['relation'] = 'treats'  # Default for CtD
            r_col = 'relation'

        # Filter to positive triples if label column exists
        if 'label' in triples_df.columns:
            triples_df = triples_df[triples_df['label'] == 1].copy()

        # Prepare paths
        tmp_dir = os.path.join(self.work_dir, 'pykeen_tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        train_path = os.path.join(tmp_dir, 'train.tsv')
        triples_df[[h_col, r_col, t_col]].to_csv(train_path, sep='\t', index=False, header=False)

        # Validation (if provided)
        if validation_triples is not None:
            val_path = os.path.join(tmp_dir, 'val.tsv')
            if 'label' in validation_triples.columns:
                validation_triples = validation_triples[validation_triples['label'] == 1].copy()
            validation_triples[[h_col, r_col, t_col]].to_csv(val_path, sep='\t', index=False, header=False)
        else:
            # Use training data for validation (not ideal but works)
            val_path = train_path

        test_path = val_path  # Use same as validation

        return train_path, val_path, test_path

    def _extract_embeddings_from_model(self, result):
        """Extract embeddings from trained PyKEEN model."""
        model = result.model

        # Extract entity embeddings
        self.entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()

        # For ComplEx and RotatE, embeddings may be complex-valued
        # Convert to real by concatenating real and imaginary parts
        if np.iscomplexobj(self.entity_embeddings):
            logger.info(f"Converting complex {self.method} embeddings to real.")
            self.entity_embeddings = np.concatenate(
                [self.entity_embeddings.real, self.entity_embeddings.imag], axis=1
            )
            # Update embedding_dim to reflect actual dimension after conversion
            self.embedding_dim = self.entity_embeddings.shape[1]
            logger.info(f"Updated embedding_dim to {self.embedding_dim} after complex→real conversion.")

        # Extract relation embeddings
        if hasattr(model, 'relation_representations'):
            self.relation_embeddings = model.relation_representations[0]().detach().cpu().numpy()

        # Build ID mappings
        ent2id = result.training.entity_to_id
        rel2id = result.training.relation_to_id

        self.entity_to_id = {str(k): int(v) for k, v in ent2id.items()}
        self.id_to_entity = {int(v): str(k) for k, v in ent2id.items()}
        self.relation_to_id = {str(k): int(v) for k, v in rel2id.items()}
        self.id_to_relation = {int(v): str(k) for k, v in rel2id.items()}

        logger.info(f"Extracted embeddings: entities {self.entity_embeddings.shape}, "
                   f"relations {self.relation_embeddings.shape if self.relation_embeddings is not None else 'N/A'}")

    def _save_embeddings(self):
        """Save embeddings and ID mappings to disk."""
        entity_emb_path, entity_ids_path, relation_emb_path = self._get_save_paths()

        # Save entity embeddings
        np.save(entity_emb_path, self.entity_embeddings)

        # Save entity IDs
        with open(entity_ids_path, 'w') as f:
            json.dump(self.entity_to_id, f)

        # Save relation embeddings if available
        if self.relation_embeddings is not None:
            np.save(relation_emb_path, self.relation_embeddings)

        logger.info(f"Saved embeddings to {entity_emb_path}")

    def load_embeddings(self) -> bool:
        """Load previously trained embeddings."""
        entity_emb_path, entity_ids_path, relation_emb_path = self._get_save_paths()

        if not (os.path.exists(entity_emb_path) and os.path.exists(entity_ids_path)):
            logger.warning(f"Embeddings not found at {entity_emb_path}")
            return False

        try:
            self.entity_embeddings = np.load(entity_emb_path)

            # Convert complex embeddings to real (e.g., from ComplEx method)
            if np.iscomplexobj(self.entity_embeddings):
                logger.info("Converting complex embeddings to real by concatenating real and imaginary parts.")
                self.entity_embeddings = np.concatenate(
                    [self.entity_embeddings.real, self.entity_embeddings.imag], axis=1
                )
                # Update embedding_dim to reflect actual dimension after conversion
                self.embedding_dim = self.entity_embeddings.shape[1]
                logger.info(f"Updated embedding_dim to {self.embedding_dim} after complex→real conversion.")

            with open(entity_ids_path, 'r') as f:
                self.entity_to_id = {str(k): int(v) for k, v in json.load(f).items()}
            self.id_to_entity = {int(v): str(k) for k, v in self.entity_to_id.items()}

            if os.path.exists(relation_emb_path):
                self.relation_embeddings = np.load(relation_emb_path)

            logger.info(f"Loaded embeddings: {self.entity_embeddings.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return False

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific entity."""
        if entity_id not in self.entity_to_id:
            return None
        idx = self.entity_to_id[entity_id]
        return self.entity_embeddings[idx]

    def get_relation_embedding(self, relation_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific relation."""
        if self.relation_embeddings is None or relation_id not in self.relation_to_id:
            return None
        idx = self.relation_to_id[relation_id]
        return self.relation_embeddings[idx]

    def get_all_embeddings(self, entity_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all entities (optionally filtered by type).

        Args:
            entity_type: Optional entity type prefix (e.g., 'Compound::', 'Disease::')

        Returns:
            Dict mapping entity_id -> embedding vector
        """
        if self.entity_embeddings is None:
            raise RuntimeError("No embeddings loaded. Call train_embeddings() or load_embeddings() first.")

        embeddings = {}
        for entity_id, idx in self.entity_to_id.items():
            if entity_type is None or entity_id.startswith(entity_type):
                embeddings[entity_id] = self.entity_embeddings[idx]

        return embeddings


def compare_embedding_methods(
    triples_df: pd.DataFrame,
    methods: list = ['TransE', 'ComplEx', 'RotatE', 'DistMult'],
    embedding_dim: int = 64,
    work_dir: str = "data",
    results_dir: str = "results"
) -> pd.DataFrame:
    """
    Compare multiple embedding methods on the same dataset.

    Args:
        triples_df: Training triples
        methods: List of methods to compare
        embedding_dim: Embedding dimension
        work_dir: Working directory
        results_dir: Results directory

    Returns:
        DataFrame with comparison results
    """
    results = []

    for method in methods:
        logger.info(f"\n{'='*60}\nTraining {method} embeddings\n{'='*60}")

        embedder = AdvancedKGEmbedder(
            embedding_dim=embedding_dim,
            method=method,
            num_epochs=100,
            work_dir=work_dir
        )

        try:
            metrics = embedder.train_embeddings(triples_df)
            results.append(metrics)
        except Exception as e:
            logger.error(f"Failed to train {method}: {e}")
            results.append({
                'method': method,
                'error': str(e)
            })

    # Save comparison results
    results_df = pd.DataFrame(results)
    os.makedirs(results_dir, exist_ok=True)

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(results_dir, f"embedding_comparison_{stamp}.csv")
    results_df.to_csv(results_path, index=False)

    logger.info(f"\nComparison results saved to: {results_path}")
    logger.info(f"\n{results_df}")

    return results_df
