# kg_layer/kg_embedder.py
"""
Robust KG embedding utility for Hetionet link prediction.

- If PyKEEN is available, trains TransE embeddings from triples.
- If PyKEEN is NOT available, falls back to deterministic random embeddings
  (seeded by entity string) so the rest of the pipeline can run and produce metrics.

Also provides PCA reduction to a lower-dimensional space for QML features.
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Optional: Only import PyKEEN if present
try:
    from pykeen.pipeline import pipeline
    from pykeen.datasets.base import PathDataset
    PYKEEN_AVAILABLE = True
except Exception:
    PYKEEN_AVAILABLE = False

def _infer_ht_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], str]:
    """
    Infer (head/source), (relation/metaedge), (tail/target) column names.

    Accepts common variants found in this repo's link splits:
    - head/source:  source, source_id, head, head_id, h, src, src_id, u
    - tail/target:  target, target_id, tail, tail_id, t, dst, dst_id, v
    - relation:     metaedge, relation, rel, r, predicate, edge_type (optional)
    """
    cols = {c.lower(): c for c in df.columns}

    head_aliases = ("source", "source_id", "head", "head_id", "h", "src", "src_id", "u")
    tail_aliases = ("target", "target_id", "tail", "tail_id", "t", "dst", "dst_id", "v")
    rel_aliases  = ("metaedge", "relation", "rel", "r", "predicate", "edge_type")

    h_col = next((cols[a] for a in head_aliases if a in cols), None)
    t_col = next((cols[a] for a in tail_aliases if a in cols), None)
    if h_col and t_col:
        r_col = next((cols[a] for a in rel_aliases if a in cols), None)
        return h_col, r_col, t_col

    raise KeyError(
        f"Could not infer head/source and tail/target columns from {list(df.columns)}. "
        "Expected one of (source|source_id|head|head_id|h|src|src_id|u) and "
        "(target|target_id|tail|tail_id|t|dst|dst_id|v)."
    )


class HetionetEmbedder:
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        qml_dim: Optional[int] = None,
        work_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        mechanism_mask: bool = False,
    ):
        """
        Initializes the HetionetEmbedder.

        Args:
            embedding_dim: The dimensionality of the KG embeddings (default 32).
            qml_dim: The dimensionality of the reduced embeddings for QML (default 5).
            work_dir: The directory to store embeddings and other artifacts (default "data").
            config_path: Optional YAML path merged into ``config`` when provided.
            config: Optional dict; overrides keys loaded from ``config_path``.
            mechanism_mask: When True, ``prepare_link_features`` can append mechanism/perturbation features.
        """
        cfg: Dict = dict(config) if config else {}
        if config_path and os.path.isfile(config_path):
            try:
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    cfg = {**loaded, **cfg}
            except Exception as e:
                logger.warning("Could not load embedder config from %s: %s", config_path, e)

        if embedding_dim is None:
            embedding_dim = int(cfg.get("embedding_dim", 32))
        else:
            embedding_dim = int(embedding_dim)
        if qml_dim is None:
            qml_dim = int(cfg.get("qml_dim", 5))
        else:
            qml_dim = int(qml_dim)
        if work_dir is None:
            work_dir = str(cfg.get("work_dir", "data"))

        self.embedding_dim = embedding_dim
        self.qml_dim = qml_dim
        self.work_dir = work_dir

        mech = cfg.get("mechanism") if isinstance(cfg.get("mechanism"), dict) else {}
        self.mechanism_mask = bool(mechanism_mask or mech.get("mechanism_mask", False))

        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.entity_embeddings: Optional[np.ndarray] = None  # shape: [N, D]
        self.reduced_embeddings: Optional[np.ndarray] = None  # shape: [N, qml_dim]
        self._pca: Optional[PCA] = None

        os.makedirs(self.work_dir, exist_ok=True)

    # --------------------------
    # Persistence
    # --------------------------
    def _embeddings_paths(self) -> Tuple[str, str]:
        """
        Gets the paths for the embeddings and entity IDs files.

        Returns:
            A tuple containing the path to the embeddings file and the entity IDs file.
        """
        emb_path = os.path.join(self.work_dir, "entity_embeddings.npy")
        ids_path = os.path.join(self.work_dir, "entity_ids.json")
        return emb_path, ids_path

    def load_saved_embeddings(self, expected_dim: Optional[int] = None) -> bool:
        """
        Loads the embeddings and entity IDs from the work directory.

        Args:
            expected_dim: Expected embedding dimension. If provided and doesn't match,
                         returns False to trigger retraining.

        Returns:
            True if the embeddings were loaded successfully, False otherwise.
        """
        emb_path, ids_path = self._embeddings_paths()
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            try:
                self.entity_embeddings = np.load(emb_path)
                with open(ids_path, "r") as f:
                    mapping = json.load(f)
                # stored as {entity: id}
                self.entity_to_id = {str(k): int(v) for k, v in mapping.items()}
                self.id_to_entity = {int(v): str(k) for k, v in mapping.items()}
                logger.info(
                    f"Loaded saved embeddings: {self.entity_embeddings.shape} for {len(self.entity_to_id)} entities."
                )
                
                # Check if embedding dimension matches expected
                if expected_dim is not None and self.entity_embeddings.shape[1] != expected_dim:
                    logger.warning(
                        f"Saved embedding dimension ({self.entity_embeddings.shape[1]}) "
                        f"doesn't match expected ({expected_dim}). Will retrain."
                    )
                    return False
                    
                return True
            except Exception as e:
                logger.warning(f"Failed to load saved embeddings: {e}")
        return False

    def _save_embeddings(self):
        """
        Saves the embeddings and entity IDs to the work directory.
        """
        emb_path, ids_path = self._embeddings_paths()
        np.save(emb_path, self.entity_embeddings)
        with open(ids_path, "w") as f:
            json.dump(self.entity_to_id, f)
        logger.info(f"Saved embeddings → {emb_path} and ids → {ids_path}")

    # --------------------------
    # Building the vocab
    # --------------------------
    def _build_entity_vocab(self, triples_df: pd.DataFrame):
        """
        Builds the entity vocabulary from a DataFrame of triples.

        Args:
            triples_df: A DataFrame with columns for head and tail entities.
        """
        h_col, _, t_col = _infer_ht_columns(triples_df)
        entities: Iterable[str] = pd.concat([triples_df[h_col], triples_df[t_col]], ignore_index=True).astype(str).unique()
        self.entity_to_id = {ent: i for i, ent in enumerate(entities)}
        self.id_to_entity = {i: ent for ent, i in self.entity_to_id.items()}
        logger.info(f"Built entity vocab of size {len(self.entity_to_id)}.")

    # --------------------------
    # PyKEEN path
    # --------------------------
    def _create_pykeen_dataset(self, triples_df: pd.DataFrame) -> PathDataset:
        """
        Creates a PyKEEN dataset from a DataFrame of triples.

        Args:
            triples_df: A DataFrame with columns for head, relation, and tail entities.

        Returns:
            A PyKEEN PathDataset.
        """
        h_col, r_col, t_col = _infer_ht_columns(triples_df)
        if r_col is None:
            # If we don't have a relation column (e.g., CtD fixed task), synthesize a single-relation column
            triples_df = triples_df.copy()
            triples_df["__rel__"] = "rel"
            r_col = "__rel__"

        # Write triples to temp files
        tmp_dir = os.path.join(self.work_dir, "pykeen_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        train_path = os.path.join(tmp_dir, "train.tsv")
        triples_df[[h_col, r_col, t_col]].to_csv(train_path, sep="\t", index=False, header=False)
        return PathDataset(
            training_path=train_path,
            testing_path=train_path,
            validation_path=train_path,
        )

    def _train_with_pykeen(self, triples_df: pd.DataFrame):
        """
        Trains embeddings using PyKEEN.

        Args:
            triples_df: A DataFrame with columns for head, relation, and tail entities.
        """
        dataset = self._create_pykeen_dataset(triples_df)
        logger.info(f"Training TransE embeddings with PyKEEN (dim={self.embedding_dim})...")
        result = pipeline(
            dataset=dataset,
            model="TransE",
            training_loop="slcwa",
            model_kwargs=dict(embedding_dim=self.embedding_dim),
            training_kwargs=dict(num_epochs=50, batch_size=1024),
            optimizer="adam",
            stopper="early",
        )
        # Extract embeddings
        embs = result.model.entity_representations[0]().detach().cpu().numpy()
        # Map entity ordering to our vocab
        # Ensure vocab built using dataset's entities
        self._build_entity_vocab(pd.DataFrame({
            "source": list(result.training.get_entity_to_id_dict().keys()),
            "target": list(result.training.get_entity_to_id_dict().keys())
        }).iloc[:0])  # vocab only
        # Above trick initializes empty concat; instead, rebuild mapping from model:
        ent2id = result.training.get_entity_to_id_dict()
        id2ent = {v: k for k, v in ent2id.items()}
        self.entity_to_id = dict(ent2id)
        self.id_to_entity = dict(id2ent)
        self.entity_embeddings = embs
        self._save_embeddings()

    # --------------------------
    # Fallback path (no PyKEEN)
    # --------------------------
    @staticmethod
    def _deterministic_vec(key: str, dim: int) -> np.ndarray:
        """
        Generates a deterministic vector for a given key.

        Args:
            key: The key to generate the vector for.
            dim: The dimensionality of the vector.

        Returns:
            A deterministic vector.
        """
        # Seed from SHA1 of the entity string for determinism across runs
        h = hashlib.sha1(key.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little", signed=False) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(size=dim).astype(np.float32)

    def _train_fallback(self, triples_df: pd.DataFrame):
        """
        Generates deterministic random embeddings as a fallback when PyKEEN is not available.

        Args:
            triples_df: A DataFrame with columns for head and tail entities.
        """
        logger.info(
            f"PyKEEN not available → generating deterministic random embeddings (dim={self.embedding_dim})"
        )
        self._build_entity_vocab(triples_df)
        N = len(self.entity_to_id)
        M = self.embedding_dim
        embs = np.zeros((N, M), dtype=np.float32)
        for ent, idx in self.entity_to_id.items():
            embs[idx] = self._deterministic_vec(ent, M)
        # Normalize for stability
        embs = normalize(embs, norm="l2")
        self.entity_embeddings = embs
        self._save_embeddings()

    # --------------------------
    # Public API
    # --------------------------
    def train_embeddings(self, triples_like_df: pd.DataFrame):
        """
        Train (or fallback-generate) node embeddings.

        Accepts either:
        - a triple table with (source/metaedge/target) or (head/relation/tail)
        - a link-prediction table with columns plus a 'label' (we'll keep only positives)

        We ignore the label column for embedding training.

        Args:
            triples_like_df: DataFrame with triples or link data, expected to have
        """
        df = triples_like_df.copy()
        # If it's a link dataset, filter to positives for triples
        if "label" in df.columns:
            df = df[df["label"] == 1].copy()
        # Ensure we have h/t columns
        h_col, r_col, t_col = _infer_ht_columns(df)

        if PYKEEN_AVAILABLE:
            try:
                self._train_with_pykeen(df[[h_col] + ([r_col] if r_col else []) + [t_col]])
                return
            except Exception as e:
                logger.warning(f"PyKEEN training failed ({e}); falling back to deterministic embeddings.")

        self._train_fallback(df[[h_col, t_col]])

    def reduce_to_qml_dim(self):
        """
        Reduces the dimensionality of the embeddings to the QML dimension using PCA.
        """
        if self.entity_embeddings is None:
            raise RuntimeError("No entity_embeddings to reduce. Call train_embeddings() or load_saved_embeddings() first.")
        if self.qml_dim >= self.entity_embeddings.shape[1]:
            logger.info("qml_dim >= embedding_dim; skipping PCA reduction.")
            self.reduced_embeddings = self.entity_embeddings
            self._pca = None
            return
        self._pca = PCA(n_components=self.qml_dim, random_state=42)
        self.reduced_embeddings = self._pca.fit_transform(self.entity_embeddings)
        logger.info(f"Reduced embeddings to shape {self.reduced_embeddings.shape}.")

    # Feature construction for link prediction
    def _get_vec(self, ent: str, reduced: bool = True) -> np.ndarray:
        """
        Gets the vector for a given entity.

        Args:
            ent: The entity to get the vector for.
            reduced: Whether to get the reduced-dimension vector.

        Returns:
            The vector for the given entity.
        """
        if ent not in self.entity_to_id:
            # unseen entity: deterministic vector
            vec = self._deterministic_vec(ent, self.embedding_dim)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            if reduced and (self._pca is not None):
                vec = self._pca.transform(vec.reshape(1, -1))[0]
            return vec.astype(np.float32)

        idx = self.entity_to_id[ent]
        base = self.reduced_embeddings if (reduced and self.reduced_embeddings is not None) else self.entity_embeddings
        return base[idx].astype(np.float32)

    def prepare_link_features(
        self,
        link_df: pd.DataFrame,
        reduced: bool = True,
        mechanism_subgraph_nodes: Optional[Iterable[str]] = None,
        perturbation_assay_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Build pairwise features for (head, tail) edges in link_df:
        [h, t, |h - t|, h * t]  (concat)
        When mechanism_mask is True and mechanism_subgraph_nodes/perturbation_assay_df provided,
        appends signed mechanism and perturbation features.

        Args:
            link_df: DataFrame with head/source and tail/target columns.
            reduced: Whether to use reduced-dimension embeddings.
            mechanism_subgraph_nodes: Optional set of entity IDs in mechanism subgraph (for mechanism indicator).
            perturbation_assay_df: Optional DataFrame with entity_id, perturbation columns.

        Returns:
            X: np.ndarray of shape [num_edges, feature_dim]
        """
        h_col, _, t_col = _infer_ht_columns(link_df)

        # Validate input DataFrame
        if link_df.empty:
            logger.warning("Empty DataFrame provided to prepare_link_features")
            return np.array([]).reshape(0, 0)

        # Convert to string and handle NaN values
        head_entities = link_df[h_col].astype(str).fillna("").values
        tail_entities = link_df[t_col].astype(str).fillna("").values

        # Check for empty strings after conversion
        empty_heads = np.sum(head_entities == "")
        empty_tails = np.sum(tail_entities == "")

        if empty_heads > 0 or empty_tails > 0:
            logger.warning(f"Found {empty_heads} empty head entities and {empty_tails} empty tail entities")

        # Pre-allocate arrays for efficiency
        n_samples = len(link_df)
        if reduced and self.reduced_embeddings is not None:
            embedding_size = self.reduced_embeddings.shape[1]
        elif self.entity_embeddings is not None:
            embedding_size = self.entity_embeddings.shape[1]
        else:
            embedding_size = self.embedding_dim
        feature_size = embedding_size * 4  # [h, t, |h-t|, h*t]
        
        X = np.zeros((n_samples, feature_size), dtype=np.float32)
        
        missing_heads = 0
        missing_tails = 0
        
        # Process entities in batches for better performance
        batch_size = min(1000, n_samples)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_heads = head_entities[start_idx:end_idx]
            batch_tails = tail_entities[start_idx:end_idx]
            
            for i, (h, t) in enumerate(zip(batch_heads, batch_tails)):
                local_idx = start_idx + i

                # Skip empty entities
                if h == "" or t == "":
                    continue

                # Track missing entities
                if h not in self.entity_to_id:
                    missing_heads += 1
                if t not in self.entity_to_id:
                    missing_tails += 1

                hv = self._get_vec(h, reduced=reduced)
                tv = self._get_vec(t, reduced=reduced)
                diff = np.abs(hv - tv)
                had = hv * tv
                X[local_idx] = np.concatenate([hv, tv, diff, had], axis=0)

        # Validation: warn if too many entities are missing
        if n_samples > 0:
            missing_head_pct = missing_heads / n_samples * 100
            missing_tail_pct = missing_tails / n_samples * 100

            if missing_head_pct > 10 or missing_tail_pct > 10:
                logger.warning(
                    "HIGH MISSING ENTITY RATE detected in prepare_link_features:\n"
                    f"   Missing heads: {missing_heads}/{n_samples} ({missing_head_pct:.1f}%)\n"
                    f"   Missing tails: {missing_tails}/{n_samples} ({missing_tail_pct:.1f}%)\n"
                    "   This may indicate a column mismatch (source vs source_id).\n"
                    "   Check that DataFrame has correct string entity ID columns."
                )

            if missing_head_pct > 90 or missing_tail_pct > 90:
                logger.error(
                    "CRITICAL: Almost all entities missing. "
                    f"Heads: {missing_head_pct:.1f}%, Tails: {missing_tail_pct:.1f}%. "
                    "Likely cause: DataFrame has integer IDs but string IDs expected. "
                    f"Sample head values: {link_df[h_col].head(3).tolist()}"
                )

        # Mechanism-specific features (only when mechanism_mask is True)
        if self.mechanism_mask and (mechanism_subgraph_nodes is not None or perturbation_assay_df is not None):
            from .perturbation_encoder import build_perturbation_features

            mech_nodes = set(mechanism_subgraph_nodes) if mechanism_subgraph_nodes else set()
            pert_dim = 4
            h_in = np.array([1.0 if str(h) in mech_nodes else 0.0 for h in head_entities], dtype=np.float32)
            t_in = np.array([1.0 if str(t) in mech_nodes else 0.0 for t in tail_entities], dtype=np.float32)
            mech_block = np.column_stack([h_in, t_in])
            if perturbation_assay_df is not None and len(perturbation_assay_df) > 0:
                pert_h = build_perturbation_features(head_entities.tolist(), perturbation_assay_df, pert_dim)
                pert_t = build_perturbation_features(tail_entities.tolist(), perturbation_assay_df, pert_dim)
                pert_block = np.concatenate([pert_h, pert_t], axis=1)
                mech_block = np.concatenate([mech_block, pert_block], axis=1)
            X = np.concatenate([X, mech_block], axis=1)

        return X

    def prepare_link_features_qml(self, link_df: pd.DataFrame, mode: str = "diff") -> np.ndarray:
        """
        QML-friendly features in reduced space (length == qml_dim):
          - "diff": |h - t|
          - "hadamard": h ⊙ t
          - "both": concat([|h - t|, h ⊙ t]) and project back to qml_dim via PCA fitted on train features

        Args:
            link_df: DataFrame with head/source and tail/target columns.
            mode: One of "diff", "hadamard", or "both".

        Returns:
            X: np.ndarray of shape [num_edges, qml_dim]
        """
        if self.reduced_embeddings is None:
            if self.entity_embeddings is None:
                raise RuntimeError("No embeddings loaded. Call train_embeddings() or load_saved_embeddings() first.")
            self.reduce_to_qml_dim()

        h_col, _, t_col = _infer_ht_columns(link_df)

        diffs, hads = [], []
        for h, t in zip(link_df[h_col].astype(str).values, link_df[t_col].astype(str).values):
            hv = self._get_vec(h, reduced=True)
            tv = self._get_vec(t, reduced=True)
            diffs.append(np.abs(hv - tv))
            hads.append(hv * tv)

        diffs = np.stack(diffs, 0)
        hads  = np.stack(hads, 0)

        if mode == "diff":
            return diffs
        if mode == "hadamard":
            return hads
        if mode == "both":
            both = np.concatenate([diffs, hads], axis=1)  # shape [n, 2*qml_dim]
            # project back to qml_dim using a small PCA trained on-the-fly
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.qml_dim, random_state=42)
            return pca.fit_transform(both)
        raise ValueError(f"Unknown mode: {mode}")
