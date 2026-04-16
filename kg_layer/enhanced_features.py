"""
Enhanced feature engineering for link prediction.

Combines multiple feature types:
1. Embedding-based features (enhanced beyond basic concat)
2. Graph structural features (centrality, neighborhoods, paths)
3. Domain-specific features for biomedical graphs
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def validate_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, edges_df: pd.DataFrame):
    """
    Validate that edges_df doesn't contain test edges (prevents data leakage).

    Args:
        train_df: Training DataFrame with source_id, target_id, label
        test_df: Test DataFrame with source_id, target_id, label
        edges_df: Edges DataFrame to validate (should only contain train edges)

    Raises:
        ValueError: If test edges are found in edges_df
    """
    # Get test positive edges
    test_positive = test_df[test_df['label'] == 1]

    # Determine column names
    edge_cols = edges_df.columns
    src_col = next((c for c in edge_cols if c.lower() in ['source', 'source_id', 'head', 'h', 'src']), None)
    tgt_col = next((c for c in edge_cols if c.lower() in ['target', 'target_id', 'tail', 't', 'dst']), None)

    if src_col is None or tgt_col is None:
        logger.warning("Could not validate leakage - column names not found")
        return

    # Create set of test edges
    test_edges = set(zip(test_positive['source_id'], test_positive['target_id']))

    # Check if any test edges are in edges_df
    edges_set = set(zip(edges_df[src_col], edges_df[tgt_col]))

    # Find intersection
    leakage = test_edges & edges_set

    if leakage:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Found {len(leakage)} test edges in edges_df. "
            f"This will lead to overly optimistic performance estimates. "
            f"Use only training edges for graph/domain features."
        )

    logger.info(f"✓ Leakage check passed: No test edges found in edges_df ({len(edges_df)} edges validated)")


class EnhancedFeatureBuilder:
    """
    Build comprehensive feature sets for link prediction.
    """

    def __init__(
        self,
        include_graph_features: bool = True,
        include_domain_features: bool = True,
        include_directional_features: bool = False,
        include_moa_features: bool = False,
        normalize: bool = True
    ):
        """
        Args:
            include_graph_features: Include topological graph features
            include_domain_features: Include biomedical domain-specific features
            include_directional_features: Include up/down regulation evidence (evidence_up, evidence_down, etc.)
            include_moa_features: Include mechanism-of-action features (binding targets, pathway overlap,
                                 drug class, chemical/disease similarity to known treatments)
            normalize: Apply standard scaling to features
        """
        self.include_graph_features = include_graph_features
        self.include_domain_features = include_domain_features
        self.include_directional_features = include_directional_features
        self.include_moa_features = include_moa_features
        self.normalize = normalize
        self.scaler: Optional[StandardScaler] = None
        self.graph: Optional[nx.Graph] = None
        self._moa_index = None  # Set via build_moa_index()

        # Cache for expensive computations
        self._degree_cache: Dict = {}
        self._pagerank_cache: Dict = {}
        self._betweenness_cache: Dict = {}

    def build_graph(self, edges_df: pd.DataFrame):
        """
        Build NetworkX graph from edges DataFrame.

        Args:
            edges_df: DataFrame with source, target columns (and optionally metaedge)
        """
        # Infer column names
        # Prioritize 'source'/'target' (string entity IDs) over 'source_id'/'target_id' (integer IDs)
        cols = edges_df.columns
        cols_lower = [c.lower() for c in cols]
        
        # Check for exact matches first (prefer string entity IDs)
        if 'source' in cols_lower:
            src_col = cols[cols_lower.index('source')]
        elif 'source_id' in cols_lower:
            src_col = cols[cols_lower.index('source_id')]
        else:
            src_col = next((c for c in cols if c.lower() in ['head', 'h', 'src']), None)
        
        if 'target' in cols_lower:
            tgt_col = cols[cols_lower.index('target')]
        elif 'target_id' in cols_lower:
            tgt_col = cols[cols_lower.index('target_id')]
        else:
            tgt_col = next((c for c in cols if c.lower() in ['tail', 't', 'dst']), None)

        if src_col is None or tgt_col is None:
            raise ValueError(f"Could not infer source/target columns from: {list(cols)}")

        # Filter to positive edges if label exists
        if 'label' in edges_df.columns:
            edges_df = edges_df[edges_df['label'] == 1].copy()

        # Build graph
        self.graph = nx.Graph()
        edges = list(zip(edges_df[src_col], edges_df[tgt_col]))
        self.graph.add_edges_from(edges)

        logger.info(f"Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Pre-compute expensive metrics
        self._precompute_metrics()

    def _precompute_metrics(self):
        """Pre-compute expensive graph metrics for caching."""
        if self.graph is None:
            return

        logger.info("Pre-computing graph metrics...")

        # Degree (fast, but cache anyway)
        self._degree_cache = dict(self.graph.degree())

        # PageRank (moderate cost)
        try:
            self._pagerank_cache = nx.pagerank(self.graph, max_iter=50)
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            self._pagerank_cache = {n: 1.0 for n in self.graph.nodes()}

        # Betweenness centrality (expensive, sample if too large)
        if self.graph.number_of_nodes() <= 1000:
            try:
                self._betweenness_cache = nx.betweenness_centrality(self.graph)
            except Exception as e:
                logger.warning(f"Betweenness centrality failed: {e}")
                self._betweenness_cache = {n: 0.0 for n in self.graph.nodes()}
        else:
            # Sample for large graphs
            logger.info("Large graph detected, sampling for betweenness centrality")
            try:
                self._betweenness_cache = nx.betweenness_centrality(self.graph, k=100)
            except Exception as e:
                logger.warning(f"Betweenness centrality (sampled) failed: {e}")
                self._betweenness_cache = {n: 0.0 for n in self.graph.nodes()}

        logger.info("Graph metrics pre-computed.")

    def build_moa_index(
        self,
        all_edges_df: pd.DataFrame,
        train_ctd_df: pd.DataFrame
    ):
        """
        Build mechanism-of-action lookup tables from full Hetionet edges.

        Args:
            all_edges_df: Full Hetionet edges DataFrame (source, metaedge, target).
            train_ctd_df: Training CtD positive edges with 'source'/'target' string entity IDs.
        """
        from .moa_features import build_moa_index as _build_moa_index
        self._moa_index = _build_moa_index(all_edges_df, train_ctd_df)

    def build_embedding_features(
        self,
        h_emb: np.ndarray,
        t_emb: np.ndarray
    ) -> np.ndarray:
        """
        Build enhanced embedding-based features.

        Features:
        - Original embeddings: h, t
        - Absolute difference: |h - t|
        - Hadamard product: h * t
        - Cosine similarity: cos(h, t)
        - L2 distance: ||h - t||
        - Dot product: h · t
        - Average: (h + t) / 2
        - Weighted combinations: 2h + t, h + 2t

        Args:
            h_emb: Head entity embedding
            t_emb: Tail entity embedding

        Returns:
            Feature vector
        """
        features = []

        # Basic embeddings
        features.append(h_emb)
        features.append(t_emb)

        # Difference and product (existing)
        features.append(np.abs(h_emb - t_emb))
        features.append(h_emb * t_emb)

        # Similarity metrics
        cos_sim = cosine_similarity(h_emb.reshape(1, -1), t_emb.reshape(1, -1))[0, 0]
        l2_dist = np.linalg.norm(h_emb - t_emb)
        dot_prod = np.dot(h_emb, t_emb)

        features.append(np.array([cos_sim, l2_dist, dot_prod]))

        # Average and weighted combinations
        avg = (h_emb + t_emb) / 2.0
        features.append(avg)

        # Squared and sqrt features (for non-linearity)
        features.append(h_emb ** 2)
        features.append(t_emb ** 2)
        features.append(np.sqrt(np.abs(h_emb)))
        features.append(np.sqrt(np.abs(t_emb)))

        return np.concatenate(features)

    def build_graph_features(
        self,
        h_id: str,
        t_id: str
    ) -> np.ndarray:
        """
        Build graph-based topological features.

        Features:
        - Node degrees (h, t)
        - Common neighbors count
        - Jaccard coefficient
        - Adamic-Adar index
        - Preferential attachment
        - Resource allocation index
        - Shortest path length (if exists)
        - PageRank scores (h, t)
        - Betweenness centrality (h, t)
        - Clustering coefficients (h, t)

        Args:
            h_id: Head entity ID
            t_id: Tail entity ID

        Returns:
            Feature vector
        """
        if self.graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        features = []

        # Handle nodes not in graph
        if h_id not in self.graph or t_id not in self.graph:
            # Return zero features
            return np.zeros(14)

        # 1. Degree features
        h_degree = self._degree_cache.get(h_id, 0)
        t_degree = self._degree_cache.get(t_id, 0)
        features.extend([h_degree, t_degree])

        # 2. Common neighbors
        h_neighbors = set(self.graph.neighbors(h_id))
        t_neighbors = set(self.graph.neighbors(t_id))
        common_neighbors = len(h_neighbors & t_neighbors)
        features.append(common_neighbors)

        # 3. Jaccard coefficient
        union_neighbors = len(h_neighbors | t_neighbors)
        jaccard = common_neighbors / union_neighbors if union_neighbors > 0 else 0.0
        features.append(jaccard)

        # 4. Adamic-Adar index
        adamic_adar = 0.0
        for common in (h_neighbors & t_neighbors):
            common_degree = self._degree_cache.get(common, 1)
            if common_degree > 1:
                adamic_adar += 1.0 / np.log(common_degree)
        features.append(adamic_adar)

        # 5. Preferential attachment
        preferential_attachment = h_degree * t_degree
        features.append(preferential_attachment)

        # 6. Resource allocation index
        resource_allocation = 0.0
        for common in (h_neighbors & t_neighbors):
            common_degree = self._degree_cache.get(common, 1)
            if common_degree > 0:
                resource_allocation += 1.0 / common_degree
        features.append(resource_allocation)

        # 7. Shortest path length (expensive, limit to max=5)
        try:
            if nx.has_path(self.graph, h_id, t_id):
                path_length = nx.shortest_path_length(self.graph, h_id, t_id)
                path_length = min(path_length, 10)  # Cap at 10
            else:
                path_length = 10  # Max value for no path
        except Exception:
            path_length = 10
        features.append(path_length)

        # 8. PageRank scores
        h_pagerank = self._pagerank_cache.get(h_id, 0.0)
        t_pagerank = self._pagerank_cache.get(t_id, 0.0)
        features.extend([h_pagerank, t_pagerank])

        # 9. Betweenness centrality
        h_betweenness = self._betweenness_cache.get(h_id, 0.0)
        t_betweenness = self._betweenness_cache.get(t_id, 0.0)
        features.extend([h_betweenness, t_betweenness])

        # 10. Clustering coefficient
        h_clustering = nx.clustering(self.graph, h_id)
        t_clustering = nx.clustering(self.graph, t_id)
        features.extend([h_clustering, t_clustering])

        return np.array(features, dtype=np.float32)

    def build_domain_features(
        self,
        h_id: str,
        t_id: str,
        edges_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Build biomedical domain-specific features for Hetionet.

        Features for Compound-Disease (CtD) pairs:
        - Entity type indicators
        - Metaedge diversity (number of relation types involving these entities)
        - Co-occurrence in other relation types

        Args:
            h_id: Head entity ID
            t_id: Tail entity ID
            edges_df: Full edges DataFrame with metaedge information

        Returns:
            Feature vector
        """
        features = []

        # 1. Entity type one-hot encoding (inferred from ID prefix)
        compound_types = ['Compound', 'Drug']
        disease_types = ['Disease', 'Symptom', 'Phenotype']
        gene_types = ['Gene', 'Protein']

        h_type = h_id.split('::')[0] if '::' in h_id else 'Unknown'
        t_type = t_id.split('::')[0] if '::' in t_id else 'Unknown'

        # Compound indicator
        h_is_compound = float(any(t in h_type for t in compound_types))
        t_is_compound = float(any(t in t_type for t in compound_types))
        features.extend([h_is_compound, t_is_compound])

        # Disease indicator
        h_is_disease = float(any(t in h_type for t in disease_types))
        t_is_disease = float(any(t in t_type for t in disease_types))
        features.extend([h_is_disease, t_is_disease])

        # Gene indicator
        h_is_gene = float(any(t in h_type for t in gene_types))
        t_is_gene = float(any(t in t_type for t in gene_types))
        features.extend([h_is_gene, t_is_gene])

        # 2. Metaedge diversity
        if 'metaedge' in edges_df.columns:
            # Count distinct metaedges involving h_id
            h_metaedges = edges_df[
                (edges_df['source'] == h_id) | (edges_df['target'] == h_id)
            ]['metaedge'].nunique()

            # Count distinct metaedges involving t_id
            t_metaedges = edges_df[
                (edges_df['source'] == t_id) | (edges_df['target'] == t_id)
            ]['metaedge'].nunique()

            features.extend([h_metaedges, t_metaedges])
        else:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def build_features(
        self,
        links_df: pd.DataFrame,
        embeddings: Dict[str, np.ndarray],
        edges_df: Optional[pd.DataFrame] = None,
        fit_scaler: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build complete feature set for link prediction.

        Args:
            links_df: DataFrame with source/target pairs to featurize
            embeddings: Dict mapping entity_id -> embedding vector
            edges_df: DataFrame for domain features (should be TRAIN edges only to avoid leakage)
            fit_scaler: If True, fit scaler on this data (use True for train, False for test)

        Returns:
            (features_array, feature_names)

        Important:
            To prevent data leakage:
            - Set fit_scaler=True ONLY for training data
            - Pass only TRAINING edges in edges_df (not full dataset)
            - Build graph only on TRAINING edges before calling this
        """
        # Infer column names
        # Prioritize 'source'/'target' (string entity IDs) over 'source_id'/'target_id' (integer IDs)
        cols = links_df.columns
        cols_lower = [c.lower() for c in cols]
        
        # Check for exact matches first (prefer string entity IDs)
        if 'source' in cols_lower:
            src_col = cols[cols_lower.index('source')]
        elif 'source_id' in cols_lower:
            src_col = cols[cols_lower.index('source_id')]
        else:
            src_col = next((c for c in cols if c.lower() in ['head', 'h', 'src']), None)
        
        if 'target' in cols_lower:
            tgt_col = cols[cols_lower.index('target')]
        elif 'target_id' in cols_lower:
            tgt_col = cols[cols_lower.index('target_id')]
        else:
            tgt_col = next((c for c in cols if c.lower() in ['tail', 't', 'dst']), None)

        if src_col is None or tgt_col is None:
            raise ValueError(f"Could not infer source/target columns from: {list(cols)}")

        # Pre-build directional features for all rows if requested
        directional_feats_array = None
        if self.include_directional_features and edges_df is not None:
            from .directional_features import build_directional_features, get_directional_feature_names
            from .evidence_weighting import EvidenceConfigDirectional, build_directional_gene_maps
            cfg = EvidenceConfigDirectional()
            comp2g_up, comp2g_down, dis2g_up, dis2g_down = build_directional_gene_maps(edges_df, cfg)
            directional_feats_array = build_directional_features(
                links_df, comp2g_up, comp2g_down, dis2g_up, dis2g_down,
                source_col=src_col, target_col=tgt_col
            )
            directional_feat_names = get_directional_feature_names()

        # Diagnostic: Log inferred columns and sample values
        logger.warning(f"build_features: Inferred src_col='{src_col}', tgt_col='{tgt_col}'")
        if len(links_df) > 0:
            sample_row = links_df.iloc[0]
            logger.warning(f"  Sample row[{src_col}]='{sample_row[src_col]}' (type={type(sample_row[src_col])})")
            logger.warning(f"  Sample row[{tgt_col}]='{sample_row[tgt_col]}' (type={type(sample_row[tgt_col])})")
            logger.warning(f"  After str(): src='{str(sample_row[src_col])}', tgt='{str(sample_row[tgt_col])}'")
            logger.warning(f"  In embeddings? src={str(sample_row[src_col]) in embeddings}, tgt={str(sample_row[tgt_col]) in embeddings}")

        all_features = []
        feature_names = []

        # Determine feature dimensions
        sample_emb = next(iter(embeddings.values()))
        emb_dim = len(sample_emb)

        for idx, row in links_df.iterrows():
            h_id = str(row[src_col])
            t_id = str(row[tgt_col])

            # Get embeddings (fallback to zero if missing)
            h_emb = embeddings.get(h_id, np.zeros(emb_dim, dtype=np.float32))
            t_emb = embeddings.get(t_id, np.zeros(emb_dim, dtype=np.float32))
            
            # Diagnostic for first few rows
            if idx < 3:
                h_found = h_id in embeddings
                t_found = t_id in embeddings
                h_is_zero = np.allclose(h_emb, 0) if h_emb is not None else True
                t_is_zero = np.allclose(t_emb, 0) if t_emb is not None else True
                logger.warning(f"Row {idx}: h_id='{h_id}' found={h_found} zero={h_is_zero}, t_id='{t_id}' found={t_found} zero={t_is_zero}")
                if h_emb is not None and not h_is_zero:
                    logger.warning(f"  h_emb sample: {h_emb[:5]}, std={np.std(h_emb):.6f}")
                if t_emb is not None and not t_is_zero:
                    logger.warning(f"  t_emb sample: {t_emb[:5]}, std={np.std(t_emb):.6f}")

            # 1. Embedding features
            emb_feats = self.build_embedding_features(h_emb, t_emb)
            
            # Diagnostic for first row
            if idx == 0:
                logger.warning(f"Row 0 emb_feats sample: {emb_feats[:10]}, std={np.std(emb_feats):.6f}, all_zero={np.allclose(emb_feats, 0)}")
            
            features_list = [emb_feats]

            if idx == 0:  # First row: build feature names
                feature_names.extend([f'emb_{i}' for i in range(len(emb_feats))])

            # 2. Graph features
            if self.include_graph_features and self.graph is not None:
                graph_feats = self.build_graph_features(h_id, t_id)
                features_list.append(graph_feats)
                if idx == 0:
                    graph_feat_names = [
                        'degree_h', 'degree_t', 'common_neighbors', 'jaccard',
                        'adamic_adar', 'preferential_attachment', 'resource_allocation',
                        'shortest_path', 'pagerank_h', 'pagerank_t',
                        'betweenness_h', 'betweenness_t', 'clustering_h', 'clustering_t'
                    ]
                    feature_names.extend(graph_feat_names)

            # 3. Domain features
            if self.include_domain_features and edges_df is not None:
                domain_feats = self.build_domain_features(h_id, t_id, edges_df)
                features_list.append(domain_feats)
                if idx == 0:
                    domain_feat_names = [
                        'h_is_compound', 't_is_compound', 'h_is_disease', 't_is_disease',
                        'h_is_gene', 't_is_gene', 'h_metaedge_diversity', 't_metaedge_diversity'
                    ]
                    feature_names.extend(domain_feat_names)

            # 4. Directional (perturbation) features
            if self.include_directional_features and directional_feats_array is not None:
                features_list.append(directional_feats_array[idx])
                if idx == 0:
                    feature_names.extend(directional_feat_names)

            # 5. Mechanism-of-Action features
            if self.include_moa_features and self._moa_index is not None:
                from .moa_features import compute_moa_features, MOA_FEATURE_NAMES
                moa_feats = compute_moa_features(h_id, t_id, self._moa_index)
                features_list.append(moa_feats)
                if idx == 0:
                    feature_names.extend(MOA_FEATURE_NAMES)

            # Concatenate all features
            combined_feats = np.concatenate(features_list)
            all_features.append(combined_feats)

        # Convert to array
        features_array = np.array(all_features, dtype=np.float32)
        
        # Diagnostic: Check variance BEFORE normalization
        if len(features_array) > 0:
            pre_norm_std = np.std(features_array, axis=0)
            zero_var_before = np.sum(pre_norm_std < 1e-10)
            logger.debug(f"Pre-normalization: {zero_var_before}/{len(pre_norm_std)} features have zero variance")
            if zero_var_before == len(pre_norm_std):
                logger.warning(f"⚠️  ALL features have zero variance BEFORE normalization!")
                logger.warning(f"  First row sample: {features_array[0, :10]}")
                logger.warning(f"  All rows identical: {np.allclose(features_array[0], features_array[1]) if len(features_array) > 1 else 'N/A'}")

        # Normalize if requested
        if self.normalize:
            if fit_scaler:
                # Fit scaler on training data
                logger.info("Fitting scaler on training features (prevents leakage)")
                self.scaler = StandardScaler()
                features_array = self.scaler.fit_transform(features_array)
            else:
                # Transform using pre-fitted scaler
                if self.scaler is None:
                    raise RuntimeError(
                        "Scaler not fitted! Call build_features with fit_scaler=True on training data first."
                    )
                features_array = self.scaler.transform(features_array)

        logger.info(f"Built features: {features_array.shape} ({len(feature_names)} features)")

        return features_array, feature_names
