"""
Advanced quantum feature engineering for improved QML performance.

Implements multiple encoding strategies optimized for quantum circuits:
1. Amplitude encoding with normalization
2. Phase encoding from feature angles
3. Hybrid encodings combining multiple strategies
4. Feature selection tailored for quantum advantage
5. Optimal dimensionality reduction
"""

import logging
from typing import Optional, Literal, Tuple, List, Any
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import normalize, MinMaxScaler

logger = logging.getLogger(__name__)

EncodingStrategy = Literal['amplitude', 'phase', 'hybrid', 'optimized_diff', 'tensor_product']


class QuantumFeatureEngineer:
    """
    Advanced feature engineering for quantum machine learning.

    Optimizes features for quantum circuits by:
    - Selecting most informative features
    - Normalizing to quantum-friendly ranges
    - Applying quantum-aware dimensionality reduction
    """

    def __init__(
        self,
        num_qubits: int = 5,
        encoding_strategy: EncodingStrategy = 'hybrid',
        feature_selection_method: Optional[str] = 'mutual_info',
        use_kernel_pca: bool = False,
        random_state: int = 42,
        reduction_method: str = 'pca',
        feature_select_k_mult: float = 4.0,
        pre_pca_dim: int = 0
    ):
        """
        Args:
            num_qubits: Number of qubits (determines feature dimensionality)
            encoding_strategy: Quantum encoding strategy
            feature_selection_method: Method for feature selection ('mutual_info', 'f_classif', 'chi2', None)
            use_kernel_pca: Use kernel PCA for non-linear reduction
            random_state: Random seed
            reduction_method: 'pca', 'lda', or 'kernel_pca'
            feature_select_k_mult: Multiplier for feature selection (select k_mult * num_qubits features)
            pre_pca_dim: Pre-PCA dimension (0 = disabled, >0 = reduce to this dim before main reduction)
        """
        self.num_qubits = num_qubits
        self.encoding_strategy = encoding_strategy
        self.feature_selection_method = feature_selection_method
        self.use_kernel_pca = use_kernel_pca
        self.random_state = random_state
        self.reduction_method = reduction_method
        self.feature_select_k_mult = feature_select_k_mult
        self.pre_pca_dim = pre_pca_dim

        # Fitted components
        self.pca: Optional[PCA] = None
        self.pre_pca: Optional[PCA] = None  # Pre-PCA for initial reduction
        self.lda: Optional[Any] = None  # Linear Discriminant Analysis
        self.kpca: Optional[KernelPCA] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.feature_scaler: Optional[Any] = None  # StandardScaler for feature hygiene
        self.scaler: Optional[MinMaxScaler] = None
        self.actual_encoding_strategy: Optional[str] = None  # Store the actual strategy used (may differ from requested)

    def fit_reduction(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None
    ):
        """
        Fit dimensionality reduction and feature selection.

        Args:
            embeddings: High-dimensional embeddings [N, D]
            labels: Labels for supervised feature selection (optional)
        """
        logger.info(f"Fitting quantum feature reduction: {embeddings.shape} -> {self.num_qubits}D")

        # Step 0: Data hygiene - Clean and standardize features
        # Remove NaN and Inf values
        nan_mask = np.isnan(embeddings).any(axis=0) | np.isinf(embeddings).any(axis=0)
        if nan_mask.any():
            logger.warning(f"Removing {nan_mask.sum()} columns with NaN/Inf values")
            embeddings = embeddings[:, ~nan_mask]

        # Remove constant columns (zero variance)
        col_std = np.std(embeddings, axis=0)
        constant_mask = col_std < 1e-10
        if constant_mask.any():
            logger.warning(f"Removing {constant_mask.sum()} constant columns")
            embeddings = embeddings[:, ~constant_mask]

        # Check if we have any features left after removing constants
        if embeddings.shape[1] == 0:
            logger.error(f"All features were constant after removal! Original shape: {embeddings.shape}")
            logger.error("This usually means embeddings are identical or encoding strategy produced constant features.")
            logger.error("Falling back to raw concatenation of head and tail embeddings.")
            # Fallback: use raw head/tail concatenation if we have access to original embeddings
            # For now, raise a more informative error
            raise ValueError(
                f"All {constant_mask.sum()} features were constant (zero variance). "
                f"This suggests the encoding strategy '{self.encoding_strategy}' produced identical features. "
                f"Try a different encoding strategy (e.g., 'tensor_product' or 'optimized_diff') "
                f"or check if embeddings are properly normalized."
            )

        # Standardize features before PCA (important for PCA to work properly)
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        embeddings = self.feature_scaler.fit_transform(embeddings)
        logger.info(f"Standardized features: {embeddings.shape}")

        # Step 0.5: Pre-PCA reduction (if enabled)
        if self.pre_pca_dim > 0 and embeddings.shape[1] > self.pre_pca_dim:
            logger.info(f"Pre-PCA reduction: {embeddings.shape[1]} -> {self.pre_pca_dim}")
            self.pre_pca = PCA(n_components=self.pre_pca_dim, random_state=self.random_state)
            embeddings = self.pre_pca.fit_transform(embeddings)
            logger.info(f"Pre-PCA completed: {embeddings.shape[1]}D")
        
        # Step 1: Feature selection (if supervised)
        if self.feature_selection_method and labels is not None:
            # Select top features based on method
            k = min(embeddings.shape[1], int(self.num_qubits * self.feature_select_k_mult))
            
            if self.feature_selection_method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_classif
                score_func = mutual_info_classif
            elif self.feature_selection_method == 'f_classif':
                from sklearn.feature_selection import f_classif
                score_func = f_classif
            elif self.feature_selection_method == 'chi2':
                from sklearn.feature_selection import chi2
                score_func = chi2
            else:
                score_func = mutual_info_classif
            
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            embeddings = self.feature_selector.fit_transform(embeddings, labels)
            logger.info(f"Feature selection ({self.feature_selection_method}): {embeddings.shape[1]} features selected (k={k})")

        # Step 2: Dimensionality reduction
        if self.reduction_method == 'lda' and labels is not None:
            # Linear Discriminant Analysis (supervised)
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            n_components = min(self.num_qubits, embeddings.shape[0], embeddings.shape[1], len(np.unique(labels)) - 1)
            self.lda = LinearDiscriminantAnalysis(n_components=n_components)
            self.lda.fit(embeddings, labels)
            reduced = self.lda.transform(embeddings)
            logger.info(f"LDA fitted: {embeddings.shape[1]} -> {n_components}D")
        elif self.use_kernel_pca or self.reduction_method == 'kernel_pca':
            # Non-linear reduction with RBF kernel
            self.kpca = KernelPCA(
                n_components=self.num_qubits,
                kernel='rbf',
                gamma=1.0 / embeddings.shape[1],
                random_state=self.random_state,
                n_jobs=-1
            )
            self.kpca.fit(embeddings)
            reduced = self.kpca.transform(embeddings)
            logger.info(f"Kernel PCA fitted: {embeddings.shape[1]} -> {self.num_qubits}D")
        else:
            # Linear PCA (default)
            n_components = min(self.num_qubits, embeddings.shape[0], embeddings.shape[1])
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            self.pca.fit(embeddings)
            reduced = self.pca.transform(embeddings)
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA fitted: {embeddings.shape[1]} -> {n_components}D "
                       f"(explained variance: {explained_var:.2%})")

        # Step 3: Fit scaler for quantum range [-1, 1] or [0, 1]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(reduced)

    def transform_to_qml_space(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to QML-ready feature space.

        Args:
            embeddings: High-dimensional embeddings [N, D]

        Returns:
            QML features [N, num_qubits]
        """
        # Apply feature standardization (if fitted)
        if self.feature_scaler is not None:
            embeddings = self.feature_scaler.transform(embeddings)

        # Apply pre-PCA if enabled
        if self.pre_pca is not None:
            embeddings = self.pre_pca.transform(embeddings)
        
        # Apply feature selection
        if self.feature_selector is not None:
            embeddings = self.feature_selector.transform(embeddings)

        # Apply dimensionality reduction
        if self.lda is not None:
            reduced = self.lda.transform(embeddings)
        elif self.kpca is not None:
            reduced = self.kpca.transform(embeddings)
        elif self.pca is not None:
            reduced = self.pca.transform(embeddings)
        else:
            # No reduction fitted, just truncate/pad
            if embeddings.shape[1] > self.num_qubits:
                reduced = embeddings[:, :self.num_qubits]
            elif embeddings.shape[1] < self.num_qubits:
                reduced = np.pad(embeddings, ((0, 0), (0, self.num_qubits - embeddings.shape[1])))
            else:
                reduced = embeddings

        # Apply scaling
        if self.scaler is not None:
            reduced = self.scaler.transform(reduced)
        else:
            # Fallback: normalize to [-1, 1]
            reduced = 2 * (reduced - reduced.min(axis=0)) / (reduced.max(axis=0) - reduced.min(axis=0) + 1e-9) - 1

        return reduced.astype(np.float32)

    def prepare_qml_features(
        self,
        h_embeddings: np.ndarray,
        t_embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        fit: bool = False
    ) -> np.ndarray:
        """
        Prepare QML features from head/tail embeddings using selected encoding strategy.

        Args:
            h_embeddings: Head embeddings [N, D]
            t_embeddings: Tail embeddings [N, D]
            labels: Labels for supervised feature selection (optional, only used if fit=True)
            fit: Whether to fit the reduction (True for training, False for test)

        Returns:
            QML features [N, num_qubits]
        """
        # Pre-check: Validate embeddings have sufficient variance
        if fit:
            h_std = np.std(h_embeddings, axis=0)
            t_std = np.std(t_embeddings, axis=0)
            h_nonzero_std = np.sum(h_std > 1e-10)
            t_nonzero_std = np.sum(t_std > 1e-10)
            
            if h_nonzero_std == 0 or t_nonzero_std == 0:
                logger.error(f"Embeddings have zero variance! Head: {h_nonzero_std}/{len(h_std)} non-zero std, "
                            f"Tail: {t_nonzero_std}/{len(t_std)} non-zero std")
                logger.error(f"Head embeddings shape: {h_embeddings.shape}, "
                            f"unique rows: {len(np.unique(h_embeddings, axis=0))}/{len(h_embeddings)}")
                logger.error(f"Tail embeddings shape: {t_embeddings.shape}, "
                            f"unique rows: {len(np.unique(t_embeddings, axis=0))}/{len(t_embeddings)}")
                raise ValueError(
                    f"Embeddings have insufficient variance. This will cause all encoding strategies to fail. "
                    f"Check if embeddings are properly loaded and normalized."
                )
        
        # Step 1: Combine embeddings using encoding strategy
        # Use actual strategy if it was set during fit (fallback), otherwise use requested strategy
        encoding_to_use = self.actual_encoding_strategy if self.actual_encoding_strategy is not None else self.encoding_strategy
        if encoding_to_use != self.encoding_strategy:
            # Temporarily switch to the actual strategy
            original_strategy = self.encoding_strategy
            self.encoding_strategy = encoding_to_use
            combined = self._combine_embeddings(h_embeddings, t_embeddings)
            self.encoding_strategy = original_strategy  # Restore original
        else:
            combined = self._combine_embeddings(h_embeddings, t_embeddings)

        # Step 2: Reduce to quantum dimension
        if fit:
            try:
                self.fit_reduction(combined, labels)
                # If original strategy worked, store it as the actual strategy
                if self.actual_encoding_strategy is None:
                    self.actual_encoding_strategy = self.encoding_strategy
            except ValueError as e:
                # If all features are constant, try fallback encoding strategies
                if "constant" in str(e).lower() or "zero variance" in str(e).lower():
                    logger.warning(f"Encoding strategy '{self.encoding_strategy}' produced constant features.")
                    
                    # Add diagnostics about embeddings
                    h_unique = len(np.unique(h_embeddings, axis=0))
                    t_unique = len(np.unique(t_embeddings, axis=0))
                    h_std = np.std(h_embeddings, axis=0)
                    t_std = np.std(t_embeddings, axis=0)
                    logger.info(f"Embedding diagnostics: Head unique={h_unique}/{len(h_embeddings)}, "
                               f"Tail unique={t_unique}/{len(t_embeddings)}, "
                               f"Head std range=[{h_std.min():.6f}, {h_std.max():.6f}], "
                               f"Tail std range=[{t_std.min():.6f}, {t_std.max():.6f}]")
                    
                    # Try fallback strategies in order (excluding the one that failed)
                    # Use 'hybrid' first as it combines multiple strategies and is more robust
                    fallback_strategies = ['hybrid', 'optimized_diff', 'phase', 'amplitude']
                    original_strategy = self.encoding_strategy
                    fallback_success = False
                    
                    for fallback_strategy in fallback_strategies:
                        if fallback_strategy == original_strategy:
                            continue
                        try:
                            logger.info(f"Trying fallback encoding: {fallback_strategy}")
                            # Temporarily use fallback strategy
                            temp_strategy = self.encoding_strategy
                            self.encoding_strategy = fallback_strategy
                            combined_fallback = self._combine_embeddings(h_embeddings, t_embeddings)
                            self.encoding_strategy = temp_strategy  # Restore immediately
                            
                            # Check if this produces non-constant features
                            col_std = np.std(combined_fallback, axis=0)
                            non_constant_count = np.sum(col_std > 1e-10)
                            if non_constant_count > 0:
                                logger.info(f"✓ Fallback encoding '{fallback_strategy}' produced {non_constant_count}/{len(col_std)} non-constant features")
                                self.fit_reduction(combined_fallback, labels)
                                self.actual_encoding_strategy = fallback_strategy  # Store the actual strategy used
                                combined = combined_fallback  # Use the fallback combined features
                                fallback_success = True
                                break
                            else:
                                logger.warning(f"Fallback '{fallback_strategy}' also produced constant features, trying next...")
                        except Exception as fallback_error:
                            logger.warning(f"Fallback '{fallback_strategy}' failed: {fallback_error}")
                            continue
                    
                    if not fallback_success:
                        # All fallbacks failed - use raw concatenation as last resort
                        logger.error("All encoding strategies failed. Using raw concatenation fallback...")
                        logger.warning("This may result in poor performance. Check embedding quality.")
                        
                        # Use simple concatenation: [h, t] then truncate/pad to num_qubits
                        combined_raw = np.concatenate([h_embeddings, t_embeddings], axis=1)
                        
                        # Check if raw concatenation has variance
                        col_std_raw = np.std(combined_raw, axis=0)
                        non_constant_raw = np.sum(col_std_raw > 1e-10)
                        if non_constant_raw == 0:
                            # Final diagnostic before raising
                            h_unique = len(np.unique(h_embeddings, axis=0))
                            t_unique = len(np.unique(t_embeddings, axis=0))
                            h_mean_std = np.mean(np.std(h_embeddings, axis=0))
                            t_mean_std = np.mean(np.std(t_embeddings, axis=0))
                            raise ValueError(
                                f"Even raw concatenation produced constant features. "
                                f"This indicates embeddings are identical or nearly identical. "
                                f"Head embeddings: {h_unique} unique out of {len(h_embeddings)} "
                                f"(mean std: {h_mean_std:.6f}), "
                                f"Tail embeddings: {t_unique} unique out of {len(t_embeddings)} "
                                f"(mean std: {t_mean_std:.6f}). "
                                f"Original error: {e}"
                            )
                        
                        # Use raw concatenation with simple PCA
                        logger.info(f"Using raw concatenation: {combined_raw.shape} -> {self.num_qubits}D "
                                   f"({non_constant_raw}/{len(col_std_raw)} non-constant features)")
                        combined = combined_raw
                        
                        # Fit simple reduction on raw concatenation
                        # Remove constant columns first
                        col_std = np.std(combined, axis=0)
                        constant_mask = col_std < 1e-10
                        if constant_mask.any():
                            logger.warning(f"Removing {constant_mask.sum()} constant columns from raw concatenation")
                            combined = combined[:, ~constant_mask]
                        
                        if combined.shape[1] == 0:
                            raise ValueError("All features are constant even after raw concatenation fallback.")
                        
                        # Fit reduction
                        from sklearn.preprocessing import StandardScaler
                        self.feature_scaler = StandardScaler()
                        combined = self.feature_scaler.fit_transform(combined)
                        
                        # Simple PCA
                        n_components = min(self.num_qubits, combined.shape[0], combined.shape[1])
                        self.pca = PCA(n_components=n_components, random_state=self.random_state)
                        self.pca.fit(combined)
                        
                        # Fit scaler
                        self.scaler = MinMaxScaler(feature_range=(-1, 1))
                        reduced_temp = self.pca.transform(combined)
                        self.scaler.fit(reduced_temp)
                        
                        self.actual_encoding_strategy = 'raw_concat_fallback'
                        logger.warning("✓ Using raw concatenation fallback (performance may be degraded)")
                else:
                    # Re-raise if it's a different error
                    raise

        qml_features = self.transform_to_qml_space(combined)

        return qml_features

    def _combine_embeddings(
        self,
        h_embeddings: np.ndarray,
        t_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Combine head/tail embeddings using selected encoding strategy.

        Args:
            h_embeddings: Head embeddings [N, D]
            t_embeddings: Tail embeddings [N, D]

        Returns:
            Combined features [N, feature_dim]
        """
        if self.encoding_strategy == 'amplitude':
            # Tensor product (Kronecker) for amplitude encoding
            # Note: This can explode dimensionality, so we take top components
            combined = []
            for h, t in zip(h_embeddings, t_embeddings):
                # Normalize first
                h_norm = h / (np.linalg.norm(h) + 1e-9)
                t_norm = t / (np.linalg.norm(t) + 1e-9)
                # Tensor product
                tensor = np.kron(h_norm, t_norm)
                combined.append(tensor)
            combined = np.array(combined)

        elif self.encoding_strategy == 'phase':
            # Phase encoding: use angles from h and t
            # arctan2(h, t) gives relative phase
            combined = []
            for h, t in zip(h_embeddings, t_embeddings):
                # Take arctan2 for phase information
                phases = np.arctan2(h, t + 1e-9)
                combined.append(phases)
            combined = np.array(combined)

        elif self.encoding_strategy == 'hybrid':
            # Hybrid: combine multiple strategies
            # [|h-t|, h*t, h+t]
            diff = np.abs(h_embeddings - t_embeddings)
            prod = h_embeddings * t_embeddings
            summ = h_embeddings + t_embeddings
            combined = np.concatenate([diff, prod, summ], axis=1)

        elif self.encoding_strategy == 'optimized_diff':
            # Optimized difference with non-linear transformations
            diff = h_embeddings - t_embeddings
            abs_diff = np.abs(diff)
            squared_diff = diff ** 2
            combined = np.concatenate([abs_diff, squared_diff], axis=1)

        elif self.encoding_strategy == 'tensor_product':
            # Element-wise product + concatenation
            prod = h_embeddings * t_embeddings
            combined = np.concatenate([h_embeddings, t_embeddings, prod], axis=1)

        elif self.encoding_strategy == 'raw_concat_fallback':
            # Raw concatenation fallback (used when all other strategies fail)
            combined = np.concatenate([h_embeddings, t_embeddings], axis=1)

        else:
            raise ValueError(f"Unknown encoding strategy: {self.encoding_strategy}")

        return combined


def compare_encoding_strategies(
    h_embeddings: np.ndarray,
    t_embeddings: np.ndarray,
    labels: np.ndarray,
    num_qubits: int = 5,
    strategies: List[EncodingStrategy] = ['amplitude', 'phase', 'hybrid', 'optimized_diff', 'tensor_product']
) -> dict:
    """
    Compare different encoding strategies on feature quality metrics.

    Args:
        h_embeddings: Head embeddings
        t_embeddings: Tail embeddings
        labels: Labels for supervised metrics
        num_qubits: Target quantum dimension
        strategies: List of strategies to compare

    Returns:
        Dictionary with comparison results
    """
    results = {}

    for strategy in strategies:
        logger.info(f"Testing encoding strategy: {strategy}")

        engineer = QuantumFeatureEngineer(
            num_qubits=num_qubits,
            encoding_strategy=strategy,
            feature_selection_method='mutual_info'
        )

        # Prepare features
        qml_features = engineer.prepare_qml_features(
            h_embeddings, t_embeddings, labels, fit=True
        )

        # Compute quality metrics
        metrics = compute_feature_quality(qml_features, labels)
        metrics['strategy'] = strategy
        metrics['explained_variance'] = float(engineer.pca.explained_variance_ratio_.sum()) if engineer.pca else 0.0

        results[strategy] = metrics

        logger.info(f"  Separability: {metrics['class_separability']:.4f}, "
                   f"MI: {metrics['mutual_information']:.4f}")

    return results


def compute_feature_quality(features: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute quality metrics for features.

    Metrics:
    - Class separability (ratio of between/within class variance)
    - Mutual information with labels
    - Feature variance

    Args:
        features: Feature matrix [N, D]
        labels: Labels [N]

    Returns:
        Dictionary with quality metrics
    """
    from sklearn.metrics import mutual_info_score

    metrics = {}

    # Class separability (Fisher criterion)
    classes = np.unique(labels)
    if len(classes) == 2:
        class_means = []
        class_vars = []
        for c in classes:
            class_feats = features[labels == c]
            class_means.append(np.mean(class_feats, axis=0))
            class_vars.append(np.var(class_feats, axis=0))

        between_class_var = np.var(class_means, axis=0)
        within_class_var = np.mean(class_vars, axis=0)

        # Fisher criterion (high is better)
        separability = np.mean(between_class_var / (within_class_var + 1e-9))
        metrics['class_separability'] = float(separability)
    else:
        metrics['class_separability'] = 0.0

    # Mutual information
    mi_scores = []
    for i in range(features.shape[1]):
        # Discretize feature for MI calculation
        feat_disc = np.digitize(features[:, i], bins=10)
        mi = mutual_info_score(labels, feat_disc)
        mi_scores.append(mi)
    metrics['mutual_information'] = float(np.mean(mi_scores))

    # Feature variance
    metrics['feature_variance'] = float(np.mean(np.var(features, axis=0)))

    return metrics


def optimize_quantum_reduction(
    h_embeddings: np.ndarray,
    t_embeddings: np.ndarray,
    labels: np.ndarray,
    num_qubits_range: List[int] = [4, 5, 6, 8],
    encoding_strategy: EncodingStrategy = 'hybrid'
) -> Tuple[int, dict]:
    """
    Find optimal number of qubits by testing different dimensions.

    Args:
        h_embeddings: Head embeddings
        t_embeddings: Tail embeddings
        labels: Labels
        num_qubits_range: Range of qubit numbers to test
        encoding_strategy: Encoding strategy to use

    Returns:
        (best_num_qubits, results_dict)
    """
    results = {}
    best_score = -np.inf
    best_num_qubits = num_qubits_range[0]

    for num_qubits in num_qubits_range:
        logger.info(f"Testing num_qubits={num_qubits}")

        engineer = QuantumFeatureEngineer(
            num_qubits=num_qubits,
            encoding_strategy=encoding_strategy,
            feature_selection_method='mutual_info'
        )

        qml_features = engineer.prepare_qml_features(
            h_embeddings, t_embeddings, labels, fit=True
        )

        metrics = compute_feature_quality(qml_features, labels)
        metrics['num_qubits'] = num_qubits

        # Score: combination of separability and MI
        score = metrics['class_separability'] + metrics['mutual_information']
        metrics['score'] = score

        results[num_qubits] = metrics

        if score > best_score:
            best_score = score
            best_num_qubits = num_qubits

        logger.info(f"  Score: {score:.4f} (sep={metrics['class_separability']:.4f}, mi={metrics['mutual_information']:.4f})")

    logger.info(f"\nBest num_qubits: {best_num_qubits} (score={best_score:.4f})")

    return best_num_qubits, results
