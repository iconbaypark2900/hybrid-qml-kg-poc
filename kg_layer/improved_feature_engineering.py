"""
Improved Feature Engineering with RandomForest-Guided Features

Creates interaction features and uses RandomForest importances to guide
feature construction, emphasizing class differences.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


class ImprovedFeatureEngineer:
    """
    Enhanced feature engineering that emphasizes class separability.
    
    Uses RandomForest importances to guide feature construction and
    creates interaction features that maximize class differences.
    """
    
    def __init__(
        self,
        use_rf_guidance: bool = True,
        max_interaction_features: int = 50,
        interaction_top_k: int = 20,
        use_domain_features: bool = False,
        random_state: int = 42
    ):
        """
        Args:
            use_rf_guidance: Use RandomForest to guide feature selection
            max_interaction_features: Maximum number of interaction features to create
            interaction_top_k: Use top K features for interactions
            use_domain_features: Add domain knowledge features (compound properties, disease categories)
            random_state: Random seed
        """
        self.use_rf_guidance = use_rf_guidance
        self.max_interaction_features = max_interaction_features
        self.interaction_top_k = interaction_top_k
        self.use_domain_features = use_domain_features
        self.random_state = random_state
        
        self.rf_model: Optional[RandomForestClassifier] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.selected_features_: Optional[List[int]] = None
    
    def create_interaction_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create interaction features that emphasize class differences.
        
        Creates:
        - Product features (x_i * x_j) for top important features
        - Ratio features (x_i / (x_j + eps)) for top important features
        - Difference features (|x_i - x_j|) for top important features
        - Polynomial features (x_i^2) for top important features
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            feature_names: Optional feature names
        
        Returns:
            Enhanced feature matrix and feature names
        """
        logger.info(f"Creating interaction features from {X.shape[1]} base features...")
        
        # Step 1: Identify top features using RandomForest
        if self.use_rf_guidance and X.shape[1] > 10:
            logger.info("  Using RandomForest to identify important features...")
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X, y)
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-self.interaction_top_k:][::-1]
            
            self.rf_model = rf
            self.feature_importances_ = importances
            
            logger.info(f"  Selected top {len(top_indices)} features for interactions")
            logger.info(f"  Top 5 feature importances: {importances[top_indices[:5]]}")
        else:
            # Use all features if RF guidance disabled or too few features
            top_indices = np.arange(X.shape[1])
        
        self.selected_features_ = top_indices.tolist()
        
        # Step 2: Create interaction features
        interaction_features = []
        interaction_names = []
        
        eps = 1e-8
        
        # Product features (x_i * x_j) - emphasizes correlations
        logger.info("  Creating product features...")
        count = 0
        for i, idx_i in enumerate(top_indices[:min(10, len(top_indices))]):
            for j, idx_j in enumerate(top_indices[i+1:min(10, len(top_indices))], start=i+1):
                if count >= self.max_interaction_features:
                    break
                product = X[:, idx_i] * X[:, idx_j]
                interaction_features.append(product)
                name_i = feature_names[idx_i] if feature_names else f"f{idx_i}"
                name_j = feature_names[idx_j] if feature_names else f"f{idx_j}"
                interaction_names.append(f"{name_i}*{name_j}")
                count += 1
        
        # Ratio features (x_i / (x_j + eps)) - emphasizes relative differences
        logger.info("  Creating ratio features...")
        for i, idx_i in enumerate(top_indices[:min(8, len(top_indices))]):
            for j, idx_j in enumerate(top_indices[:min(8, len(top_indices))]):
                if idx_i == idx_j or count >= self.max_interaction_features:
                    continue
                ratio = X[:, idx_i] / (X[:, idx_j] + eps)
                interaction_features.append(ratio)
                name_i = feature_names[idx_i] if feature_names else f"f{idx_i}"
                name_j = feature_names[idx_j] if feature_names else f"f{idx_j}"
                interaction_names.append(f"{name_i}/{name_j}")
                count += 1
        
        # Difference features (|x_i - x_j|) - emphasizes absolute differences
        logger.info("  Creating difference features...")
        for i, idx_i in enumerate(top_indices[:min(8, len(top_indices))]):
            for j, idx_j in enumerate(top_indices[i+1:min(8, len(top_indices))], start=i+1):
                if count >= self.max_interaction_features:
                    break
                diff = np.abs(X[:, idx_i] - X[:, idx_j])
                interaction_features.append(diff)
                name_i = feature_names[idx_i] if feature_names else f"f{idx_i}"
                name_j = feature_names[idx_j] if feature_names else f"f{idx_j}"
                interaction_names.append(f"{name_i}-{name_j}")
                count += 1
        
        # Polynomial features (x_i^2) - emphasizes non-linear patterns
        logger.info("  Creating polynomial features...")
        for idx in top_indices[:min(10, len(top_indices))]:
            if count >= self.max_interaction_features:
                break
            squared = X[:, idx] ** 2
            interaction_features.append(squared)
            name = feature_names[idx] if feature_names else f"f{idx}"
            interaction_names.append(f"{name}^2")
            count += 1
        
        # Combine original and interaction features
        if interaction_features:
            X_interactions = np.column_stack(interaction_features)
            X_enhanced = np.hstack([X, X_interactions])
            
            all_names = (feature_names or [f"f{i}" for i in range(X.shape[1])]) + interaction_names
            
            logger.info(f"  Created {len(interaction_features)} interaction features")
            logger.info(f"  Total features: {X.shape[1]} → {X_enhanced.shape[1]}")
            
            return X_enhanced, all_names
        else:
            return X, feature_names or [f"f{i}" for i in range(X.shape[1])]
    
    def select_class_separable_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: Optional[int] = None,
        method: str = 'f_classif'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select features that best separate classes.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            k: Number of features to select (None = use all significant)
            method: Selection method ('f_classif' or 'rf_importance')
        
        Returns:
            Selected features and boolean mask
        """
        if k is None:
            k = min(X.shape[1], X.shape[0] // 10)  # Heuristic
        
        logger.info(f"Selecting top {k} class-separable features using {method}...")
        
        if method == 'rf_importance' and self.rf_model is not None:
            # Use RandomForest importances
            importances = self.feature_importances_
            top_k_indices = np.argsort(importances)[-k:][::-1]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_k_indices] = True
        else:
            # Use F-test (ANOVA F-value)
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            mask = selector.get_support()
            top_k_indices = np.where(mask)[0]
        
        logger.info(f"  Selected {np.sum(mask)} features")
        logger.info(f"  Feature indices: {top_k_indices[:min(10, len(top_k_indices))]}")
        
        return X[:, mask], mask
    
    def create_class_difference_features(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Create features that explicitly encode class differences.
        
        For each feature, creates:
        - Distance to positive class centroid
        - Distance to negative class centroid
        - Ratio of distances
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
        
        Returns:
            Enhanced feature matrix with class difference features
        """
        logger.info("Creating class difference features...")
        
        pos_mask = y == 1
        neg_mask = y == 0
        
        if np.sum(pos_mask) == 0 or np.sum(neg_mask) == 0:
            logger.warning("  Cannot create class difference features: missing classes")
            return X
        
        # Compute class centroids
        pos_centroid = np.mean(X[pos_mask], axis=0)
        neg_centroid = np.mean(X[neg_mask], axis=0)
        
        # Distance to centroids
        pos_distances = np.linalg.norm(X - pos_centroid, axis=1, keepdims=True)
        neg_distances = np.linalg.norm(X - neg_centroid, axis=1, keepdims=True)
        
        # Ratio (emphasizes which class is closer)
        ratio = pos_distances / (neg_distances + 1e-8)
        
        # Difference (emphasizes separation)
        diff = np.abs(pos_distances - neg_distances)
        
        # Combine
        class_diff_features = np.hstack([pos_distances, neg_distances, ratio, diff])
        
        X_enhanced = np.hstack([X, class_diff_features])
        
        logger.info(f"  Added {class_diff_features.shape[1]} class difference features")
        logger.info(f"  Total features: {X.shape[1]} → {X_enhanced.shape[1]}")
        
        return X_enhanced
    
    def create_domain_features(
        self,
        X: np.ndarray,
        entity_ids: Optional[List[str]] = None,
        head_ids: Optional[np.ndarray] = None,
        tail_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create domain knowledge features based on entity types and properties.
        
        For Hetionet CtD task:
        - Compound features: Extract compound properties (if available)
        - Disease features: Extract disease categories (if available)
        - Entity type indicators: Binary features for compound/disease types
        
        Args:
            X: Feature matrix [n_samples, n_features]
            entity_ids: Optional list of entity IDs (for extracting domain info)
            head_ids: Optional head entity IDs [n_samples]
            tail_ids: Optional tail entity IDs [n_samples]
        
        Returns:
            Enhanced feature matrix with domain features
        """
        if not self.use_domain_features:
            return X
        
        logger.info("Creating domain knowledge features...")
        
        domain_features = []
        
        if head_ids is not None and tail_ids is not None:
            # Extract entity type information from IDs
            # Hetionet format: "Compound::DB00001" or "Disease::DOID:1234"
            
            # Compound type features (from head entities)
            compound_types = []
            for head_id in head_ids:
                if isinstance(head_id, str) and '::' in head_id:
                    entity_type = head_id.split('::')[0]
                    compound_types.append(1.0 if entity_type == 'Compound' else 0.0)
                else:
                    compound_types.append(0.0)
            
            # Disease type features (from tail entities)
            disease_types = []
            for tail_id in tail_ids:
                if isinstance(tail_id, str) and '::' in tail_id:
                    entity_type = tail_id.split('::')[0]
                    disease_types.append(1.0 if entity_type == 'Disease' else 0.0)
                else:
                    disease_types.append(0.0)
            
            domain_features.append(np.array(compound_types))
            domain_features.append(np.array(disease_types))
            
            # Compound-Disease interaction indicator
            interaction_indicator = np.array(compound_types) * np.array(disease_types)
            domain_features.append(interaction_indicator)
            
            # Extract numeric IDs (if available) for potential ordering/grouping
            compound_numeric_ids = []
            disease_numeric_ids = []
            
            for head_id in head_ids:
                if isinstance(head_id, str) and '::' in head_id:
                    parts = head_id.split('::')
                    if len(parts) > 1:
                        # Try to extract numeric part
                        numeric_part = ''.join(filter(str.isdigit, parts[1]))
                        compound_numeric_ids.append(float(numeric_part) if numeric_part else 0.0)
                    else:
                        compound_numeric_ids.append(0.0)
                else:
                    compound_numeric_ids.append(0.0)
            
            for tail_id in tail_ids:
                if isinstance(tail_id, str) and '::' in tail_id:
                    parts = tail_id.split('::')
                    if len(parts) > 1:
                        # Try to extract numeric part
                        numeric_part = ''.join(filter(str.isdigit, parts[1]))
                        disease_numeric_ids.append(float(numeric_part) if numeric_part else 0.0)
                    else:
                        disease_numeric_ids.append(0.0)
                else:
                    disease_numeric_ids.append(0.0)
            
            # Normalize numeric IDs (for potential grouping effects)
            if len(compound_numeric_ids) > 0:
                compound_numeric_array = np.array(compound_numeric_ids)
                if compound_numeric_array.std() > 0:
                    compound_numeric_array = (compound_numeric_array - compound_numeric_array.mean()) / (compound_numeric_array.std() + 1e-8)
                domain_features.append(compound_numeric_array)
            
            if len(disease_numeric_ids) > 0:
                disease_numeric_array = np.array(disease_numeric_ids)
                if disease_numeric_array.std() > 0:
                    disease_numeric_array = (disease_numeric_array - disease_numeric_array.mean()) / (disease_numeric_array.std() + 1e-8)
                domain_features.append(disease_numeric_array)
        
        if domain_features:
            domain_feature_matrix = np.column_stack(domain_features)
            X_enhanced = np.hstack([X, domain_feature_matrix])
            
            logger.info(f"  Added {domain_feature_matrix.shape[1]} domain knowledge features")
            logger.info(f"  Total features: {X.shape[1]} → {X_enhanced.shape[1]}")
            
            return X_enhanced
        else:
            logger.warning("  Could not create domain features (missing entity IDs)")
            return X

