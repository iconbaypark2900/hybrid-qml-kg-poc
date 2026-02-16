"""
Enhanced Cure Prediction Framework with Improved Performance

This module implements several performance enhancements:
1. Advanced feature engineering
2. Hyperparameter optimization
3. Ensemble methods
4. Better preprocessing
5. Quantum feature engineering
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Import optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.kg_embedder import HetionetEmbedder
from kg_layer.kg_visualizer import KGVisualizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def compute_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["pr_auc"] = float("nan")

    return metrics

logger = logging.getLogger(__name__)


class EnhancedCurePredictionFramework:
    """
    Enhanced framework for predicting potential cures with improved performance.
    """

    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.task_edges = None
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.embeddings = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.models = {}
        self.results = {}
        
        os.makedirs(results_dir, exist_ok=True)

    def load_knowledge_graph(self, relation_type: str = "CtD", max_entities: Optional[int] = None):
        """
        Load the knowledge graph for cure prediction.
        """
        logger.info(f"Loading knowledge graph for relation: {relation_type}")
        
        # Load Hetionet edges
        df_edges = load_hetionet_edges(data_dir=self.data_dir)
        
        # Extract task-specific edges
        self.task_edges, self.entity_to_id, self.id_to_entity = extract_task_edges(
            df_edges, 
            relation_type=relation_type, 
            max_entities=max_entities
        )
        
        logger.info(f"Loaded {len(self.task_edges)} {relation_type} edges with {len(self.entity_to_id)} entities")

    def train_embeddings(self, method: str = "ComplEx", embedding_dim: int = 128, epochs: int = 100):
        """
        Train knowledge graph embeddings with enhanced methods.
        Falls back to deterministic embeddings if PyKEEN is not available.
        """
        logger.info(f"Training {method} embeddings (dim={embedding_dim}, epochs={epochs})")
        
        # Check if PyKEEN is available
        try:
            from kg_layer.advanced_embeddings import PYKEEN_AVAILABLE
            if PYKEEN_AVAILABLE:
                # Use AdvancedKGEmbedder for state-of-the-art embeddings
                from kg_layer.advanced_embeddings import AdvancedKGEmbedder
                embedder = AdvancedKGEmbedder(
                    embedding_dim=embedding_dim,
                    method=method,
                    num_epochs=epochs,
                    work_dir=self.data_dir
                )
                
                # Prepare training data
                train_data = self.task_edges[['source', 'metaedge', 'target']].copy()
                train_data.columns = ['head', 'relation', 'tail']
                
                # Train embeddings
                metrics = embedder.train_embeddings(train_data)
                self.embeddings = embedder.entity_embeddings
                
                logger.info(f"Embedding training completed. Metrics: {metrics}")
                return metrics
            else:
                logger.warning(f"PyKEEN not available, using deterministic fallback embeddings")
        except ImportError:
            logger.warning(f"PyKEEN not available, using deterministic fallback embeddings")
        
        # Fallback to deterministic embeddings
        from kg_layer.kg_embedder import HetionetEmbedder
        embedder = HetionetEmbedder(
            embedding_dim=embedding_dim,
            work_dir=self.data_dir
        )
        
        # Prepare training data
        train_data = self.task_edges[['source', 'target']].copy()
        train_data.columns = ['source', 'target']
        
        # Add dummy relation column to match expected format
        train_data['metaedge'] = 'treats'
        
        # Train embeddings (this will use deterministic fallback if PyKEEN not available)
        embedder.train_embeddings(train_data)
        
        self.embeddings = embedder.entity_embeddings
        self.entity_to_id = embedder.entity_to_id
        self.id_to_entity = embedder.id_to_entity
        
        logger.info(f"Embedding training completed. Shape: {self.embeddings.shape}")
        return {"method": "deterministic_fallback", "embedding_dim": embedding_dim, "shape": self.embeddings.shape}

    def prepare_link_prediction_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Prepare link prediction dataset with enhanced negative sampling.
        """
        logger.info(f"Preparing link prediction dataset (test_size={test_size})")
        
        # Prepare training dataset with negative samples
        train_df, test_df = prepare_link_prediction_dataset(
            self.task_edges,
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"Dataset prepared: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df

    def extract_advanced_pair_features(self, pairs_df: pd.DataFrame, feature_type: str = "combined"):
        """
        Extract enhanced features for (compound, disease) pairs.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not trained. Call train_embeddings() first.")
        
        features = []
        
        for _, row in pairs_df.iterrows():
            source_id = int(row['source_id'])
            target_id = int(row['target_id'])
            
            # Get embeddings for source and target
            source_emb = self.embeddings[source_id]
            target_emb = self.embeddings[target_id]
            
            # Calculate multiple types of features
            concat_feat = np.concatenate([source_emb, target_emb])
            diff_feat = np.abs(source_emb - target_emb)
            hadamard_feat = source_emb * target_emb
            l2_dist = np.linalg.norm(source_emb - target_emb)
            cosine_sim = np.dot(source_emb, target_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8)
            jaccard_coeff = np.sum(np.minimum(source_emb, target_emb)) / np.sum(np.maximum(source_emb, target_emb)) + 1e-8
            
            # Combine all features
            all_features = np.concatenate([
                concat_feat, 
                diff_feat, 
                hadamard_feat, 
                [l2_dist, cosine_sim, jaccard_coeff]
            ])
            
            features.append(all_features)
        
        return np.array(features)

    def optimize_hyperparameters(self, X_train, y_train, model_name):
        """
        Perform hyperparameter optimization for a given model.
        """
        logger.info(f"Optimizing hyperparameters for {model_name}")
        
        if model_name == "LogisticRegression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        elif model_name == "RandomForest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
        else:
            return None  # Return None if model not supported
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,  # Reduced for faster execution
            scoring='average_precision',  # PR-AUC equivalent
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

    def train_enhanced_classical_models(self, X_train, y_train, X_test, y_test):
        """
        Train enhanced classical models with hyperparameter optimization.
        """
        logger.info("Training enhanced classical models with hyperparameter optimization...")
        
        # Define models to optimize (only include available ones)
        model_configs = [
            ("LogisticRegression", "LogisticRegression"),
            ("RandomForest", "RandomForest"),
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            model_configs.append(("XGBoost", "XGBoost"))
        
        for model_name, config_name in model_configs:
            logger.info(f"Optimizing {model_name}...")
            
            # Optimize hyperparameters
            best_model = self.optimize_hyperparameters(X_train, y_train, model_name)
            
            if best_model is not None:
                # Make predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Compute metrics
                metrics = compute_metrics(y_test, y_pred, y_pred_proba)
                
                # Store results
                self.models[config_name] = best_model
                self.results[f'enhanced_classical_{config_name}'] = {
                    'model': best_model,
                    'test_metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{config_name} - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        # Create ensemble model
        logger.info("Creating ensemble model...")
        ensemble_models = []
        for name, model in self.models.items():
            if name in ["LogisticRegression", "RandomForest", "XGBoost"]:
                ensemble_models.append((name, model))
        
        if len(ensemble_models) > 1:
            ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
            
            # Fit ensemble on training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Fit individual models first
            for name, model in ensemble_models:
                if hasattr(model.named_steps['classifier'], 'predict_proba'):
                    model.fit(X_train_scaled, y_train)
            
            # Create ensemble predictions
            ensemble_preds = []
            for name, model in ensemble_models:
                pred = model.predict_proba(X_test_scaled)[:, 1]
                ensemble_preds.append(pred)
            
            # Average the predictions
            avg_pred_proba = np.mean(ensemble_preds, axis=0)
            avg_pred = (avg_pred_proba >= 0.5).astype(int)
            
            # Compute ensemble metrics
            ensemble_metrics = compute_metrics(y_test, avg_pred, avg_pred_proba)
            
            self.results['enhanced_classical_ensemble'] = {
                'model': ensemble_models,  # Store individual models since ensemble is just averaging
                'test_metrics': ensemble_metrics,
                'predictions': avg_pred,
                'probabilities': avg_pred_proba
            }
            
            logger.info(f"Ensemble - Test PR-AUC: {ensemble_metrics['pr_auc']:.4f}, Accuracy: {ensemble_metrics['accuracy']:.4f}")

    def train_quantum_ready_models(self, X_train, y_train, X_test, y_test, num_qubits: int = 10):
        """
        Prepare quantum-ready features and train models that can interface with quantum systems.
        """
        logger.info(f"Preparing quantum-ready features with {num_qubits} qubits...")
        
        # Reduce dimensions to match number of qubits using PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=num_qubits, random_state=42)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        
        # Normalize to quantum-safe range
        X_train_reduced = np.tanh(X_train_reduced)
        X_test_reduced = np.tanh(X_test_reduced)
        
        # Train classical models on quantum-ready features for comparison
        classical_on_quantum_features = {
            'QuantumReady_LR': LogisticRegression(random_state=42, max_iter=1000),
            'QuantumReady_RF': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in classical_on_quantum_features.items():
            logger.info(f"Training {name} on quantum-ready features...")
            
            # Fit model
            model.fit(X_train_reduced, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_reduced)
            y_pred_proba = model.predict_proba(X_test_reduced)[:, 1]
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[f'quantum_ready_{name}'] = {
                'model': model,
                'test_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def run_enhanced_simulation(self, 
                               relation_type: str = "CtD", 
                               max_entities: Optional[int] = None,
                               embedding_method: str = "ComplEx",
                               embedding_dim: int = 128,
                               test_size: float = 0.2,
                               num_qubits: int = 10):
        """
        Run the enhanced simulation for cure prediction.
        """
        logger.info("Starting enhanced cure prediction simulation...")
        
        # Step 1: Load knowledge graph
        self.load_knowledge_graph(relation_type=relation_type, max_entities=max_entities)
        
        # Step 2: Train embeddings
        self.train_embeddings(method=embedding_method, embedding_dim=embedding_dim)
        
        # Step 3: Prepare data
        train_df, test_df = self.prepare_link_prediction_data(test_size=test_size)
        
        # Step 4: Extract enhanced features
        X_train = self.extract_advanced_pair_features(train_df, feature_type="combined")
        X_test = self.extract_advanced_pair_features(test_df, feature_type="combined")
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        logger.info(f"Enhanced feature extraction completed: X_train shape = {X_train.shape}")
        
        # Step 5: Train enhanced classical models
        self.train_enhanced_classical_models(X_train, y_train, X_test, y_test)
        
        # Step 6: Train quantum-ready models
        self.train_quantum_ready_models(X_train, y_train, X_test, y_test, num_qubits=num_qubits)
        
        # Step 7: Compile results
        self.compile_results()
        
        logger.info("Enhanced simulation completed successfully!")
        return self.results

    def compile_results(self):
        """
        Compile and summarize all results.
        """
        logger.info("Compiling enhanced results...")
        
        summary = {}
        
        for key, result in self.results.items():
            if 'test_metrics' in result and result['test_metrics']:
                summary[key] = {
                    'pr_auc': result['test_metrics'].get('pr_auc', 0.0),
                    'accuracy': result['test_metrics'].get('accuracy', 0.0),
                    'precision': result['test_metrics'].get('precision', 0.0),
                    'recall': result['test_metrics'].get('recall', 0.0),
                    'f1': result['test_metrics'].get('f1', 0.0)
                }
        
        # Sort by PR-AUC
        sorted_results = sorted(summary.items(), key=lambda x: x[1]['pr_auc'], reverse=True)
        
        logger.info("Top performing models by PR-AUC:")
        for model_name, metrics in sorted_results[:10]:  # Show top 10
            logger.info(f"  {model_name}: PR-AUC = {metrics['pr_auc']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
        
        self.results['summary'] = summary
        self.results['sorted_by_pr_auc'] = sorted_results

    def predict_cures(self, compounds: List[str], diseases: List[str], top_k: int = 10):
        """
        Predict potential cures for given compounds and diseases using the best model.
        """
        if not self.models:
            raise ValueError("No models trained. Run simulation first.")
        
        # Get the best performing model
        best_model_name = self.results['sorted_by_pr_auc'][0][0] if self.results.get('sorted_by_pr_auc') else None
        
        if not best_model_name:
            raise ValueError("No trained models found in results.")
        
        # Handle ensemble case
        if 'ensemble' in best_model_name.lower():
            # For ensemble, we need to use the individual models
            best_result = self.results[best_model_name]
            probabilities = best_result['probabilities']
        else:
            best_model = self.results[best_model_name]['model']
            
            # Create all possible pairs
            pairs = []
            for comp in compounds:
                for disease in diseases:
                    if comp in self.entity_to_id and disease in self.entity_to_id:
                        pairs.append({
                            'compound': comp,
                            'disease': disease,
                            'compound_id': self.entity_to_id[comp],
                            'disease_id': self.entity_to_id[disease]
                        })
            
            if not pairs:
                raise ValueError("No valid compound-disease pairs found.")
            
            # Create DataFrame
            pairs_df = pd.DataFrame(pairs)
            pairs_df = pairs_df.rename(columns={'compound_id': 'source_id', 'disease_id': 'target_id'})
            
            # Extract enhanced features
            X = self.extract_advanced_pair_features(pairs_df, feature_type="combined")
            
            # Make predictions using the best model
            if 'quantum_ready' in best_model_name:
                # For quantum-ready models, we need to transform the features
                from sklearn.decomposition import PCA
                pca = PCA(n_components=10, random_state=42)  # Assuming 10 qubits
                X_reduced = pca.fit_transform(X)
                X_reduced = np.tanh(X_reduced)
                probabilities = best_model.predict_proba(X_reduced)[:, 1]
            else:
                # For classical models, scale the features
                X_scaled = self.scaler.transform(X)
                probabilities = best_model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to DataFrame
        if 'pairs_df' in locals():
            pairs_df['prediction_probability'] = probabilities
            pairs_df['prediction_score'] = probabilities
            
            # Sort by prediction score
            pairs_df = pairs_df.sort_values('prediction_score', ascending=False)
            
            logger.info(f"Predicted top {top_k} potential cures:")
            for i, (_, row) in enumerate(pairs_df.head(top_k).iterrows()):
                logger.info(f"  {i+1}. {row['compound']} -> {row['disease']}: {row['prediction_score']:.4f}")
            
            return pairs_df.head(top_k)
        else:
            # For ensemble case, we need to recreate the pairs
            logger.info(f"Using ensemble model - returning top {top_k} predictions")
            # This is a simplified return for ensemble case
            return pd.DataFrame({'prediction_score': probabilities[:top_k]})

    def get_compound_disease_interactions(self, compound_id: str, top_k: int = 10):
        """
        Get diseases most associated with a specific compound.
        """
        if self.task_edges is None:
            raise ValueError("Knowledge graph not loaded.")
        
        if compound_id not in self.entity_to_id:
            available_compounds = [eid for eid in self.entity_to_id.keys() if eid.startswith("Compound::")]
            raise ValueError(f"Compound {compound_id} not found. Available: {available_compounds[:10]}...")
        
        # Find all edges involving the compound
        compound_idx = self.entity_to_id[compound_id]
        
        # Get all edges where this compound is the source
        related_edges = self.task_edges[
            (self.task_edges['source_id'] == compound_idx) |
            (self.task_edges['target_id'] == compound_idx)
        ].copy()
        
        # Determine which column contains the disease
        if related_edges.empty:
            return pd.DataFrame(columns=['disease', 'relation', 'score'])
        
        # Add entity names
        related_edges['source_entity'] = related_edges['source_id'].map(self.id_to_entity)
        related_edges['target_entity'] = related_edges['target_id'].map(self.id_to_entity)
        
        # Identify diseases (could be in source or target)
        diseases = []
        for _, row in related_edges.iterrows():
            if row['source_entity'] == compound_id and row['target_entity'].startswith('Disease::'):
                diseases.append({
                    'disease': row['target_entity'],
                    'relation': row['metaedge'],
                    'score': 1.0  # Known association
                })
            elif row['target_entity'] == compound_id and row['source_entity'].startswith('Disease::'):
                diseases.append({
                    'disease': row['source_entity'],
                    'relation': row['metaedge'],
                    'score': 1.0  # Known association
                })
        
        df_diseases = pd.DataFrame(diseases)
        
        if not df_diseases.empty:
            df_diseases = df_diseases.sort_values('score', ascending=False).head(top_k)
        
        return df_diseases


def run_enhanced_cure_prediction_pipeline(
    relation_type: str = "CtD",
    max_entities: Optional[int] = None,
    embedding_method: str = "ComplEx",
    embedding_dim: int = 128,
    test_size: float = 0.2,
    num_qubits: int = 10,
    data_dir: str = "data",
    results_dir: str = "results"
):
    """
    Run the enhanced cure prediction pipeline.
    """
    framework = EnhancedCurePredictionFramework(data_dir=data_dir, results_dir=results_dir)
    
    results = framework.run_enhanced_simulation(
        relation_type=relation_type,
        max_entities=max_entities,
        embedding_method=embedding_method,
        embedding_dim=embedding_dim,
        test_size=test_size,
        num_qubits=num_qubits
    )
    
    return framework