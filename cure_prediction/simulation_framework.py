"""
Cure Prediction Simulation Framework

Implements a comprehensive testing framework for predicting potential cures
using the knowledge graph and quantum/classical models.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.kg_embedder import HetionetEmbedder
from kg_layer.kg_visualizer import KGVisualizer
from quantum_layer.qml_model import QMLLinkPredictor
from utils.evaluation import compute_metrics

logger = logging.getLogger(__name__)


class CurePredictionFramework:
    """
    Framework for predicting potential cures using knowledge graph embeddings
    and quantum/classical machine learning models.
    """

    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.task_edges = None
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.embeddings = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        os.makedirs(results_dir, exist_ok=True)

    def load_knowledge_graph(self, relation_type: str = "CtD", max_entities: Optional[int] = None):
        """
        Load the knowledge graph for cure prediction.

        Args:
            relation_type: The relation type to focus on (e.g., "CtD" for compound-treats-disease)
            max_entities: Maximum number of entities to include (for performance)
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
        Train knowledge graph embeddings.

        Args:
            method: Embedding method ("ComplEx", "RotatE", "DistMult", "TransE")
            embedding_dim: Dimension of embeddings
            epochs: Number of training epochs
        """
        logger.info(f"Training {method} embeddings (dim={embedding_dim}, epochs={epochs})")
        
        # Use AdvancedKGEmbedder for state-of-the-art embeddings
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

    def prepare_link_prediction_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Prepare link prediction dataset with positive and negative samples.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
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

    def extract_pair_features(self, pairs_df: pd.DataFrame, feature_type: str = "concat"):
        """
        Extract features for (compound, disease) pairs.

        Args:
            pairs_df: DataFrame with 'source_id' and 'target_id' columns
            feature_type: Type of features to extract ("concat", "diff", "hadamard", "combined")

        Returns:
            Feature matrix X
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
            
            # Extract features based on type
            if feature_type == "concat":
                # Concatenate source and target embeddings
                feat = np.concatenate([source_emb, target_emb])
            elif feature_type == "diff":
                # Difference between embeddings
                feat = np.abs(source_emb - target_emb)
            elif feature_type == "hadamard":
                # Hadamard (element-wise) product
                feat = source_emb * target_emb
            elif feature_type == "combined":
                # Combined features: [concat, diff, hadamard]
                concat_feat = np.concatenate([source_emb, target_emb])
                diff_feat = np.abs(source_emb - target_emb)
                hadamard_feat = source_emb * target_emb
                feat = np.concatenate([concat_feat, diff_feat, hadamard_feat])
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            features.append(feat)
        
        return np.array(features)

    def train_classical_models(self, X_train, y_train, X_test, y_test):
        """
        Train classical machine learning models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
        """
        logger.info("Training classical models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        classical_models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        for name, model in classical_models.items():
            logger.info(f"Training {name}...")
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[f'classical_{name}'] = {
                'model': model,
                'test_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def train_quantum_models(self, X_train, y_train, X_test, y_test, num_qubits: int = 10):
        """
        Train quantum machine learning models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            num_qubits: Number of qubits for quantum models
        """
        logger.info(f"Training quantum models with {num_qubits} qubits...")
        
        # Reduce dimensions to match number of qubits
        from sklearn.decomposition import PCA
        pca = PCA(n_components=num_qubits, random_state=42)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        
        # Normalize to quantum-safe range
        X_train_reduced = np.tanh(X_train_reduced)
        X_test_reduced = np.tanh(X_test_reduced)
        
        quantum_models = {
            'QSVC': QMLLinkPredictor(model_type='QSVC', num_qubits=num_qubits),
            'VQC': QMLLinkPredictor(model_type='VQC', num_qubits=num_qubits)
        }
        
        for name, model in quantum_models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Fit model
                model.fit(X_train_reduced, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_reduced)
                y_pred_proba = model.predict_proba(X_test_reduced)[:, 1]
                
                # Compute metrics
                metrics = compute_metrics(y_test, y_pred, y_pred_proba)
                
                # Store results
                self.models[name] = model
                self.results[f'quantum_{name}'] = {
                    'model': model,
                    'test_metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                self.results[f'quantum_{name}'] = {
                    'model': None,
                    'test_metrics': {},
                    'predictions': None,
                    'probabilities': None,
                    'error': str(e)
                }

    def run_simulation(self, 
                      relation_type: str = "CtD", 
                      max_entities: Optional[int] = None,
                      embedding_method: str = "ComplEx",
                      embedding_dim: int = 128,
                      test_size: float = 0.2,
                      num_qubits: int = 10):
        """
        Run the complete simulation for cure prediction.

        Args:
            relation_type: Relation type to predict (e.g., "CtD")
            max_entities: Max entities to include (None for all)
            embedding_method: KG embedding method
            embedding_dim: Embedding dimension
            test_size: Test set proportion
            num_qubits: Number of qubits for quantum models
        """
        logger.info("Starting cure prediction simulation...")
        
        # Step 1: Load knowledge graph
        self.load_knowledge_graph(relation_type=relation_type, max_entities=max_entities)
        
        # Step 2: Train embeddings
        self.train_embeddings(method=embedding_method, embedding_dim=embedding_dim)
        
        # Step 3: Prepare data
        train_df, test_df = self.prepare_link_prediction_data(test_size=test_size)
        
        # Step 4: Extract features
        X_train = self.extract_pair_features(train_df, feature_type="combined")
        X_test = self.extract_pair_features(test_df, feature_type="combined")
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        logger.info(f"Feature extraction completed: X_train shape = {X_train.shape}")
        
        # Step 5: Train classical models
        self.train_classical_models(X_train, y_train, X_test, y_test)
        
        # Step 6: Train quantum models
        self.train_quantum_models(X_train, y_train, X_test, y_test, num_qubits=num_qubits)
        
        # Step 7: Compile results
        self.compile_results()
        
        logger.info("Simulation completed successfully!")
        return self.results

    def compile_results(self):
        """
        Compile and summarize all results.
        """
        logger.info("Compiling results...")
        
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
        for model_name, metrics in sorted_results[:5]:
            logger.info(f"  {model_name}: PR-AUC = {metrics['pr_auc']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
        
        self.results['summary'] = summary
        self.results['sorted_by_pr_auc'] = sorted_results

    def predict_cures(self, compounds: List[str], diseases: List[str], top_k: int = 10):
        """
        Predict potential cures for given compounds and diseases.

        Args:
            compounds: List of compound IDs (e.g., ["Compound::DB00001", ...])
            diseases: List of disease IDs (e.g., ["Disease::DOID:1234", ...])
            top_k: Number of top predictions to return

        Returns:
            DataFrame with predictions
        """
        if not self.models:
            raise ValueError("No models trained. Run simulation first.")
        
        # Get the best performing model
        best_model_name = self.results['sorted_by_pr_auc'][0][0] if self.results.get('sorted_by_pr_auc') else None
        
        if not best_model_name:
            raise ValueError("No trained models found in results.")
        
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
        
        # Extract features
        X = self.extract_pair_features(pairs_df, feature_type="combined")
        
        # Make predictions using the best model
        if 'classical' in best_model_name:
            # Classical model
            X_scaled = self.scaler.transform(X)
            probabilities = best_model.predict_proba(X_scaled)[:, 1]
        else:
            # Quantum model - need to reduce dimensions
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10, random_state=42)  # Assuming 10 qubits
            X_reduced = pca.fit_transform(X)
            X_reduced = np.tanh(X_reduced)
            probabilities = best_model.predict_proba(X_reduced)[:, 1]
        
        # Add predictions to DataFrame
        pairs_df['prediction_probability'] = probabilities
        pairs_df['prediction_score'] = probabilities  # Same for now, but could be different
        
        # Sort by prediction score
        pairs_df = pairs_df.sort_values('prediction_score', ascending=False)
        
        logger.info(f"Predicted top {top_k} potential cures:")
        for i, (_, row) in enumerate(pairs_df.head(top_k).iterrows()):
            logger.info(f"  {i+1}. {row['compound']} -> {row['disease']}: {row['prediction_score']:.4f}")
        
        return pairs_df.head(top_k)

    def save_results(self, filename: str = None):
        """
        Save results to file.

        Args:
            filename: Output filename (auto-generated if None)
        """
        import json
        from datetime import datetime
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cure_prediction_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare serializable results
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key in ['model', 'predictions', 'probabilities']:  # Skip non-serializable
                        continue
                    serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = str(value)  # Convert to string if not dict
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath

    def get_compound_disease_interactions(self, compound_id: str, top_k: int = 10):
        """
        Get diseases most associated with a specific compound.

        Args:
            compound_id: The compound ID to analyze
            top_k: Number of top diseases to return

        Returns:
            DataFrame with compound-disease interactions
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


def run_cure_prediction_pipeline(
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
    Convenience function to run the complete cure prediction pipeline.

    Args:
        relation_type: Relation type to predict
        max_entities: Max entities to include
        embedding_method: KG embedding method
        embedding_dim: Embedding dimension
        test_size: Test set proportion
        num_qubits: Number of qubits for quantum models
        data_dir: Directory for data
        results_dir: Directory for results

    Returns:
        CurePredictionFramework instance with results
    """
    framework = CurePredictionFramework(data_dir=data_dir, results_dir=results_dir)
    
    results = framework.run_simulation(
        relation_type=relation_type,
        max_entities=max_entities,
        embedding_method=embedding_method,
        embedding_dim=embedding_dim,
        test_size=test_size,
        num_qubits=num_qubits
    )
    
    return framework


def find_potential_cures(compound_list: List[str], disease_list: List[str], framework: CurePredictionFramework, top_k: int = 10):
    """
    Find potential cures for given compounds and diseases using a trained framework.

    Args:
        compound_list: List of compound IDs
        disease_list: List of disease IDs
        framework: Trained CurePredictionFramework instance
        top_k: Number of top predictions to return

    Returns:
        DataFrame with top potential cures
    """
    return framework.predict_cures(compound_list, disease_list, top_k=top_k)