"""
Quantum-Enhanced Cure Prediction Framework

This module implements quantum machine learning models for enhanced cure prediction.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Try to import quantum libraries with graceful fallbacks
try:
    from qiskit import QuantumCircuit, ClassicalRegister
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, RealAmplitudes, TwoLocal
    from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
    from qiskit_machine_learning.algorithms import VQC, QSVC
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    # Try different Sampler import options
    try:
        from qiskit.primitives import Sampler
    except ImportError:
        from qiskit_aer.primitives import Sampler
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: Quantum libraries not available. Install qiskit, qiskit-machine-learning, and qiskit-algorithms for quantum features.")

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


class QuantumCurePredictionFramework:
    """
    Framework for predicting potential cures using quantum machine learning models.
    """

    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.task_edges = None
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.embeddings = None
        self.scaler = StandardScaler()
        self.pca = None
        self.q_models = {}
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

    def extract_quantum_ready_features(self, pairs_df: pd.DataFrame, feature_type: str = "combined"):
        """
        Extract quantum-ready features for (compound, disease) pairs.
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
            
            # Combine all features
            all_features = np.concatenate([
                concat_feat, 
                diff_feat, 
                hadamard_feat, 
                [l2_dist, cosine_sim]
            ])
            
            features.append(all_features)
        
        return np.array(features)

    def create_quantum_feature_map(self, num_qubits: int, feature_map_type: str = "ZZ", reps: int = 2):
        """
        Create quantum feature map for embedding classical data into quantum states.
        """
        if feature_map_type == "ZZ":
            return ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement="linear")
        elif feature_map_type == "Z":
            return ZFeatureMap(feature_dimension=num_qubits, reps=reps)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")

    def create_quantum_ansatz(self, num_qubits: int, ansatz_type: str = "TwoLocal", reps: int = 3):
        """
        Create quantum ansatz (variational form) for VQC.
        """
        if ansatz_type == "TwoLocal":
            return TwoLocal(num_qubits, "ry", "cz", reps=reps, skip_final_rotation_layer=True)
        elif ansatz_type == "RealAmplitudes":
            return RealAmplitudes(num_qubits, reps=reps)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")

    def train_quantum_models(self, X_train, y_train, X_test, y_test, num_qubits: int = 10):
        """
        Train quantum machine learning models (QSVC and VQC).
        """
        if not QUANTUM_AVAILABLE:
            logger.warning("Quantum libraries not available. Skipping quantum model training.")
            return
        
        logger.info(f"Training quantum models with {num_qubits} qubits...")
        
        # Reduce dimensions to match number of qubits using PCA
        if X_train.shape[1] > num_qubits:
            self.pca = PCA(n_components=num_qubits, random_state=42)
            X_train_reduced = self.pca.fit_transform(X_train)
            X_test_reduced = self.pca.transform(X_test)
        else:
            # If features are fewer than qubits, pad with zeros
            X_train_reduced = np.pad(X_train, ((0, 0), (0, max(0, num_qubits - X_train.shape[1]))), mode='constant')
            X_test_reduced = np.pad(X_test, ((0, 0), (0, max(0, num_qubits - X_test.shape[1]))), mode='constant')
        
        # Normalize to quantum-safe range [-π, π]
        X_train_reduced = np.tanh(X_train_reduced) * np.pi
        X_test_reduced = np.tanh(X_test_reduced) * np.pi
        
        # Initialize quantum models
        quantum_models = {}
        
        # 1. Quantum Support Vector Classifier (QSVC)
        try:
            logger.info("Training QSVC...")
            
            # Create feature map
            feature_map = self.create_quantum_feature_map(num_qubits, feature_map_type="ZZ", reps=2)
            
            # Create quantum kernel
            sampler = AerSamplerV2()
            fidelity = ComputeUncompute(sampler=sampler)
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map.decompose(reps=10), fidelity=fidelity)
            
            # Create and train QSVC
            qsvc = QSVC(quantum_kernel=quantum_kernel)
            qsvc.fit(X_train_reduced, y_train)
            
            # Make predictions
            y_pred = qsvc.predict(X_test_reduced)
            # For QSVC, get decision function for probabilities
            y_pred_proba = qsvc.decision_function(X_test_reduced)
            # Convert to probabilities using sigmoid
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            quantum_models['QSVC'] = {
                'model': qsvc,
                'test_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"QSVC - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train QSVC: {e}")
            quantum_models['QSVC'] = {
                'model': None,
                'test_metrics': {},
                'predictions': None,
                'probabilities': None,
                'error': str(e)
            }
        
        # 2. Variational Quantum Classifier (VQC)
        try:
            logger.info("Training VQC...")
            
            # Create feature map and ansatz
            feature_map = self.create_quantum_feature_map(num_qubits, feature_map_type="ZZ", reps=2)
            ansatz = self.create_quantum_ansatz(num_qubits, ansatz_type="RealAmplitudes", reps=3)
            
            # Combine into circuit
            vqc_circuit = feature_map.compose(ansatz)
            
            # Create optimizer
            optimizer = COBYLA(maxiter=100)  # Reduced iterations for faster training
            
            # Create and train VQC
            vqc = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
                sampler=AerSamplerV2()
            )
            vqc.fit(X_train_reduced, y_train)
            
            # Make predictions
            y_pred = vqc.predict(X_test_reduced)
            # For VQC, get decision function for probabilities
            y_pred_proba = vqc.score(X_test_reduced, y_test)  # This gives accuracy, not probabilities
            # Use a workaround to get probabilities
            y_pred_proba = vqc.predict(X_test_reduced).astype(float)  # Use predictions as proxy
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            quantum_models['VQC'] = {
                'model': vqc,
                'test_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"VQC - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train VQC: {e}")
            quantum_models['VQC'] = {
                'model': None,
                'test_metrics': {},
                'predictions': None,
                'probabilities': None,
                'error': str(e)
            }
        
        # Store quantum models
        self.q_models = quantum_models
        
        # Add to results
        for model_name, model_data in quantum_models.items():
            if model_data.get('model') is not None:
                self.results[f'quantum_{model_name}'] = model_data

    def train_quantum_enhanced_classical(self, X_train, y_train, X_test, y_test, num_qubits: int = 10):
        """
        Train classical models on quantum-ready features for comparison.
        """
        logger.info(f"Training classical models on quantum-ready features ({num_qubits} dimensions)...")
        
        # Reduce dimensions to match number of qubits using PCA
        if X_train.shape[1] > num_qubits:
            pca = PCA(n_components=num_qubits, random_state=42)
            X_train_reduced = pca.fit_transform(X_train)
            X_test_reduced = pca.transform(X_test)
        else:
            # If features are fewer than qubits, pad with zeros
            X_train_reduced = np.pad(X_train, ((0, 0), (0, max(0, num_qubits - X_train.shape[1]))), mode='constant')
            X_test_reduced = np.pad(X_test, ((0, 0), (0, max(0, num_qubits - X_test.shape[1]))), mode='constant')
        
        # Normalize to quantum-safe range
        X_train_reduced = np.tanh(X_train_reduced)
        X_test_reduced = np.tanh(X_test_reduced)
        
        # Train classical models on quantum-ready features
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
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
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_reduced)[:, 1]
            else:
                y_pred_proba = y_pred.astype(float)  # Use predictions as proxy
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            self.q_models[name] = model
            self.results[f'quantum_ready_{name}'] = {
                'model': model,
                'test_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def run_quantum_simulation(self, 
                              relation_type: str = "CtD", 
                              max_entities: Optional[int] = None,
                              embedding_method: str = "ComplEx",
                              embedding_dim: int = 128,
                              test_size: float = 0.2,
                              num_qubits: int = 10):
        """
        Run the quantum-enhanced simulation for cure prediction.
        """
        logger.info("Starting quantum-enhanced cure prediction simulation...")
        
        # Step 1: Load knowledge graph
        self.load_knowledge_graph(relation_type=relation_type, max_entities=max_entities)
        
        # Step 2: Train embeddings
        self.train_embeddings(method=embedding_method, embedding_dim=embedding_dim)
        
        # Step 3: Prepare data
        train_df, test_df = self.prepare_link_prediction_data(test_size=test_size)
        
        # Step 4: Extract quantum-ready features
        X_train = self.extract_quantum_ready_features(train_df, feature_type="combined")
        X_test = self.extract_quantum_ready_features(test_df, feature_type="combined")
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        logger.info(f"Quantum-ready feature extraction completed: X_train shape = {X_train.shape}")
        
        # Step 5: Train quantum models
        self.train_quantum_models(X_train, y_train, X_test, y_test, num_qubits=num_qubits)
        
        # Step 6: Train quantum-ready classical models for comparison
        self.train_quantum_enhanced_classical(X_train, y_train, X_test, y_test, num_qubits=num_qubits)
        
        # Step 7: Compile results
        self.compile_results()
        
        logger.info("Quantum-enhanced simulation completed successfully!")
        return self.results

    def compile_results(self):
        """
        Compile and summarize all quantum results.
        """
        logger.info("Compiling quantum-enhanced results...")
        
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
        
        logger.info("Top performing quantum models by PR-AUC:")
        for model_name, metrics in sorted_results:
            logger.info(f"  {model_name}: PR-AUC = {metrics['pr_auc']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
        
        self.results['summary'] = summary
        self.results['sorted_by_pr_auc'] = sorted_results

    def predict_cures(self, compounds: List[str], diseases: List[str], top_k: int = 10):
        """
        Predict potential cures for given compounds and diseases using the best quantum model.
        """
        if not self.q_models:
            raise ValueError("No quantum models trained. Run simulation first.")
        
        # Get the best performing quantum model
        best_model_name = None
        best_score = -1
        for key, result in self.results.items():
            if 'test_metrics' in result and 'pr_auc' in result['test_metrics']:
                score = result['test_metrics']['pr_auc']
                if score > best_score:
                    best_score = score
                    best_model_name = key
        
        if not best_model_name:
            raise ValueError("No trained quantum models found in results.")
        
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
        
        # Extract quantum-ready features
        X = self.extract_quantum_ready_features(pairs_df, feature_type="combined")
        
        # Make predictions using the best model
        # Need to determine if this is a quantum or classical model
        if best_model_name.startswith('quantum_'):
            # For quantum models, we need to reduce dimensions
            if self.pca:
                X_reduced = self.pca.transform(X)
            else:
                # If no PCA was applied during training, apply it now
                if X.shape[1] > 10:  # Assuming 10 qubits as default
                    pca_temp = PCA(n_components=10, random_state=42)
                    X_reduced = pca_temp.fit_transform(X)
                else:
                    X_reduced = X
            
            # Normalize to quantum-safe range
            X_reduced = np.tanh(X_reduced) * np.pi
            
            # Make predictions
            if 'QSVC' in best_model_name:
                y_pred_proba = best_model.decision_function(X_reduced)
                y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
            elif 'VQC' in best_model_name:
                y_pred_proba = best_model.predict(X_reduced).astype(float)
            else:
                y_pred_proba = np.random.rand(len(X_reduced))  # Fallback
        else:
            # For classical models on quantum-ready features
            if self.pca:
                X_reduced = self.pca.transform(X)
            else:
                # If no PCA was applied during training, apply it now
                if X.shape[1] > 10:  # Assuming 10 qubits as default
                    pca_temp = PCA(n_components=10, random_state=42)
                    X_reduced = pca_temp.fit_transform(X)
                else:
                    X_reduced = X
            
            # Normalize to quantum-safe range
            X_reduced = np.tanh(X_reduced)
            
            # Make predictions
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_reduced)[:, 1]
            else:
                y_pred_proba = best_model.predict(X_reduced).astype(float)
        
        # Add predictions to DataFrame
        pairs_df['prediction_probability'] = y_pred_proba
        pairs_df['prediction_score'] = y_pred_proba
        
        # Sort by prediction score
        pairs_df = pairs_df.sort_values('prediction_score', ascending=False)
        
        logger.info(f"Predicted top {top_k} potential cures:")
        for i, (_, row) in enumerate(pairs_df.head(top_k).iterrows()):
            logger.info(f"  {i+1}. {row['compound']} -> {row['disease']}: {row['prediction_score']:.4f}")
        
        return pairs_df.head(top_k)


def run_quantum_cure_prediction_pipeline(
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
    Run the quantum-enhanced cure prediction pipeline.
    """
    if not QUANTUM_AVAILABLE:
        raise ImportError("Quantum libraries not available. Install qiskit and related packages.")
    
    framework = QuantumCurePredictionFramework(data_dir=data_dir, results_dir=results_dir)
    
    results = framework.run_quantum_simulation(
        relation_type=relation_type,
        max_entities=max_entities,
        embedding_method=embedding_method,
        embedding_dim=embedding_dim,
        test_size=test_size,
        num_qubits=num_qubits
    )
    
    return framework