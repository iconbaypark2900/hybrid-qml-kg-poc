# quantum_layer/qml_trainer.py

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from .qml_model import QMLLinkPredictor
from classical_baseline.train_baseline import ClassicalLinkPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QMLTrainer:
    """
    Trainer for Quantum Machine Learning models in the KG link prediction pipeline.
    
    Handles:
      - Training QML models (VQC/QSVC)
      - Evaluating against classical baselines
      - Logging metrics for benchmarking
      - Saving results for dashboard integration
    """
    
    def __init__(
        self,
        results_dir: str = "results",
        random_state: int = 42
    ):
        self.results_dir = results_dir
        self.random_state = random_state
        os.makedirs(results_dir, exist_ok=True)
    
    def count_trainable_parameters(self, model) -> int:
        """
        Count trainable parameters in the quantum model.
        For VQC: number of ansatz parameters.
        """
        if hasattr(model, 'ansatz') and hasattr(model.ansatz, 'num_parameters'):
            return model.ansatz.num_parameters
        elif hasattr(model, 'quantum_kernel'):
            # QSVC doesn't have trainable params in kernel (fixed feature map)
            return 0
        else:
            return -1  # Unknown
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "QML"
    ) -> Dict[str, float]:
        """
        Evaluate model and return comprehensive metrics.
        """
        logger.info(f"Evaluating {model_name} model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Core metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # AUC metrics
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = float('nan')
        
        pr_auc = average_precision_score(y_test, y_proba)
        
        # Parameter count
        n_params = self.count_trainable_parameters(model)
        
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "num_parameters": n_params,
            "model_type": model_name
        }
        
        logger.info(f"{model_name} Test Metrics:")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall: {rec:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        logger.info(f"  Trainable Parameters: {n_params}")
        
        return metrics
    
    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        embedder,
        qml_config: Dict[str, Any],
        classical_model_type: str = "LogisticRegression",
        quantum_config_path: str = "config/quantum_config.yaml"
    ) -> Dict[str, Dict[str, float]]:
        """
        Full training and evaluation pipeline.
        
        Args:
            train_df, test_df: DataFrames with 'source', 'target', 'label'
            embedder: Trained HetionetEmbedder instance
            qml_config: Dict with QMLLinkPredictor kwargs
            classical_model_type: Baseline model type
            quantum_config_path: Path to quantum configuration file
        
        Returns:
            Dict with 'classical' and 'quantum' metric dictionaries
        """
        # Prepare features
        logger.info("Preparing features for training...")
        X_train = embedder.prepare_link_features(train_df)
        y_train = train_df["label"].values
        X_test = embedder.prepare_link_features(test_df)
        y_test = test_df["label"].values
        
        # Remove any invalid samples
        valid_train = ~np.isnan(X_train).any(axis=1)
        valid_test = ~np.isnan(X_test).any(axis=1)
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]
        
        logger.info(f"Final train set: {X_train.shape[0]} samples")
        logger.info(f"Final test set: {X_test.shape[0]} samples")
        
        # Train classical baseline
        logger.info("Training classical baseline...")
        classical_predictor = ClassicalLinkPredictor(
            model_type=classical_model_type,
            random_state=self.random_state
        )
        classical_predictor.train(train_df, embedder, test_df)
        classical_metrics = self.evaluate_model(
            classical_predictor.model, X_test, y_test, "Classical"
        )
        
        # Train QML model
        logger.info("Training QML model...")
        try:
            qml_predictor = QMLLinkPredictor(
                quantum_config_path=quantum_config_path,
                **qml_config
            )
            qml_predictor.fit(X_train, y_train)
            qml_metrics = self.evaluate_model(
                qml_predictor.model, X_test, y_test, "Quantum"
            )
        except Exception as e:
            logger.error(f"QML training failed: {e}")
            # Create dummy metrics to avoid breaking pipeline
            qml_metrics = {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "roc_auc": 0.0, "pr_auc": 0.0, "num_parameters": -1, "model_type": "Quantum"
            }
        
        # Combine results
        results = {
            "classical": classical_metrics,
            "quantum": qml_metrics
        }
        
        # Save results
        self.save_results(results, qml_config)
        return results
    
    def save_results(self, results: Dict, qml_config: Dict) -> None:
        """
        Save results to CSV for dashboard and benchmarking.
        """
        # Flatten results for CSV
        flat_results = {}
        for model_type, metrics in results.items():
            for key, value in metrics.items():
                flat_results[f"{model_type}_{key}"] = value
        
        # Add QML config
        for key, value in qml_config.items():
            flat_results[f"qml_{key}"] = str(value)
        
        # Save as single-row CSV
        df = pd.DataFrame([flat_results])
        csv_path = os.path.join(self.results_dir, "latest_run.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        # Also save full history (append mode)
        history_path = os.path.join(self.results_dir, "experiment_history.csv")
        if os.path.exists(history_path):
            df_history = pd.read_csv(history_path)
            df_history = pd.concat([df_history, df], ignore_index=True)
        else:
            df_history = df
        df_history.to_csv(history_path, index=False)


# Example usage (uncomment to test)
# if __name__ == "__main__":
#     from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
#     from kg_layer.kg_embedder import HetionetEmbedder
#     
#     # Load data
#     df = load_hetionet_edges()
#     task_edges, _, _ = extract_task_edges(df, relation_type="CtD", max_entities=300)
#     train_df, test_df = prepare_link_prediction_dataset(task_edges)
#     
#     # Load embeddings
#     embedder = HetionetEmbedder(embedding_dim=32, qml_dim=5)
#     if not embedder.load_saved_embeddings():
#         embedder.train_embeddings(train_df)
#         embedder.reduce_to_qml_dim()
#     
#     # QML config
#     qml_config = {
#         "model_type": "VQC",
#         "encoding_method": "feature_map",
#         "num_qubits": 5,
#         "feature_map_type": "ZZ",
#         "feature_map_reps": 2,
#         "ansatz_type": "RealAmplitudes",
#         "ansatz_reps": 3,
#         "optimizer": "COBYLA",
#         "max_iter": 50,
#         "random_state": 42
#     }
#     
#     # Train and evaluate
#     trainer = QMLTrainer()
#     results = trainer.train_and_evaluate(
#         train_df, test_df, embedder, qml_config,
#         classical_model_type="LogisticRegression"
#     )
#     
#     print("\nFinal Comparison:")
#     print(f"Classical PR-AUC: {results['classical']['pr_auc']:.4f}")
#     print(f"Quantum PR-AUC:   {results['quantum']['pr_auc']:.4f}")
#     print(f"Classical Params: {results['classical']['num_parameters']}")
#     print(f"Quantum Params:   {results['quantum']['num_parameters']}")