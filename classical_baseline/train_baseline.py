# classical_baseline/train_baseline.py

import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalLinkPredictor:
    """
    Classical ML baseline for knowledge graph link prediction.
    Supports Logistic Regression, SVM, and Random Forest.
    Optimized for biomedical tasks (imbalanced data, PR-AUC focus).
    """
    
    def __init__(
        self,
        model_type: str = "LogisticRegression",
        random_state: int = 42,
        data_dir: str = "data",
        model_dir: str = "models"
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metrics: Dict[str, float] = {}
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight="balanced"  # Handle imbalanced data
            )
        elif model_type == "SVM":
            self.model = SVC(
                random_state=random_state,
                probability=True,
                class_weight="balanced",
                kernel="rbf"
            )
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced",
                n_estimators=100
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def prepare_features_and_labels(
        self,
        edge_df: pd.DataFrame,
        embedder
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the embedder to generate features and extract labels.
        """
        logger.info("Preparing features and labels...")
        X = embedder.prepare_link_features(edge_df)
        y = edge_df["label"].values
        
        # Remove any rows where embedding failed (should be rare)
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"Feature matrix shape: {X_clean.shape}")
        return X_clean, y_clean
    
    def train(
        self,
        train_df: pd.DataFrame,
        embedder,
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train the model and evaluate on test set (if provided).
        """
        # Prepare data
        X_train, y_train = self.prepare_features_and_labels(train_df, embedder)
        
        # Scale features (important for SVM/LogReg)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        logger.info(f"Training {self.model_type}...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on train set
        self._evaluate(X_train_scaled, y_train, "train")
        
        # Evaluate on test set if provided
        if test_df is not None:
            X_test, y_test = self.prepare_features_and_labels(test_df, embedder)
            X_test_scaled = self.scaler.transform(X_test)
            self._evaluate(X_test_scaled, y_test, "test")
        
        # Save model and scaler
        self.save_model()
        return self.metrics
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray, prefix: str = "test") -> None:
        """Evaluate model and store metrics."""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Core metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # AUC metrics (robust to imbalance)
        try:
            roc_auc = roc_auc_score(y, y_proba)
        except ValueError:
            roc_auc = float('nan')  # Only one class present
        
        pr_auc = average_precision_score(y, y_proba)
        
        # Store metrics
        self.metrics.update({
            f"{prefix}_accuracy": acc,
            f"{prefix}_precision": prec,
            f"{prefix}_recall": rec,
            f"{prefix}_f1": f1,
            f"{prefix}_roc_auc": roc_auc,
            f"{prefix}_pr_auc": pr_auc
        })
        
        logger.info(f"{prefix.capitalize()} Metrics:")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall: {rec:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        
        # Detailed report (for test set)
        if prefix == "test":
            logger.info("\nClassification Report:")
            logger.info(classification_report(y, y_pred))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for positive link."""
        if self.scaler is None or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self) -> None:
        """Save model and scaler to disk."""
        model_path = os.path.join(self.model_dir, f"classical_{self.model_type.lower()}.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved model to {model_path}")
    
    def load_model(self) -> bool:
        """Load model and scaler from disk."""
        model_path = os.path.join(self.model_dir, f"classical_{self.model_type.lower()}.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded saved model and scaler.")
            return True
        return False


if __name__ == "__main__":
    import argparse, os, json, time, logging
    from kg_layer.kg_loader import (
        load_hetionet_edges,
        extract_task_edges,
        prepare_link_prediction_dataset,
    )
    from kg_layer.kg_embedder import HetionetEmbedder

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("baseline.cli")

    parser = argparse.ArgumentParser(description="Run classical baseline and write results/ metrics.")
    parser.add_argument("--relation", type=str, default="CtD", help="Hetionet relation, e.g., CtD")
    parser.add_argument("--max_entities", type=int, default=300, help="Subsample cap for smaller runs")
    parser.add_argument("--embedding_dim", type=int, default=32, help="KG embedding dimension")
    parser.add_argument("--qml_dim", type=int, default=5, help="PCA-reduced dim (for QML parity)")
    parser.add_argument("--model", type=str, default="LogisticRegression",
                        choices=["LogisticRegression", "SVM", "RandomForest"])
    parser.add_argument("--results_dir", type=str, default="results", help="Output directory for metrics")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1) Load and prepare the task-specific triples and link dataset
    log.info("Loading Hetionet edges…")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(df, relation_type=args.relation, max_entities=args.max_entities)
    train_df, test_df = prepare_link_prediction_dataset(task_edges)

    # 2) Train/load embeddings on the **triples**, not on the link-split
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=args.qml_dim)
    if not embedder.load_saved_embeddings():
        log.info("No saved embeddings found; training/generating embeddings on task triples.")
        embedder.train_embeddings(task_edges)   # <-- key change (triples with source/metaedge/target)
        embedder.reduce_to_qml_dim()

    # 3) Train classical baseline and evaluate
    predictor = ClassicalLinkPredictor(
        model_type=args.model,
        random_state=args.random_state,
        model_dir="models"
    )
    metrics = predictor.train(train_df, embedder, test_df)

    # 4) Persist metrics to results/
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_json = os.path.join(args.results_dir, f"baseline_metrics_{args.model}_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "args": {
                    "relation": args.relation,
                    "max_entities": args.max_entities,
                    "embedding_dim": args.embedding_dim,
                    "qml_dim": args.qml_dim,
                    "model": args.model,
                    "random_state": args.random_state,
                },
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    latest = os.path.join(args.results_dir, "baseline_metrics_latest.json")
    try:
        import shutil
        shutil.copyfile(out_json, latest)
    except Exception:
        pass

    log.info(f"Wrote metrics → {out_json}")
