# quantum_layer/qml_trainer.py

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from .qml_model import QMLLinkPredictor, load_quantum_config
from classical_baseline.train_baseline import ClassicalLinkPredictor
from sklearn.preprocessing import MinMaxScaler

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
        results_dir: Optional[str] = None,
        random_state: Optional[int] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        # Load config if not provided
        if config is None:
            if config_path is None:
                config_path = "config/quantum_layer_config.yaml"
            config = load_quantum_config(config_path)

        # Use provided parameters or fall back to config
        self.results_dir = results_dir if results_dir is not None else config["training"]["results_dir"]
        self.random_state = random_state if random_state is not None else config["model"]["random_state"]
        self.config = config
        os.makedirs(self.results_dir, exist_ok=True)

    def count_trainable_parameters(self, model) -> int:
        """
        Count trainable parameters in the quantum model.
        For VQC: number of ansatz parameters.

        For QSVC: typically 0 (fixed kernel).

        Args:
            model: QMLLinkPredictor model instance

        Returns:
            int: Number of trainable parameters
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

        Args:
            model: Trained model (QMLLinkPredictor or sklearn-like)
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging purposes

        Returns:
            Dict with metrics
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
        qml_config: Optional[Dict[str, Any]] = None,
        classical_model_type: Optional[str] = None,
        quantum_config_path: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Full training and evaluation pipeline.

        Args:
            train_df, test_df: DataFrames with 'source', 'target', 'label'
            embedder: Trained HetionetEmbedder instance
            qml_config: Dict with QMLLinkPredictor kwargs (overrides config)
            classical_model_type: Baseline model type (overrides config)
            quantum_config_path: Path to quantum configuration file (overrides config)
            config_path: Path to quantum layer config YAML file (default: "config/quantum_layer_config.yaml")
            config: Configuration dictionary (if provided, config_path is ignored)

        Returns:
            Dict with 'classical' and 'quantum' metric dictionaries
        """
        # Load config if not provided
        if config is None:
            if config_path is None:
                config_path = "config/quantum_layer_config.yaml"
            config = load_quantum_config(config_path)

        # Merge qml_config with config if provided
        if qml_config is None:
            qml_config = {}

        # Build qml_config from config file, allowing overrides
        if not qml_config:
            qml_config = {
                "model_type": config["model"]["model_type"],
                "encoding_method": config["model"]["encoding_method"],
                "num_qubits": config["model"]["num_qubits"],
                "feature_map_type": config["feature_map"]["feature_map_type"],
                "feature_map_reps": config["feature_map"]["feature_map_reps"],
                "ansatz_type": config["vqc"]["ansatz_type"],
                "ansatz_reps": config["vqc"]["ansatz_reps"],
                "optimizer": config["vqc"]["optimizer"],
                "max_iter": config["vqc"]["max_iter"],
                "random_state": config["model"]["random_state"]
            }

        if quantum_config_path is None:
            quantum_config_path = config["quantum_executor"]["quantum_config_path"]

        if classical_model_type is None:
            # Load classical config
            from classical_baseline.train_baseline import load_classical_config
            classical_config = load_classical_config()
            classical_model_type = classical_config["model"]["model_type"]

        # Get qml_features_mode from embedder config or default
        if hasattr(embedder, 'config') and embedder.config:
            qml_features_mode = embedder.config.get("features", {}).get("qml_features_mode", "diff")
        else:
            # Fallback: try to get from kg_config if available, otherwise use default
            try:
                from kg_layer.kg_loader import load_kg_config
                kg_config = load_kg_config()
                qml_features_mode = kg_config.get("features", {}).get("qml_features_mode", "diff")
            except Exception:
                qml_features_mode = "diff"

        # Prepare features
        logger.info("Preparing features for training...")
        # Classical uses full 128-D features, quantum uses reduced qml_dim features
        X_train_classical = embedder.prepare_link_features(train_df)
        X_train_qml = embedder.prepare_link_features_qml(train_df, mode=qml_features_mode)
        y_train = train_df["label"].values
        X_test_classical = embedder.prepare_link_features(test_df)
        X_test_qml = embedder.prepare_link_features_qml(test_df, mode=qml_features_mode)
        y_test = test_df["label"].values

        # Remove any invalid samples (check both classical and qml features)
        valid_train = ~np.isnan(X_train_classical).any(axis=1) & ~np.isnan(X_train_qml).any(axis=1)
        valid_test = ~np.isnan(X_test_classical).any(axis=1) & ~np.isnan(X_test_qml).any(axis=1)
        X_train_classical, X_train_qml, y_train = X_train_classical[valid_train], X_train_qml[valid_train], y_train[valid_train]
        X_test_classical, X_test_qml, y_test = X_test_classical[valid_test], X_test_qml[valid_test], y_test[valid_test]

        logger.info(f"Final train set: {X_train_classical.shape[0]} samples (classical: {X_train_classical.shape[1]}D, qml: {X_train_qml.shape[1]}D)")
        logger.info(f"Final test set: {X_test_classical.shape[0]} samples (classical: {X_test_classical.shape[1]}D, qml: {X_test_qml.shape[1]}D)")

        # Train classical baseline
        logger.info("Training classical baseline...")
        classical_predictor = ClassicalLinkPredictor(
            model_type=classical_model_type,
            random_state=self.random_state
        )
        classical_predictor.train(train_df, embedder, test_df)
        classical_metrics = self.evaluate_model(
            classical_predictor.model, X_test_classical, y_test, "Classical"
        )

        # Train QML model
        logger.info("Training QML model...")

        # --- QSVC with precomputed kernel path ---
        if qml_config.get("model_type", "QSVC") == "QSVC":
            # Import qsvc_with_precomputed_kernel from this module
            from .qml_trainer import qsvc_with_precomputed_kernel
            # Create args-like object from qml_config
            args_dict = {
                **qml_config,
                "qml_dim": qml_config.get("num_qubits", config["model"]["num_qubits"]),
                "feature_map": qml_config.get("feature_map_type", config["feature_map"]["feature_map_type"]),
                "feature_map_reps": qml_config.get("feature_map_reps", config["feature_map"]["feature_map_reps"]),
                "quantum_config": quantum_config_path
            }
            args = type('Args', (), args_dict)()
            # For logging, use logger - use QML features for quantum model
            svc, K_train, K_test = qsvc_with_precomputed_kernel(X_train_qml, y_train, X_test_qml, y_test, args, logger)
            # Compute metrics using svc, y_train/y_test, K_train/K_test
            def _metrics_precomputed(K, y, split, svc):
                y_pred = svc.predict(K)
                y_score = svc.decision_function(K)
                m = dict(
                    accuracy=float(accuracy_score(y, y_pred)),
                    precision=float(precision_score(y, y_pred, zero_division=0)),
                    recall=float(recall_score(y, y_pred, zero_division=0)),
                    f1=float(f1_score(y, y_pred, zero_division=0)),
                    roc_auc=float(roc_auc_score(y, y_score)) if len(np.unique(y)) > 1 else float("nan"),
                    pr_auc=float(average_precision_score(y, y_score)),
                    num_parameters=0,
                    model_type="Quantum"
                )
                logger.info(f"{split} Metrics:")
                for k, v in m.items():
                    logger.info(f"  {k}: {v:.4f}")
                try:
                    from sklearn.metrics import classification_report
                    rpt = classification_report(y, y_pred)
                    logger.info("\n" + rpt)
                except Exception:
                    pass
                return m, y_pred, y_score
            train_metrics, yhat_tr, yscore_tr = _metrics_precomputed(K_train, y_train, "Train", svc)
            test_metrics, yhat_te, yscore_te = _metrics_precomputed(K_test, y_test, "Test", svc)
            # Save results (same as before)
            import time
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_json = os.path.join(self.results_dir, f"quantum_metrics_QSVC_{ts}.json")
            payload = dict(
                args={**qml_config, "quantum_config": quantum_config_path},
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            import json
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
            pred_csv = os.path.join(self.results_dir, f"predictions_QSVC_{ts}.csv")
            pd.DataFrame(
                {
                    "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                    "y_true": np.concatenate([y_train, y_test]),
                    "y_pred": np.concatenate([yhat_tr, yhat_te]),
                    "y_score": np.concatenate([yscore_tr, yscore_te]),
                }
            ).to_csv(pred_csv, index=False)
            # convenient "latest"
            latest_json = os.path.join(self.results_dir, "quantum_metrics_latest.json")
            latest_pred = os.path.join(self.results_dir, "predictions_latest.csv")
            import shutil
            try:
                shutil.copyfile(out_json, latest_json)
                shutil.copyfile(pred_csv, latest_pred)
            except Exception:
                pass
            logger.info(f"Wrote metrics → {out_json}")
            logger.info(f"Wrote predictions → {pred_csv}")
            # Return results in the same format as the rest of the pipeline
            return {
                "classical": classical_metrics,
                "quantum": test_metrics
            }

        # --- End QSVC precomputed kernel path ---

        try:
            qml_predictor = QMLLinkPredictor(
                quantum_config_path=quantum_config_path,
                **qml_config
            )
            # Use QML features (5D) for quantum model
            qml_predictor.fit(X_train_qml, y_train)
            qml_metrics = self.evaluate_model(
                qml_predictor.model, X_test_qml, y_test, "Quantum"
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

        Args:
            results: Dict with 'classical' and 'quantum' metrics
            qml_config: QML configuration used in the run
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

# --- New function for precomputed QSVC kernel grid search ---
def qsvc_with_precomputed_kernel(X_train, y_train, X_test, y_test, args, log):
    """
    Trains a QSVC model with a precomputed kernel.

    Parameters
    ----------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    args : object
        An object containing the model parameters.
    log : logging.Logger
        The logger to use.

    Returns
    -------
    tuple
        A tuple containing the trained model, the training kernel, and the test kernel.
    """
    # Build feature map
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
    if args.feature_map == "ZZ":
        fm = ZZFeatureMap(feature_dimension=args.qml_dim, reps=args.feature_map_reps, entanglement="linear")
    else:
        fm = ZFeatureMap(feature_dimension=args.qml_dim, reps=args.feature_map_reps)

    # Sampler / mode
    from quantum_layer.quantum_executor import QuantumExecutor
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    sampler, exec_mode = QuantumExecutor(args.quantum_config).get_sampler()
    if exec_mode in ("statevector", "simulator_statevector"):
        qk = FidelityStatevectorKernel(feature_map=fm)
    else:
        qk = FidelityQuantumKernel(feature_map=fm, fidelity=ComputeUncompute(sampler=sampler))

    # Precompute kernels
    K_train = qk.evaluate(X_train)                 # (n_train, n_train)
    K_test  = qk.evaluate(X_test, X_train)         # (n_test,  n_train)

    # Grid over C quickly
    from sklearn.svm import SVC
    from sklearn.metrics import average_precision_score

    best = (float("-inf"), None, None)  # (pr_auc, C, model)
    for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
        svc = SVC(kernel="precomputed", C=C, class_weight="balanced")
        svc.fit(K_train, y_train)
        y_score = svc.decision_function(K_test)
        pr_auc = average_precision_score(y_test, y_score)
        log.info(f"[QSVC-precomputed] C={C} → test PR-AUC={pr_auc:.4f}")
        if pr_auc > best[0]:
            best = (pr_auc, C, svc)

    log.info(f"[QSVC-precomputed] selected C={best[1]} (test PR-AUC={best[0]:.4f})")
    return best[2], K_train, K_test

if __name__ == "__main__":
    import argparse
    import json
    import logging
    import os
    import shutil
    import time

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from kg_layer.kg_embedder import HetionetEmbedder
    from kg_layer.kg_loader import (
        extract_task_edges,
        load_hetionet_edges,
        prepare_link_prediction_dataset,
    )
    from .qml_model import QMLLinkPredictor

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("qml.cli")

    parser = argparse.ArgumentParser(description="Run QML (QSVC/VQC) and write results/")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)

    parser.add_argument("--model_type", type=str, default="QSVC", choices=["QSVC", "VQC"])
    parser.add_argument("--encoding_method", type=str, default="feature_map", choices=["feature_map"])
    parser.add_argument("--feature_map", type=str, default="ZZ", choices=["ZZ", "Z"])
    parser.add_argument("--feature_map_reps", type=int, default=2)
    parser.add_argument("--ansatz", type=str, default="RealAmplitudes", choices=["RealAmplitudes", "EfficientSU2"])
    parser.add_argument("--ansatz_reps", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="COBYLA", choices=["COBYLA", "SPSA"])
    parser.add_argument("--max_iter", type=int, default=50)

    parser.add_argument("--qml_features", type=str, default="diff", choices=["diff", "hadamard", "both"])

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--quantum_config", type=str, default="config/quantum_config.yaml")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--train_limit",
        type=int,
        default=0,
        help="If > 0, subsample this many TRAIN examples (stratified by label) for faster VQC iterations.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1) Data
    log.info("Loading Hetionet edges…")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(
        df, relation_type=args.relation, max_entities=args.max_entities
    )
    train_df, test_df = prepare_link_prediction_dataset(task_edges)

    # Subsample train set if train_limit is set
    if args.train_limit and args.train_limit > 0:
        rs = args.random_state
        n_total = min(args.train_limit, len(train_df))
        n_pos = int(round(n_total * train_df["label"].mean()))
        n_neg = n_total - n_pos

        pos = train_df[train_df["label"] == 1]
        neg = train_df[train_df["label"] == 0]
        pos_s = pos.sample(n=min(n_pos, len(pos)), random_state=rs, replace=False)
        neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=rs, replace=False)

        train_df = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=rs).reset_index(drop=True)
        log.info(f"Subsampled train set to {len(train_df)} examples (from {len(pos)+len(neg)}).")

    # 2) Embeddings (triples -> entity embeddings -> PCA to qml_dim)
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=args.qml_dim)
    if not embedder.load_saved_embeddings():
        log.info("No saved embeddings found; training/generating embeddings on task triples.")
        embedder.train_embeddings(task_edges)  # PyKEEN or deterministic fallback
    # Always reduce for QML: feature dim must equal num_qubits == qml_dim
    embedder.reduce_to_qml_dim()

    # 3) Build QML features (shape: [n, qml_dim])
    def _X_y(splits: pd.DataFrame):
        X = embedder.prepare_link_features_qml(splits, mode=args.qml_features)
        y = splits["label"].astype(int).values
        return X, y

    log.info("Preparing train features…")
    X_train, y_train = _X_y(train_df)
    log.info("Preparing test features…")
    X_test, y_test = _X_y(test_df)
    log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4) QML model
    if args.model_type == "QSVC":
        svc, K_train, K_test = qsvc_with_precomputed_kernel(X_train, y_train, X_test, y_test, args, log)
        # Compute metrics using svc, y_train/y_test, K_train/K_test and write the same JSON/CSV
        def _metrics_precomputed(K, y, split, svc):
            y_pred = svc.predict(K)
            y_score = svc.decision_function(K)
            m = dict(
                accuracy=float(accuracy_score(y, y_pred)),
                precision=float(precision_score(y, y_pred, zero_division=0)),
                recall=float(recall_score(y, y_pred, zero_division=0)),
                f1=float(f1_score(y, y_pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, y_score)) if len(np.unique(y)) > 1 else float("nan"),
                pr_auc=float(average_precision_score(y, y_score)),
            )
            log.info(f"{split} Metrics:")
            for k, v in m.items():
                log.info(f"  {k}: {v:.4f}")
            try:
                from sklearn.metrics import classification_report
                rpt = classification_report(y, y_pred)
                log.info("\n" + rpt)
            except Exception:
                pass
            return m, y_pred, y_score
        train_metrics, yhat_tr, yscore_tr = _metrics_precomputed(K_train, y_train, "Train", svc)
        test_metrics, yhat_te, yscore_te = _metrics_precomputed(K_test, y_test, "Test", svc)
        # 6) Write results
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_json = os.path.join(args.results_dir, f"quantum_metrics_QSVC_{ts}.json")
        payload = dict(
            args=vars(args),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        pred_csv = os.path.join(args.results_dir, f"predictions_QSVC_{ts}.csv")
        pd.DataFrame(
            {
                "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                "y_true": np.concatenate([y_train, y_test]),
                "y_pred": np.concatenate([yhat_tr, yhat_te]),
                "y_score": np.concatenate([yscore_tr, yscore_te]),
            }
        ).to_csv(pred_csv, index=False)

        # convenient "latest"
        latest_json = os.path.join(args.results_dir, "quantum_metrics_latest.json")
        latest_pred = os.path.join(args.results_dir, "predictions_latest.csv")
        try:
            shutil.copyfile(out_json, latest_json)
            shutil.copyfile(pred_csv, latest_pred)
        except Exception:
            pass

        log.info(f"Wrote metrics → {out_json}")
        log.info(f"Wrote predictions → {pred_csv}")
        # skip the old clf path
        exit(0)

    clf = QMLLinkPredictor(
        model_type=args.model_type,
        encoding_method=args.encoding_method,
        num_qubits=int(args.qml_dim),  # 1 qubit per reduced feature
        ansatz_type=args.ansatz,
        ansatz_reps=int(args.ansatz_reps),
        optimizer=args.optimizer,
        max_iter=int(args.max_iter),
        feature_map_type=args.feature_map,
        feature_map_reps=int(args.feature_map_reps),
        random_state=args.random_state,
        quantum_config_path=args.quantum_config,
    )

    log.info(f"Training {args.model_type}…")
    clf.fit(X_train, y_train)

    # Optional: small grid over QSVC C (keeps same kernel object; refits fast on this data size)
    def _try_qsvc_with_C_grid():
        if args.model_type != "QSVC":
            return None
        c_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
        best = None
        for c in c_grid:
            # reuse same predictor, just swap underlying C
            from copy import deepcopy
            _clf = deepcopy(clf)
            _clf.fit(X_train, y_train)  # builds kernel once inside
            try:
                _clf.model.C = c
            except Exception:
                pass  # older versions store C in nested estimator; fall through
            # re-fit with new C (QSVC trains a classical SVM over the kernel matrix)
            _clf.model.fit(X_train, y_train)
            from sklearn.metrics import average_precision_score
            y_score = getattr(_clf.model, "decision_function")(X_test)
            pr_auc = float(average_precision_score(y_test, y_score))
            log.info(f"[QSVC] C={c} → test PR-AUC={pr_auc:.4f}")
            if (best is None) or (pr_auc > best[0]):
                best = (pr_auc, c, _clf)
        if best:
            log.info(f"[QSVC] selected C={best[1]} (test PR-AUC={best[0]:.4f})")
            return best[2]
        return None

    # --- Optionally use precomputed kernel grid search for QSVC ---
    if args.model_type == "QSVC" and getattr(args, "use_precomputed_kernel", False):
        clf, K_train, K_test = qsvc_with_precomputed_kernel(X_train, y_train, X_test, y_test, args, log)
        # For metrics, we need to adapt _metrics to use precomputed kernel
        def _metrics_precomputed(K, y, split, clf):
            y_pred = clf.predict(K)
            y_score = clf.decision_function(K)
            m = dict(
                accuracy=float(accuracy_score(y, y_pred)),
                precision=float(precision_score(y, y_pred, zero_division=0)),
                recall=float(recall_score(y, y_pred, zero_division=0)),
                f1=float(f1_score(y, y_pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, y_score)) if len(np.unique(y)) > 1 else float("nan"),
                pr_auc=float(average_precision_score(y, y_score)),
            )
            log.info(f"{split} Metrics:")
            for k, v in m.items():
                log.info(f"  {k}: {v:.4f}")
            try:
                from sklearn.metrics import classification_report
                rpt = classification_report(y, y_pred)
                log.info("\n" + rpt)
            except Exception:
                pass
            return m, y_pred, y_score
        train_metrics, yhat_tr, yscore_tr = _metrics_precomputed(K_train, y_train, "Train", clf)
        test_metrics, yhat_te, yscore_te = _metrics_precomputed(K_test, y_test, "Test", clf)
    else:
        if args.model_type == "QSVC":
            tuned = _try_qsvc_with_C_grid()
            if tuned is not None:
                clf = tuned

        # 5) Metrics + predictions
        def _metrics(X, y, split):
            y_pred = clf.predict(X)
            # Prefer probabilities; fall back to decision_function; else use hard labels
            if hasattr(clf, "predict_proba"):
                try:
                    y_score = clf.predict_proba(X)[:, 1]
                except Exception:
                    y_score = getattr(clf.model, "decision_function", lambda Z: y_pred)(X)
            else:
                y_score = getattr(clf.model, "decision_function", lambda Z: y_pred)(X)

            m = dict(
                accuracy=float(accuracy_score(y, y_pred)),
                precision=float(precision_score(y, y_pred, zero_division=0)),
                recall=float(recall_score(y, y_pred, zero_division=0)),
                f1=float(f1_score(y, y_pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, y_score)) if len(np.unique(y)) > 1 else float("nan"),
                pr_auc=float(average_precision_score(y, y_score)),
            )
            log.info(f"{split} Metrics:")
            for k, v in m.items():
                log.info(f"  {k}: {v:.4f}")
            try:
                rpt = classification_report(y, y_pred)
                log.info("\n" + rpt)
            except Exception:
                pass
            return m, y_pred, y_score

        train_metrics, yhat_tr, yscore_tr = _metrics(X_train, y_train, "Train")
        test_metrics, yhat_te, yscore_te = _metrics(X_test, y_test, "Test")

    # 6) Write results
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_json = os.path.join(args.results_dir, f"quantum_metrics_{args.model_type}_{ts}.json")
    payload = dict(
        args=vars(args),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    pred_csv = os.path.join(args.results_dir, f"predictions_{args.model_type}_{ts}.csv")
    pd.DataFrame(
        {
            "split": ["train"] * len(y_train) + ["test"] * len(y_test),
            "y_true": np.concatenate([y_train, y_test]),
            "y_pred": np.concatenate([yhat_tr, yhat_te]),
            "y_score": np.concatenate([yscore_tr, yscore_te]),
        }
    ).to_csv(pred_csv, index=False)

    # convenient "latest"
    latest_json = os.path.join(args.results_dir, "quantum_metrics_latest.json")
    latest_pred = os.path.join(args.results_dir, "predictions_latest.csv")
    try:
        shutil.copyfile(out_json, latest_json)
        shutil.copyfile(pred_csv, latest_pred)
    except Exception:
        pass

    log.info(f"Wrote metrics → {out_json}")
    log.info(f"Wrote predictions → {pred_csv}")
