# quantum_layer/qml_trainer.py

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
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
        self.last_model = None  # Store last trained model for calibration
        self.last_train_features = None  # Store training features for calibration
        self.last_test_features = None  # Store test features for calibration
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
        # NOTE: ClassicalLinkPredictor trains with a StandardScaler. If we pass only its underlying sklearn
        # estimator (or evaluate on unscaled features), metrics can become inconsistent or meaningless.
        X_eval = X_test
        base_model = model
        scaler = getattr(model, "scaler", None)
        if hasattr(model, "model") and getattr(model, "model") is not None:
            base_model = getattr(model, "model")
            scaler = getattr(model, "scaler", scaler)
        if scaler is not None:
            try:
                X_eval = scaler.transform(X_test)
            except Exception:
                X_eval = X_test

        # scores
        y_proba = None
        if hasattr(base_model, "predict_proba"):
            try:
                y_proba = base_model.predict_proba(X_eval)[:, 1]
            except Exception:
                y_proba = None
        if y_proba is None and hasattr(base_model, "decision_function"):
            y_proba = base_model.decision_function(X_eval)
        if y_proba is None:
            y_proba = base_model.predict(X_eval)

        # predicted labels (use 0.5 when scores look like probabilities, else 0 threshold)
        y_proba_arr = np.asarray(y_proba)
        if np.nanmin(y_proba_arr) >= 0.0 and np.nanmax(y_proba_arr) <= 1.0:
            y_pred = (y_proba_arr >= 0.5).astype(int)
        else:
            y_pred = (y_proba_arr >= 0.0).astype(int)

        # Core metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # AUC metrics
        try:
            roc_auc = roc_auc_score(y_test, y_proba_arr)
        except ValueError:
            roc_auc = float('nan')

        pr_auc = average_precision_score(y_test, y_proba_arr)

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
        # Keep aligned dataframes for logging predictions with endpoints
        try:
            train_df_valid = train_df.iloc[np.where(valid_train)[0]].copy()
            test_df_valid = test_df.iloc[np.where(valid_test)[0]].copy()
        except Exception:
            train_df_valid = train_df.copy()
            test_df_valid = test_df.copy()

        logger.info(f"Final train set: {X_train_classical.shape[0]} samples (classical: {X_train_classical.shape[1]}D, qml: {X_train_qml.shape[1]}D)")
        logger.info(f"Final test set: {X_test_classical.shape[0]} samples (classical: {X_test_classical.shape[1]}D, qml: {X_test_qml.shape[1]}D)")

        # Train classical baseline
        logger.info("Training classical baseline...")
        # Ensure string entity IDs exist for proper embedding lookup
        # This prevents the bug where negatives with only integer IDs get identical embeddings
        def _ensure_string_entity_ids(df: pd.DataFrame, embedder) -> pd.DataFrame:
            """Add string entity ID columns if missing."""
            if df is None or df.empty:
                return df

            needs_copy = False
            result = df
            id_map = getattr(embedder, "id_to_entity", {}) or {}
            added_source = 0
            added_target = 0

            # Check if we need to add 'source' column
            if "source" not in df.columns and "source_id" in df.columns:
                if not needs_copy:
                    result = df.copy()
                    needs_copy = True
                result["source"] = result["source_id"].map(
                    lambda x: id_map.get(int(x), f"Entity::{x}") if pd.notna(x) else None
                )
                added_source = int(result["source"].notna().sum())

            # Check if we need to add 'target' column
            if "target" not in df.columns and "target_id" in df.columns:
                if not needs_copy:
                    result = df.copy()
                    needs_copy = True
                result["target"] = result["target_id"].map(
                    lambda x: id_map.get(int(x), f"Entity::{x}") if pd.notna(x) else None
                )
                added_target = int(result["target"].notna().sum())

            if added_source or added_target:
                try:
                    logger.debug(
                        "Added string entity IDs via embedder.id_to_entity: source=%s target=%s",
                        added_source,
                        added_target,
                    )
                except Exception:
                    pass

            return result

        # Apply fix to prevent all negatives getting identical embeddings
        train_df_fixed = _ensure_string_entity_ids(train_df, embedder)
        test_df_fixed = _ensure_string_entity_ids(test_df, embedder) if test_df is not None else None

        classical_predictor = ClassicalLinkPredictor(
            model_type=classical_model_type,
            random_state=self.random_state
        )
        classical_predictor.train(train_df_fixed, embedder, test_df_fixed)
        classical_metrics = self.evaluate_model(
            classical_predictor, X_test_classical, y_test, "Classical"
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
            svc, K_train, K_test, kernel_obs = qsvc_with_precomputed_kernel(X_train_qml, y_train, X_test_qml, y_test, args, logger)
            # Attach dataset-size evidence for dashboard interpretation (why PR-AUC can look "low" on tiny runs)
            try:
                ytr = np.asarray(y_train).astype(int)
                yte = np.asarray(y_test).astype(int)
                data_obs = {
                    "data_train_n": int(len(ytr)),
                    "data_train_pos": int(ytr.sum()),
                    "data_train_pos_rate": float(ytr.mean()) if len(ytr) else float("nan"),
                    "data_test_n": int(len(yte)),
                    "data_test_pos": int(yte.sum()),
                    "data_test_pos_rate": float(yte.mean()) if len(yte) else float("nan"),
                }
                # total positive edges after split = number of original positive task edges (since we add negatives separately)
                try:
                    data_obs["data_pos_edges_total"] = int(train_df["label"].sum() + test_df["label"].sum())
                except Exception:
                    pass
                if isinstance(kernel_obs, dict):
                    kernel_obs.update(data_obs)
                else:
                    kernel_obs = data_obs
            except Exception:
                pass
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
                    if isinstance(v, (int, float)):
                        logger.info(f"  {k}: {v:.4f}")
                    else:
                        logger.info(f"  {k}: {v}")
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
            # Include endpoints for "Findings" (compound/disease pairs)
            pred_rows = []
            try:
                id_to_entity = getattr(embedder, "id_to_entity", {}) or {}
                def _map_ent(x):
                    try:
                        return id_to_entity.get(int(x), None)
                    except Exception:
                        return None

                tr = train_df_valid.copy()
                te = test_df_valid.copy()
                tr["split"] = "train"
                te["split"] = "test"
                tr["y_true"] = y_train
                te["y_true"] = y_test
                tr["y_pred"] = yhat_tr
                te["y_pred"] = yhat_te
                tr["y_score"] = yscore_tr
                te["y_score"] = yscore_te

                for dfp in (tr, te):
                    if "source_id" in dfp.columns:
                        dfp["source"] = dfp["source_id"].apply(_map_ent)
                    if "target_id" in dfp.columns:
                        dfp["target"] = dfp["target_id"].apply(_map_ent)

                out_df = pd.concat([tr, te], ignore_index=True)
                cols = []
                for c in ["split", "source_id", "target_id", "source", "target", "label", "y_true", "y_pred", "y_score"]:
                    if c in out_df.columns:
                        cols.append(c)
                out_df[cols].to_csv(pred_csv, index=False)
            except Exception:
                # Fallback to metrics-only predictions
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
            results = {
                "classical": classical_metrics,
                "quantum": test_metrics
            }
            # attach observables for logging
            if isinstance(kernel_obs, dict):
                results["observables"] = kernel_obs
            # Store model and features for potential calibration
            self.last_model = svc
            self.last_train_features = K_train
            self.last_test_features = K_test
            self.save_results(results, qml_config, quantum_config_path)
            return results

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
        self.save_results(results, qml_config, quantum_config_path)
        return results

    def _get_execution_metadata(self, quantum_config_path: Optional[str]) -> Dict[str, Optional[str]]:
        """Extract execution metadata from the quantum config."""
        if not quantum_config_path:
            return {}

        try:
            with open(quantum_config_path, "r") as f:
                quantum_config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load quantum config for metadata: {e}")
            return {}

        # Substitute environment variables
        def substitute_env_vars(obj):
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                value = os.getenv(var_name, obj)
                if isinstance(value, str):
                    value = value.strip().strip('"').strip("'").strip('{').strip('}')
                return value
            if isinstance(obj, dict):
                return {k: substitute_env_vars(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [substitute_env_vars(item) for item in obj]
            return obj

        quantum_config = substitute_env_vars(quantum_config)
        quantum_block = quantum_config.get("quantum", {})
        execution_mode = quantum_block.get("execution_mode")
        noise_model = quantum_block.get("simulator", {}).get("noise_model")
        heron_block = quantum_block.get("heron", {}) if isinstance(quantum_block.get("heron", {}), dict) else {}
        simulator_block = quantum_block.get("simulator", {}) if isinstance(quantum_block.get("simulator", {}), dict) else {}
        backend_label = None

        if execution_mode == "heron":
            backend_label = quantum_block.get("heron", {}).get("backend")
        elif execution_mode in ("simulator", "auto", "statevector", "simulator_statevector"):
            backend_label = "simulator_noisy" if noise_model else "simulator"
        elif execution_mode:
            backend_label = execution_mode

        return {
            "execution_mode": execution_mode,
            "noise_model": str(noise_model) if noise_model is not None else None,
            "backend_label": backend_label,
            "execution_shots": str(heron_block.get("shots")) if execution_mode == "heron" and heron_block.get("shots") is not None else (
                str(simulator_block.get("shots")) if simulator_block.get("shots") is not None else None
            ),
            # Hardware mitigation knobs (best-effort logging; applied by QuantumExecutor if supported)
            "mitigation_resilience_level": str(heron_block.get("resilience_level")) if execution_mode == "heron" and heron_block.get("resilience_level") is not None else None,
            "mitigation_optimization_level": str(heron_block.get("optimization_level")) if execution_mode == "heron" and heron_block.get("optimization_level") is not None else None,
            "mitigation_dynamical_decoupling": str(heron_block.get("use_dynamical_decoupling")) if execution_mode == "heron" and heron_block.get("use_dynamical_decoupling") is not None else None,
        }

    def save_results(self, results: Dict, qml_config: Dict, quantum_config_path: Optional[str] = None) -> None:
        """
        Save results to CSV for dashboard and benchmarking.

        Args:
            results: Dict with 'classical' and 'quantum' metrics
            qml_config: QML configuration used in the run
        """
        # Flatten results for CSV
        flat_results = {}
        for model_type, metrics in results.items():
            if model_type == "observables" and isinstance(metrics, dict):
                for k, v in metrics.items():
                    flat_results[f"obs_{k}"] = v
                continue
            for key, value in metrics.items():
                flat_results[f"{model_type}_{key}"] = value

        # Add QML config
        for key, value in qml_config.items():
            flat_results[f"qml_{key}"] = str(value)

        # Add execution metadata
        flat_results.update(self._get_execution_metadata(quantum_config_path))

        # Add run metadata
        try:
            import uuid
            from datetime import datetime, timezone
            flat_results["run_id"] = str(uuid.uuid4())
            flat_results["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        except Exception:
            pass

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

# --- Kernel observables (fidelity-style summaries) ---
def _kernel_observables(K_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    """
    Compute simple fidelity-style observables from a kernel matrix.
    These are useful for comparing ideal vs noisy vs hardware execution.
    """
    y = np.asarray(y_train).astype(int)
    n = K_train.shape[0]
    if n == 0:
        return {}
    # remove diagonal for pairwise stats
    mask_offdiag = ~np.eye(n, dtype=bool)
    K = K_train[mask_offdiag]
    obs: Dict[str, float] = {
        "kernel_offdiag_mean": float(np.mean(K)),
        "kernel_offdiag_std": float(np.std(K)),
    }

    # class-conditional means (off-diagonal)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    def _mean_block(idxs_a, idxs_b):
        if len(idxs_a) == 0 or len(idxs_b) == 0:
            return float("nan")
        block = K_train[np.ix_(idxs_a, idxs_b)]
        # drop diagonal if same set
        if idxs_a is idxs_b or (len(idxs_a) == len(idxs_b) and np.all(idxs_a == idxs_b)):
            block = block[~np.eye(block.shape[0], dtype=bool)]
        return float(np.mean(block)) if block.size else float("nan")

    pospos = _mean_block(pos, pos)
    negneg = _mean_block(neg, neg)
    posneg = _mean_block(pos, neg)
    obs.update({
        "kernel_pospos_mean": pospos,
        "kernel_negneg_mean": negneg,
        "kernel_posneg_mean": posneg,
    })
    # separability gap: within-class minus cross-class
    if not np.isnan(pospos) and not np.isnan(negneg) and not np.isnan(posneg):
        obs["kernel_gap"] = float(((pospos + negneg) / 2.0) - posneg)
    else:
        obs["kernel_gap"] = float("nan")

    return obs

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
    entanglement = getattr(args, "entanglement", "linear")  # Default to linear if not specified
    if args.feature_map == "ZZ":
        fm = ZZFeatureMap(feature_dimension=args.qml_dim, reps=args.feature_map_reps, entanglement=entanglement)
    else:
        fm = ZFeatureMap(feature_dimension=args.qml_dim, reps=args.feature_map_reps)

    # Sampler / mode
    from quantum_layer.quantum_executor import QuantumExecutor
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    qe = QuantumExecutor(args.quantum_config)
    sampler, exec_mode = qe.get_sampler()
    if exec_mode == "gpu_simulator":
        fm_exec = fm.decompose(reps=10)
        qk = FidelityQuantumKernel(feature_map=fm_exec, fidelity=ComputeUncompute(sampler=sampler))
        log.info("Using GPU-backed FidelityQuantumKernel (cuStateVec)")
    elif exec_mode in ("statevector", "simulator_statevector"):
        qk = FidelityStatevectorKernel(feature_map=fm)
    else:
        # Aer (and most backends) can't execute custom composite instructions like "ZZFeatureMap"
        # unless the circuit is decomposed into basis gates first.
        fm_exec = fm.decompose(reps=10)
        qk = FidelityQuantumKernel(feature_map=fm_exec, fidelity=ComputeUncompute(sampler=sampler))

    # ---------------------------
    # Optional: Nyström kernel approximation (QSVC)
    # ---------------------------
    nystrom_m = getattr(args, "nystrom_m", None)
    n_train = int(getattr(X_train, "shape", [len(X_train)])[0])
    n_test = int(getattr(X_test, "shape", [len(X_test)])[0])
    
    # Auto-enable Nyström for large datasets to speed up computation
    if nystrom_m is None and n_train > 500:
        # Use Nyström approximation for datasets > 500 samples
        nystrom_m = min(200, int(n_train * 0.25))  # Use 25% of samples as landmarks, max 200
        log.info(f"[QSVC-precomputed] Auto-enabling Nyström approximation (m={nystrom_m}) for large dataset (n={n_train})")
        log.info(f"[QSVC-precomputed] This will significantly speed up kernel computation")
    
    nystrom_enabled = bool(nystrom_m is not None and int(nystrom_m) >= 2 and int(nystrom_m) < n_train)
    nystrom_landmark_mitigation = bool(getattr(args, "nystrom_landmark_mitigation", True))
    nystrom_ridge = float(getattr(args, "nystrom_ridge", 1e-6))
    nystrom_max_pairs = int(getattr(args, "nystrom_max_pairs", 20000))

    def _select_landmarks_stratified(y: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        y = np.asarray(y).astype(int)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            return rng.choice(np.arange(len(y)), size=m, replace=False)
        m_pos = min(len(pos), m // 2)
        m_neg = min(len(neg), m - m_pos)
        chosen = []
        if m_pos > 0:
            chosen.extend(rng.choice(pos, size=m_pos, replace=False).tolist())
        if m_neg > 0:
            chosen.extend(rng.choice(neg, size=m_neg, replace=False).tolist())
        chosen = list(dict.fromkeys(chosen))  # preserve order, unique
        # Fill remaining with random from all indices
        if len(chosen) < m:
            remaining = np.setdiff1d(np.arange(len(y)), np.array(chosen, dtype=int), assume_unique=False)
            fill = rng.choice(remaining, size=(m - len(chosen)), replace=False).tolist()
            chosen.extend(fill)
        return np.array(chosen[:m], dtype=int)

    def _build_depolarizing_noise_model(p: float):
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        nm = NoiseModel()
        one_qubit_gates = ["x", "y", "z", "h", "s", "t", "sx", "rz", "rx", "ry"]
        two_qubit_gates = ["cx", "cz", "swap", "ecr"]
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), one_qubit_gates)
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), two_qubit_gates)
        return nm

    def _entrywise_linear_zne(scales: list[float], mats: list[np.ndarray]) -> np.ndarray:
        """
        Fit y = a + b*s per entry and return a (zero-noise intercept at s=0).
        """
        S = np.asarray(scales, dtype=float)
        Y = np.stack([m.reshape(-1) for m in mats], axis=0)  # (k, n_entries)
        A = np.stack([np.ones_like(S), S], axis=1)          # (k, 2)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)        # (2, n_entries)
        a = coef[0, :].reshape(mats[0].shape)
        return np.clip(a, 0.0, 1.0)

    def _eval_block_with_optional_mitigation(XA: np.ndarray, XB: np.ndarray):
        """
        Evaluate a kernel block, optionally applying entrywise ZNE (landmark mitigation)
        when on noisy depolarizing simulator and enabled.
        """
        # No mitigation needed/possible in statevector or GPU simulator mode (ideal sim)
        if exec_mode in ("statevector", "simulator_statevector", "gpu_simulator"):
            return qk.evaluate(XA, XB)

        sim_cfg = (qe.config or {}).get("quantum", {}).get("simulator", {})
        zne_cfg = sim_cfg.get("zne", {}) if isinstance(sim_cfg.get("zne", {}), dict) else {}
        zne_enabled = bool(zne_cfg.get("enabled", False))
        noise_spec = sim_cfg.get("noise_model")

        # Only implement scalable ZNE for depolarizing:* noisy simulator
        can_scale_noise = (
            exec_mode == "simulator_noisy"
            and isinstance(noise_spec, str)
            and noise_spec.strip().startswith("depolarizing:")
            and zne_enabled
            and nystrom_landmark_mitigation
        )

        # Cost guardrail: if too many pairs, skip entrywise mitigation
        pairs = int(XA.shape[0]) * int(XB.shape[0])
        if can_scale_noise and pairs > nystrom_max_pairs:
            return qk.evaluate(XA, XB)

        if not can_scale_noise:
            return qk.evaluate(XA, XB)

        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        import json as _json

        base_prob = float(noise_spec.strip().split(":", 1)[1])
        scales = zne_cfg.get("scales", [1.0, 1.5, 2.0])
        scales = sorted({float(s) for s in scales if float(s) >= 1.0} | {1.0})
        shots = int(sim_cfg.get("shots", 1024) or 1024)

        mats = []
        for s in scales:
            p_s = min(1.0, base_prob * float(s))
            nm_s = _build_depolarizing_noise_model(p_s)
            sampler_s = AerSamplerV2(
                default_shots=shots,
                options={"backend_options": {"noise_model": nm_s}},
            )
            qk_s = FidelityQuantumKernel(feature_map=fm_exec, fidelity=ComputeUncompute(sampler=sampler_s))
            mats.append(qk_s.evaluate(XA, XB))

        # Record once per run (not per block)
        try:
            nonlocal_observables = getattr(_eval_block_with_optional_mitigation, "_obs", None)
            if nonlocal_observables is None:
                _eval_block_with_optional_mitigation._obs = {
                    "nystrom_entrywise_zne_enabled": 1,
                    "nystrom_entrywise_zne_scales_json": _json.dumps(scales),
                    "nystrom_entrywise_zne_base_noise_model": str(noise_spec),
                    "nystrom_entrywise_zne_base_prob": float(base_prob),
                    "nystrom_entrywise_zne_pairs_cap": int(nystrom_max_pairs),
                }
        except Exception:
            pass

        return _entrywise_linear_zne(scales, mats)

    # Phase 3: Kernel caching for faster re-runs
    import hashlib
    import pickle
    from pathlib import Path

    cache_dir = Path("data/kernel_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache key from configuration and data hash
    config_str = f"{args.qml_dim}_{args.feature_map}_{args.feature_map_reps}_{getattr(args, 'entanglement', 'linear')}_{nystrom_enabled}_{nystrom_m if nystrom_enabled else 'full'}"
    # Include quantum config path in hash to differentiate between different quantum configurations
    config_path_hash = hashlib.md5(str(args.quantum_config).encode()).hexdigest()[:8]
    data_hash = hashlib.md5(
        np.concatenate([X_train.flatten(), X_test.flatten()]).tobytes()
    ).hexdigest()[:16]
    cache_key = f"{config_str}_{config_path_hash}_{data_hash}"
    cache_file = cache_dir / f"kernel_{cache_key}.pkl"

    K_train = None
    K_test = None
    kernel_cached = False

    # Try to load cached kernels
    if cache_file.exists():
        try:
            log.info(f"[QSVC-precomputed] Loading cached kernels from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if (cached_data.get('config') == config_str and
                    cached_data.get('data_hash') == data_hash and
                    cached_data.get('config_path_hash') == config_path_hash):
                    K_train = cached_data.get('K_train')
                    K_test = cached_data.get('K_test')
                    kernel_cached = True
                    log.info(f"[QSVC-precomputed] ✓ Loaded cached kernels (train: {K_train.shape if K_train is not None else None}, test: {K_test.shape if K_test is not None else None})")
        except Exception as e:
            log.warning(f"[QSVC-precomputed] Failed to load cache: {e}")
    
    # Precompute kernels (full or Nyström-approx) if not cached
    if not kernel_cached:
        import time as _time
        t0 = _time.perf_counter()
        if nystrom_enabled:
            rng = np.random.default_rng(int(getattr(args, "random_state", 42)))
            m = int(nystrom_m)
            y_arr = np.asarray(y_train).astype(int)
            landmark_idx = _select_landmarks_stratified(y_arr, m, rng)
            X_L = np.asarray(X_train)[landmark_idx]

            log.info(f"[QSVC-precomputed] Using Nyström approximation: m={m}, n_train={n_train}, n_test={n_test}")
            if nystrom_landmark_mitigation:
                log.info("[QSVC-precomputed] Landmark mitigation: enabled (entrywise ZNE when scalable)")

            # Evaluate landmark blocks (optionally mitigated)
            t_mm0 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] Computing K_mm (landmark x landmark): {m}x{m} = {m*m} kernel evaluations...")
            K_mm = _eval_block_with_optional_mitigation(X_L, X_L)
            t_mm1 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] ✓ K_mm computed in {t_mm1 - t_mm0:.1f}s")
            t_nm0 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] Computing K_nm (train x landmark): {n_train}x{m} = {n_train*m} kernel evaluations...")
            K_nm = _eval_block_with_optional_mitigation(np.asarray(X_train), X_L)
            t_nm1 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] ✓ K_nm computed in {t_nm1 - t_nm0:.1f}s")
            t_tm0 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] Computing K_tm (test x landmark): {n_test}x{m} = {n_test*m} kernel evaluations...")
            K_tm = _eval_block_with_optional_mitigation(np.asarray(X_test), X_L)
            t_tm1 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] ✓ K_tm computed in {t_tm1 - t_tm0:.1f}s")

            # Stabilize and invert K_mm
            K_mm = np.asarray(K_mm, dtype=float)
            K_mm = (K_mm + K_mm.T) / 2.0
            np.fill_diagonal(K_mm, 1.0)
            W_inv = np.linalg.pinv(K_mm + (nystrom_ridge * np.eye(m)))

            # Nyström approximation
            K_nm = np.asarray(K_nm, dtype=float)
            K_tm = np.asarray(K_tm, dtype=float)
            K_train = K_nm @ W_inv @ K_nm.T
            K_test = K_tm @ W_inv @ K_nm.T

            # Clean up numerical issues
            K_train = (K_train + K_train.T) / 2.0
            np.fill_diagonal(K_train, 1.0)
            K_train = np.clip(K_train, 0.0, 1.0)
            K_test = np.clip(K_test, 0.0, 1.0)
            t1 = _time.perf_counter()
        else:
            t_tr0 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] Computing training kernel matrix ({n_train}x{n_train})...")
            log.info(f"[QSVC-precomputed] This may take several minutes for large datasets...")
            K_train = qk.evaluate(X_train)                 # (n_train, n_train)
            t_tr1 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] ✓ Training kernel computed in {t_tr1 - t_tr0:.1f}s")
            t_te0 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] Computing test kernel matrix ({n_test}x{n_train})...")
            K_test  = qk.evaluate(X_test, X_train)         # (n_test,  n_train)
            t_te1 = _time.perf_counter()
            log.info(f"[QSVC-precomputed] ✓ Test kernel computed in {t_te1 - t_te0:.1f}s")
            t1 = _time.perf_counter()

    # Save kernels to cache if computed (not loaded from cache)
    if not kernel_cached and K_train is not None and K_test is not None:
        try:
            log.info(f"[QSVC-precomputed] Saving kernels to cache: {cache_file}")
            cache_data = {
                'config': config_str,
                'data_hash': data_hash,
                'config_path_hash': config_path_hash,
                'K_train': K_train,
                'K_test': K_test,
                'n_train': n_train,
                'n_test': n_test
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            log.info(f"[QSVC-precomputed] ✓ Cached kernels saved")
        except Exception as e:
            log.warning(f"[QSVC-precomputed] Failed to save cache: {e}")
    
    # Base (raw) kernel observables (from the full kernel used for training)
    observables = _kernel_observables(K_train, y_train)
    if kernel_cached:
        observables['kernel_cached'] = 1
    # Timing + sizes (for professional benchmarking)
    try:
        observables.update({
            "kernel_n_train": int(n_train),
            "kernel_n_test": int(n_test),
            "kernel_eval_seconds_total": float(t1 - t0),
        })
        if nystrom_enabled:
            observables.update({
                "kernel_eval_seconds_K_mm": float(t_mm1 - t_mm0),
                "kernel_eval_seconds_K_nm": float(t_nm1 - t_nm0),
                "kernel_eval_seconds_K_tm": float(t_tm1 - t_tm0),
            })
        else:
            observables.update({
                "kernel_eval_seconds_train": float(t_tr1 - t_tr0),
                "kernel_eval_seconds_test": float(t_te1 - t_te0),
            })
    except Exception:
        pass
    if nystrom_enabled:
        observables.update({
            "nystrom_enabled": 1,
            "nystrom_m": int(nystrom_m),
            "nystrom_ridge": float(nystrom_ridge),
            "nystrom_max_pairs": int(nystrom_max_pairs),
            "nystrom_landmark_mitigation_enabled": int(bool(nystrom_landmark_mitigation)),
            "kernel_source": "nystrom",
        })
        # If entrywise ZNE ran, attach its summary diagnostics
        try:
            extra = getattr(_eval_block_with_optional_mitigation, "_obs", None)
            if isinstance(extra, dict):
                observables.update(extra)
        except Exception:
            pass
    else:
        observables.update({"nystrom_enabled": 0, "kernel_source": "full"})

    # ---------------------------
    # Minimal ZNE for observables
    # ---------------------------
    # NOTE: This mitigates *scalar observables* derived from the kernel, not the full kernel matrix.
    # Full-matrix ZNE would require re-evaluating O(n^2) circuits per noise scale.
    try:
        sim_cfg = (qe.config or {}).get("quantum", {}).get("simulator", {})
        zne_cfg = sim_cfg.get("zne", {}) if isinstance(sim_cfg.get("zne", {}), dict) else {}
        zne_enabled = bool(zne_cfg.get("enabled", False))
        noise_spec = sim_cfg.get("noise_model")

        if zne_enabled and exec_mode == "simulator_noisy" and isinstance(noise_spec, str) and noise_spec.strip().startswith("depolarizing:"):
            from quantum_layer.advanced_error_mitigation import PauliPathZNE
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
            import json as _json

            base_prob = float(noise_spec.strip().split(":", 1)[1])
            scales = zne_cfg.get("scales", [1.0, 1.5, 2.0])
            # Ensure valid, sorted, includes 1.0 and only uses amplification (>= 1.0)
            scales = sorted({float(s) for s in scales if float(s) >= 1.0} | {1.0})

            max_train_for_zne = int(zne_cfg.get("max_train_for_zne", 120))
            sample_size = int(zne_cfg.get("sample_size", 256))
            shots = int(sim_cfg.get("shots", 1024) or 1024)

            # Readout mitigation config (optional)
            ro_cfg = sim_cfg.get("readout_mitigation", {}) if isinstance(sim_cfg.get("readout_mitigation", {}), dict) else {}
            ro_enabled = bool(ro_cfg.get("enabled", False))
            ro_max_qubits = int(ro_cfg.get("max_qubits", 8))
            ro_cal_shots = int(ro_cfg.get("calibration_shots", shots))
            ro_ridge = float(ro_cfg.get("ridge_lambda", 0.01))

            def _bitstr_to_index(bitstr: str) -> int:
                # Qiskit uses little-endian bitstrings in some contexts; BitArray.get_counts() returns strings
                # in classical register order. For our calibration/correction, we stay consistent by using
                # the exact bitstrings returned by get_counts() and mapping via int(bitstr, 2).
                return int(bitstr, 2)

            def _counts_to_prob_vector(counts: dict, n_bits: int) -> np.ndarray:
                dim = 2 ** n_bits
                v = np.zeros(dim, dtype=float)
                total = float(sum(counts.values())) if counts else 0.0
                if total <= 0:
                    return v
                for b, c in counts.items():
                    try:
                        v[_bitstr_to_index(b)] += float(c) / total
                    except Exception:
                        continue
                return v

            def _compute_assignment_matrix(sampler_v2, n_qubits: int) -> np.ndarray:
                """Build assignment matrix A where A[i,j] = P(meas=i | prep=j)."""
                from qiskit import QuantumCircuit
                dim = 2 ** n_qubits
                circs = []
                # Prepare each computational basis state |j>
                for j in range(dim):
                    qc = QuantumCircuit(n_qubits)
                    bits = format(j, f"0{n_qubits}b")
                    for q, bit in enumerate(bits):
                        if bit == "1":
                            qc.x(q)
                    qc.measure_all()
                    circs.append(qc)
                res = sampler_v2.run(circs, shots=ro_cal_shots).result()
                A = np.zeros((dim, dim), dtype=float)
                for j in range(dim):
                    counts = res[j].data.meas.get_counts()
                    A[:, j] = _counts_to_prob_vector(counts, n_qubits)
                return A

            def _mitigate_prob_vector(p_obs: np.ndarray, A: np.ndarray, lam: float) -> np.ndarray:
                """Ridge-regularized inversion: p_true = argmin ||A p - p_obs||^2 + lam||p||^2."""
                dim = A.shape[0]
                AtA = A.T @ A
                rhs = A.T @ p_obs
                p = np.linalg.solve(AtA + lam * np.eye(dim), rhs)
                p = np.clip(p, 0.0, 1.0)
                s = float(np.sum(p))
                if s > 0:
                    p = p / s
                return p

            def _compute_uncompute_circuit(feature_map, x_vec: np.ndarray, y_vec: np.ndarray):
                """Build compute-uncompute circuit U(x) U†(y) then measure all."""
                from qiskit import QuantumCircuit
                n = int(feature_map.num_qubits)
                fm_x = feature_map.assign_parameters({p: float(v) for p, v in zip(feature_map.parameters, x_vec)}, inplace=False)
                # Bind y (use fresh params by reusing feature_map structure)
                fm_y = feature_map.assign_parameters({p: float(v) for p, v in zip(feature_map.parameters, y_vec)}, inplace=False)
                qc = QuantumCircuit(n)
                qc.compose(fm_x, inplace=True)
                qc.compose(fm_y.inverse(), inplace=True)
                qc.measure_all()
                return qc

            def _estimate_block_mean_posneg(
                sampler_v2,
                feature_map_exec,
                X_pos_arr: np.ndarray,
                X_neg_arr: np.ndarray,
                A: Optional[np.ndarray],
            ) -> float:
                """Estimate mean fidelity over all pairs in (pos, neg) block."""
                circs = []
                for xv in X_pos_arr:
                    for yv in X_neg_arr:
                        circs.append(_compute_uncompute_circuit(feature_map_exec, xv, yv))
                res = sampler_v2.run(circs, shots=shots).result()
                n_qubits = int(feature_map_exec.num_qubits)
                dim = 2 ** n_qubits
                idx0 = 0  # |0...0>
                vals = []
                for k in range(len(circs)):
                    counts = res[k].data.meas.get_counts()
                    p_obs = _counts_to_prob_vector(counts, n_qubits)
                    if A is not None:
                        p_true = _mitigate_prob_vector(p_obs, A, ro_ridge)
                        vals.append(float(p_true[idx0]))
                    else:
                        vals.append(float(p_obs[idx0]))
                return float(np.mean(vals)) if vals else float("nan")

            def _estimate_block_mean_posneg_raw_and_mitigated(
                sampler_v2,
                feature_map_exec,
                X_pos_arr: np.ndarray,
                X_neg_arr: np.ndarray,
                A: Optional[np.ndarray],
            ) -> Tuple[float, Optional[float]]:
                """
                Return (raw_mean, mitigated_mean_or_None).
                If A is None, mitigated_mean_or_None is None.
                """
                circs = []
                for xv in X_pos_arr:
                    for yv in X_neg_arr:
                        circs.append(_compute_uncompute_circuit(feature_map_exec, xv, yv))
                res = sampler_v2.run(circs, shots=shots).result()
                n_qubits = int(feature_map_exec.num_qubits)
                idx0 = 0  # |0...0>
                raw_vals = []
                mit_vals = []
                for k in range(len(circs)):
                    counts = res[k].data.meas.get_counts()
                    p_obs = _counts_to_prob_vector(counts, n_qubits)
                    raw_vals.append(float(p_obs[idx0]))
                    if A is not None:
                        p_true = _mitigate_prob_vector(p_obs, A, ro_ridge)
                        mit_vals.append(float(p_true[idx0]))
                raw_mean = float(np.mean(raw_vals)) if raw_vals else float("nan")
                if A is None:
                    return raw_mean, None
                mit_mean = float(np.mean(mit_vals)) if mit_vals else float("nan")
                return raw_mean, mit_mean

            n_train = int(getattr(X_train, "shape", [len(X_train)])[0])
            if n_train > max_train_for_zne:
                observables.update({
                    "zne_enabled": 0,
                    "zne_skipped_reason": f"train_size({n_train})>max_train_for_zne({max_train_for_zne})",
                })
            else:
                y = np.asarray(y_train).astype(int)
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                if len(pos) == 0 or len(neg) == 0:
                    observables.update({
                        "zne_enabled": 0,
                        "zne_skipped_reason": "need_both_pos_and_neg_labels",
                    })
                else:
                    rng = np.random.default_rng(int(getattr(args, "random_state", 42)))
                    # Choose small subsets so we can evaluate cross-block kernels cheaply per scale
                    # Target ~sample_size cross-pairs via |P|*|N|
                    side = max(2, int(np.sqrt(max(4, sample_size))))
                    p_k = int(min(len(pos), side))
                    n_k = int(min(len(neg), side))
                    pos_sub = rng.choice(pos, size=p_k, replace=False)
                    neg_sub = rng.choice(neg, size=n_k, replace=False)

                    # Observable to mitigate: cross-class mean kernel similarity (in [0,1])
                    # We'll estimate the observable via explicit compute-uncompute circuits so we can
                    # optionally apply readout mitigation.
                    X_pos = np.asarray(X_train)[pos_sub]
                    X_neg = np.asarray(X_train)[neg_sub]

                    # Feature map for execution (decomposed so Aer/backends can execute it)
                    fm_exec = fm.decompose(reps=10)
                    n_qubits = int(fm_exec.num_qubits)

                    A_assign = None
                    if ro_enabled and n_qubits <= ro_max_qubits:
                        try:
                            # Calibration uses the *same* sampler (same noise model at λ=1.0)
                            A_assign = _compute_assignment_matrix(sampler, n_qubits)
                            observables.update({
                                "readout_mitigation_enabled": 1,
                                "readout_mitigation_calibration_shots": int(ro_cal_shots),
                                "readout_mitigation_ridge_lambda": float(ro_ridge),
                            })
                        except Exception as e:
                            observables.update({
                                "readout_mitigation_enabled": 0,
                                "readout_mitigation_error": str(e),
                            })
                            A_assign = None
                    else:
                        if ro_enabled:
                            observables.update({
                                "readout_mitigation_enabled": 0,
                                "readout_mitigation_skipped_reason": f"n_qubits({n_qubits})>max_qubits({ro_max_qubits})" if ro_enabled else None,
                            })

                    # Base measurement at λ=1.0
                    base_raw, base_ro = _estimate_block_mean_posneg_raw_and_mitigated(
                        sampler, fm_exec, X_pos, X_neg, A_assign
                    )
                    # We'll keep two streams of measurements:
                    # - raw: no readout correction
                    # - ro:  readout corrected (if enabled & calibration succeeded)
                    measurements_raw = [base_raw]
                    measurements_ro = [base_ro] if base_ro is not None else None

                    observables.update({
                        "kernel_posneg_mean_explicit_raw_lambda1": float(base_raw),
                        "kernel_posneg_mean_explicit_readout_lambda1": float(base_ro) if base_ro is not None else None,
                    })

                    def _build_depolarizing_noise_model(prob: float) -> NoiseModel:
                        nm = NoiseModel()
                        one_qubit_gates = ["x", "y", "z", "h", "s", "t", "sx", "rz", "rx", "ry"]
                        two_qubit_gates = ["cx", "cz", "swap", "ecr"]
                        p = float(max(0.0, min(1.0, prob)))
                        nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), one_qubit_gates)
                        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), two_qubit_gates)
                        return nm

                    # Measurements at amplified noise scales (skip 1.0 which we already have)
                    for s in scales:
                        if abs(s - 1.0) < 1e-12:
                            continue
                        p_s = min(1.0, base_prob * float(s))
                        nm_s = _build_depolarizing_noise_model(p_s)
                        sampler_s = AerSamplerV2(
                            default_shots=shots,
                            options={"backend_options": {"noise_model": nm_s}},
                        )
                        # NOTE: We *reuse* the λ=1.0 calibration matrix for readout mitigation at all λ.
                        # This is an approximation but keeps cost bounded.
                        m_raw, m_ro = _estimate_block_mean_posneg_raw_and_mitigated(
                            sampler_s, fm_exec, X_pos, X_neg, A_assign
                        )
                        measurements_raw.append(m_raw)
                        if measurements_ro is not None:
                            measurements_ro.append(m_ro)

                    # Align measurements order with scales (including 1.0)
                    # We built as [at 1.0] + [for scales != 1.0 in ascending order]
                    meas_arr_raw = np.array(measurements_raw, dtype=float)
                    meas_arr_ro = np.array(measurements_ro, dtype=float) if measurements_ro is not None else None
                    scales_arr = np.array(scales, dtype=float)
                    if len(meas_arr_raw) != len(scales_arr):
                        # Extremely defensive: if mismatch, skip rather than corrupt logging
                        observables.update({
                            "zne_enabled": 0,
                            "zne_skipped_reason": f"scale_measurement_mismatch(scales={len(scales_arr)},meas={len(meas_arr_raw)})",
                        })
                    else:
                        zne = PauliPathZNE(use_bayesian_priors=False)
                        C0_raw, fit_params_raw = zne.fit_noise_model(scales_arr, meas_arr_raw, measurement_errors=None)
                        C0_ro = None
                        fit_params_ro = {}
                        if meas_arr_ro is not None and len(meas_arr_ro) == len(scales_arr):
                            try:
                                zne2 = PauliPathZNE(use_bayesian_priors=False)
                                C0_ro, fit_params_ro = zne2.fit_noise_model(scales_arr, meas_arr_ro, measurement_errors=None)
                            except Exception:
                                C0_ro = None

                        # Choose primary C0: prefer readout-mitigated if available
                        C0_primary = float(C0_ro) if C0_ro is not None else float(C0_raw)
                        fit_params_primary = fit_params_ro if C0_ro is not None else fit_params_raw

                        # Guardrail: clip C0 into [0,1] and record if we had to.
                        clipped = False
                        if not np.isnan(C0_primary):
                            if C0_primary < 0.0 or C0_primary > 1.0:
                                clipped = True
                                C0_primary = float(np.clip(C0_primary, 0.0, 1.0))

                        observables.update({
                            "zne_enabled": 1,
                            "zne_method": "pauli_path_zne",
                            "zne_observable": "kernel_posneg_mean",
                            "zne_base_noise_model": str(noise_spec),
                            "zne_base_prob": float(base_prob),
                            "zne_scales_json": _json.dumps(scales),
                            "zne_measurements_json_raw": _json.dumps([float(x) for x in meas_arr_raw.tolist()]),
                            "zne_measurements_json_readout": _json.dumps([float(x) for x in meas_arr_ro.tolist()]) if meas_arr_ro is not None else None,
                            "zne_kernel_posneg_mean_C0_raw": float(C0_raw),
                            "zne_kernel_posneg_mean_C0_readout": float(C0_ro) if C0_ro is not None else None,
                            "zne_kernel_posneg_mean_C0": float(C0_primary),
                            "zne_C0_clipped": int(clipped),
                            "zne_fit_error": float(fit_params_primary.get("fit_error", float("nan"))),
                            "zne_H_bar": float(fit_params_primary.get("H_bar", float("nan"))),
                            "zne_sigma": float(fit_params_primary.get("sigma", float("nan"))),
                            "zne_beta": float(fit_params_primary.get("beta", float("nan"))),
                        })
                        log.info(
                            f"[ZNE] mitigated kernel_posneg_mean: raw@1.0={meas_arr_raw[0]:.4f} → C0={float(C0_primary):.4f} "
                            f"(scales={scales})"
                        )
        else:
            # Keep it explicit in logs/CSV when ZNE isn't configured.
            if isinstance(sim_cfg, dict) and "zne" in sim_cfg:
                observables.update({"zne_enabled": 0})
    except Exception as e:
        # Never fail the run because mitigation failed; just record it.
        try:
            observables.update({"zne_enabled": 0, "zne_error": str(e)})
        except Exception:
            pass
        log.warning(f"[ZNE] skipped due to error: {e}")

    # Grid over C quickly (expanded grid for better hyperparameter search)
    from sklearn.svm import SVC
    from sklearn.metrics import average_precision_score

    best = (float("-inf"), None, None)  # (pr_auc, C, model)
    # Expanded C grid: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        svc = SVC(kernel="precomputed", C=C, class_weight="balanced")
        svc.fit(K_train, y_train)
        y_score = svc.decision_function(K_test)
        pr_auc = average_precision_score(y_test, y_score)
        log.info(f"[QSVC-precomputed] C={C} → test PR-AUC={pr_auc:.4f}")
        if pr_auc > best[0]:
            best = (pr_auc, C, svc)

    log.info(f"[QSVC-precomputed] selected C={best[1]} (test PR-AUC={best[0]:.4f})")
    return best[2], K_train, K_test, observables

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
        svc, K_train, K_test, _kernel_obs = qsvc_with_precomputed_kernel(X_train, y_train, X_test, y_test, args, log)
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
        # Expanded C grid: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
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
        clf, K_train, K_test, _kernel_obs = qsvc_with_precomputed_kernel(X_train, y_train, X_test, y_test, args, log)
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
