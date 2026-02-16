"""
Optuna hyperparameter search over the optimized pipeline.

Tunes key parameters (embedding_dim, embedding_epochs, qml_dim, qsvc_C,
qml_feature_map_reps, ensemble_quantum_weight) by invoking
run_optimized_pipeline.py as a subprocess and parsing its output metrics.

Usage:
    python scripts/optuna_pipeline_search.py --n_trials 30 --objective ensemble
    python scripts/optuna_pipeline_search.py --n_trials 20 --objective qsvc
    python scripts/optuna_pipeline_search.py --n_trials 20 --objective classical

The script writes an Optuna study DB (SQLite) and a CSV of all trial results.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_optimized_pipeline.py"


def _find_python() -> str:
    """Return the Python interpreter path (prefer venv)."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _parse_pr_auc_from_log(log_text: str):
    """Extract model PR-AUC values from pipeline ranking output."""
    results = {}

    ranking_pattern = re.compile(
        r"^\d+\s+\|\s+([\w\-]+)\s+\|\s+(\w+)\s+\|\s+([\d.]+)",
        re.MULTILINE,
    )
    for match in ranking_pattern.finditer(log_text):
        model_name = match.group(1).strip()
        pr_auc = float(match.group(3))
        results[model_name] = pr_auc

    return results


def _run_pipeline(params: dict, extra_fixed_args: list) -> dict:
    """Run the pipeline with given params and return parsed PR-AUC dict."""
    python = _find_python()
    cmd = [
        python, str(PIPELINE_SCRIPT),
        "--relation", "CtD",
        "--full_graph_embeddings",
        "--embedding_method", "RotatE",
        "--negative_sampling", "hard",
        "--optimize_feature_map_reps",
        "--run_ensemble",
        "--fast_mode",
    ]

    cmd += [
        "--embedding_dim", str(params["embedding_dim"]),
        "--embedding_epochs", str(params["embedding_epochs"]),
        "--qml_dim", str(params["qml_dim"]),
        "--qsvc_C", str(params["qsvc_C"]),
        "--qml_feature_map_reps", str(params["qml_feature_map_reps"]),
        "--ensemble_method", params["ensemble_method"],
        "--ensemble_quantum_weight", str(params["ensemble_quantum_weight"]),
        "--qml_feature_map", params["qml_feature_map"],
        "--qml_reduction_method", params["qml_reduction_method"],
    ]

    if params.get("qml_pre_pca_dim", 0) > 0:
        cmd += ["--qml_pre_pca_dim", str(params["qml_pre_pca_dim"])]
    if params.get("tune_classical", False):
        cmd.append("--tune_classical")

    cmd += extra_fixed_args

    logger.info(f"Running pipeline: {' '.join(cmd[-20:])}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600
    )

    combined = result.stdout + "\n" + result.stderr
    pr_auc_dict = _parse_pr_auc_from_log(combined)

    if not pr_auc_dict:
        logger.warning("No PR-AUC values parsed from pipeline output.")
        logger.debug(f"Last 500 chars of output:\n{combined[-500:]}")

    return pr_auc_dict


def objective(trial: optuna.Trial, args) -> float:
    """Optuna objective: maximise the chosen PR-AUC metric."""

    params = {
        "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 128, 256]),
        "embedding_epochs": trial.suggest_int("embedding_epochs", 100, 300, step=50),
        "qml_dim": trial.suggest_categorical("qml_dim", [12, 16, 20]),
        "qsvc_C": trial.suggest_float("qsvc_C", 0.01, 1.0, log=True),
        "qml_feature_map_reps": trial.suggest_int("qml_feature_map_reps", 1, 4),
        "ensemble_method": trial.suggest_categorical("ensemble_method", ["weighted_average", "stacking"]),
        "ensemble_quantum_weight": trial.suggest_float("ensemble_quantum_weight", 0.2, 0.8, step=0.1),
        "qml_feature_map": trial.suggest_categorical("qml_feature_map", ["ZZ", "Pauli"]),
        "qml_reduction_method": trial.suggest_categorical("qml_reduction_method", ["pca", "kpca"]),
        "qml_pre_pca_dim": trial.suggest_categorical("qml_pre_pca_dim", [0, 24, 32]),
        "tune_classical": trial.suggest_categorical("tune_classical", [True, False]),
    }

    try:
        pr_auc_dict = _run_pipeline(params, args.extra_args)
    except subprocess.TimeoutExpired:
        logger.warning(f"Trial {trial.number} timed out.")
        return 0.0
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        return 0.0

    if not pr_auc_dict:
        return 0.0

    if args.objective == "classical":
        val = max(
            pr_auc_dict.get("RandomForest-Optimized", 0.0),
            pr_auc_dict.get("ExtraTrees-Optimized", 0.0),
        )
    elif args.objective == "qsvc":
        val = pr_auc_dict.get("QSVC-Optimized", 0.0)
    elif args.objective == "ensemble":
        val = max(v for k, v in pr_auc_dict.items() if "Ensemble" in k) if any(
            "Ensemble" in k for k in pr_auc_dict
        ) else 0.0
    else:
        val = max(pr_auc_dict.values()) if pr_auc_dict else 0.0

    for model, prauc in pr_auc_dict.items():
        trial.set_user_attr(model, prauc)

    logger.info(f"Trial {trial.number}: objective={val:.4f} | {pr_auc_dict}")
    return val


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for the pipeline")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument(
        "--objective",
        type=str,
        default="ensemble",
        choices=["classical", "qsvc", "ensemble", "best"],
        help="Which PR-AUC to maximize (default: ensemble)",
    )
    parser.add_argument("--study_name", type=str, default="pipeline_hpo", help="Optuna study name")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: sqlite in results/)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/optuna",
        help="Directory for CSV output",
    )

    args, extra = parser.parse_known_args()
    args.extra_args = extra

    results_dir = PROJECT_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{results_dir / 'optuna_study.db'}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
    )

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"  Value (PR-AUC): {study.best_value:.4f}")
    logger.info(f"  Params: {study.best_params}")

    trials_df = study.trials_dataframe()
    csv_path = results_dir / "optuna_trials.csv"
    trials_df.to_csv(csv_path, index=False)
    logger.info(f"Trials saved to {csv_path}")

    best_path = results_dir / "optuna_best.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial": study.best_trial.number,
                "n_trials": len(study.trials),
            },
            f,
            indent=2,
        )
    logger.info(f"Best params saved to {best_path}")


if __name__ == "__main__":
    main()
