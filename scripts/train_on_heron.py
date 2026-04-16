#!/usr/bin/env python3
"""
CLI entry point for training QSVC or VQC on IBM Quantum Heron hardware.

Includes token validation, backend reachability check, cost estimate,
hard negative generation via kg_layer, result persistence, and provenance
registration.

Usage:
    # Dry-run (validate token + backend, no jobs submitted)
    IBM_Q_TOKEN=<token> python scripts/train_on_heron.py \\
        --relation CtD --max_entities 100 --qubits 4 --shots 100 --dry_run

    # Real hardware run
    IBM_Q_TOKEN=<token> python scripts/train_on_heron.py \\
        --relation CtD --max_entities 200 --qubits 4 --model_type QSVC \\
        --feature_map ZZ --shots 2000
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, get_hard_negatives
from kg_layer.kg_embedder import HetionetEmbedder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_token() -> str:
    """Return IBM Quantum token from environment, or empty string."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    for var in ("IBM_Q_TOKEN", "IBM_QUANTUM_TOKEN"):
        val = os.environ.get(var, "").strip().strip('"').strip("'")
        if val and val != "your_actual_token_here":
            return val
    return ""


def _preflight(args) -> None:
    """Validate token, backend reachability, and print cost estimate.

    Exits on failure or when ``--dry_run`` is set.
    """
    token = _resolve_token()
    if not token:
        logger.error(
            "IBM Quantum token not found. Set IBM_Q_TOKEN or IBM_QUANTUM_TOKEN "
            "in your environment or .env file."
        )
        sys.exit(1)

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        logger.error(
            "qiskit_ibm_runtime is not installed. "
            "Run: pip install qiskit-ibm-runtime"
        )
        sys.exit(1)

    logger.info("Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    available = [b.name for b in service.backends()]
    logger.info("Available backends (%d): %s", len(available), ", ".join(available))

    if args.backend not in available:
        logger.error(
            "Backend '%s' not found. Available: %s", args.backend, ", ".join(available)
        )
        sys.exit(1)

    logger.info("Backend '%s' reachable.", args.backend)

    if args.model_type.upper() == "VQC":
        total_shots = args.shots * args.max_iter
        trainable = args.qubits * args.ansatz_reps * 2
        logger.info(
            "Cost estimate: %d shots x ~%d VQC iterations -> O(%d) total shots, "
            "%d trainable parameters.",
            args.shots, args.max_iter, total_shots, trainable,
        )
    else:
        logger.info(
            "Cost estimate: QSVC kernel matrix evaluation, %d shots per circuit pair.",
            args.shots,
        )

    if args.feature_map_reps > 1:
        logger.warning(
            "feature_map_reps=%d may produce deep circuits on hardware. "
            "Consider --feature_map_reps 1 to reduce decoherence risk.",
            args.feature_map_reps,
        )

    if args.dry_run:
        logger.info("--dry_run: pre-flight passed. Exiting without submitting jobs.")
        sys.exit(0)


def _write_heron_config(base_config_path: str, backend: str, shots: int) -> str:
    """Write a temporary quantum config YAML for the Heron run."""
    import yaml

    cfg = {
        "quantum": {
            "execution_mode": "heron",
            "heron": {
                "backend": backend,
                "shots": shots,
                "max_runtime_minutes": 30,
                "use_dynamical_decoupling": True,
            },
            "ibm_quantum": {
                "token": "${IBM_Q_TOKEN}",
                "channel": "ibm_quantum_platform",
            },
        }
    }

    if os.path.isfile(base_config_path):
        try:
            with open(base_config_path) as f:
                base = yaml.safe_load(f) or {}
            ibm_sec = base.get("quantum", {}).get("ibm_quantum", {})
            if ibm_sec:
                cfg["quantum"]["ibm_quantum"].update(ibm_sec)
        except Exception:
            pass

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="quantum_config_heron_",
        dir="config", delete=False,
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser(
        description="Train QSVC/VQC on IBM Quantum Heron hardware"
    )

    data = parser.add_argument_group("Data")
    data.add_argument("--relation", default="CtD")
    data.add_argument("--max_entities", type=int, default=200)
    data.add_argument("--embedding_dim", type=int, default=16)
    data.add_argument("--negative_sampling", default="degree_corrupt",
                       choices=["random", "degree_corrupt", "type_aware"])
    data.add_argument("--results_dir", default="results")
    data.add_argument("--model_dir", default="models")

    circ = parser.add_argument_group("Circuit")
    circ.add_argument("--qubits", type=int, default=4)
    circ.add_argument("--model_type", default="QSVC", choices=["QSVC", "VQC"])
    circ.add_argument("--feature_map", default="ZZ", choices=["ZZ", "Pauli", "Z"])
    circ.add_argument("--feature_map_reps", type=int, default=1)
    circ.add_argument("--ansatz_reps", type=int, default=2)
    circ.add_argument("--optimizer", default="SPSA", choices=["SPSA", "COBYLA"])
    circ.add_argument("--max_iter", type=int, default=25)
    circ.add_argument("--shots", type=int, default=2000)

    hw = parser.add_argument_group("Backend")
    hw.add_argument("--backend", default="ibm_torino")

    flags = parser.add_argument_group("Flags")
    flags.add_argument("--dry_run", action="store_true")
    flags.add_argument("--random_state", type=int, default=42)
    flags.add_argument("--quantum_config_path", default="config/quantum_config.yaml")

    args = parser.parse_args()

    # ── Pre-flight ────────────────────────────────────────────────────
    _preflight(args)

    # ── Data ──────────────────────────────────────────────────────────
    logger.info("Loading Hetionet edges...")
    df = load_hetionet_edges()
    task_edges, entity_to_id, id_to_entity = extract_task_edges(
        df, relation_type=args.relation, max_entities=args.max_entities,
    )
    logger.info("Task edges: %d, entities: %d", len(task_edges), len(entity_to_id))

    pos = task_edges.copy()
    pos["label"] = 1
    pos_train, pos_test = train_test_split(
        pos, test_size=0.2, random_state=args.random_state,
    )

    neg_train = get_hard_negatives(
        pos_train[["source", "target"]],
        strategy=args.negative_sampling,
        num_negatives=len(pos_train),
        random_state=args.random_state,
    )
    neg_test = get_hard_negatives(
        pos_test[["source", "target"]],
        strategy=args.negative_sampling,
        num_negatives=len(pos_test),
        random_state=args.random_state + 1,
    )

    import pandas as pd
    train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(
        frac=1, random_state=args.random_state,
    )
    test_df = pd.concat([pos_test, neg_test], ignore_index=True).sample(
        frac=1, random_state=args.random_state,
    )

    # ── Embeddings ────────────────────────────────────────────────────
    logger.info("Preparing embeddings (dim=%d, qml_dim=%d)...", args.embedding_dim, args.qubits)
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=args.qubits)
    if not embedder.load_saved_embeddings(expected_dim=args.embedding_dim):
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()

    X_train = embedder.prepare_link_features_qml(train_df)
    X_test = embedder.prepare_link_features_qml(test_df)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # ── Quantum model ─────────────────────────────────────────────────
    tmp_config = _write_heron_config(args.quantum_config_path, args.backend, args.shots)
    logger.info("Heron config written to: %s", tmp_config)

    try:
        from quantum_layer.qml_model import QMLLinkPredictor

        model = QMLLinkPredictor(
            model_type=args.model_type,
            num_qubits=args.qubits,
            feature_map_type=args.feature_map,
            feature_map_reps=args.feature_map_reps,
            ansatz_reps=args.ansatz_reps,
            optimizer=args.optimizer,
            max_iter=args.max_iter,
            random_state=args.random_state,
            quantum_config_path=tmp_config,
        )

        logger.info("Training %s on %s (%d train, %d test)...",
                     args.model_type, args.backend, len(X_train), len(X_test))
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        pr_auc = float(average_precision_score(y_test, y_proba))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        logger.info("PR-AUC:  %.4f", pr_auc)
        logger.info("ROC-AUC: %.4f", roc_auc)

        # ── Persist results ───────────────────────────────────────────
        os.makedirs(args.results_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(
            args.results_dir, f"heron_{args.model_type.lower()}_{stamp}.json",
        )
        payload = {
            "backend": args.backend,
            "model_type": args.model_type,
            "qubits": args.qubits,
            "feature_map": args.feature_map,
            "feature_map_reps": args.feature_map_reps,
            "shots": args.shots,
            "negative_sampling": args.negative_sampling,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "timestamp": stamp,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Results saved to: %s", out_path)

        # ── Provenance ────────────────────────────────────────────────
        try:
            from scripts.benchmark_registry import register_run

            register_run(
                run_id=stamp,
                relation=args.relation,
                embedding={
                    "method": "RotatE",
                    "dim": args.embedding_dim,
                    "epochs": None,
                    "full_graph": False,
                },
                reduction={
                    "method": "PCA",
                    "pre_pca_dim": None,
                    "output_dim": args.qubits,
                },
                model={
                    "name": f"{args.model_type}-Heron",
                    "type": "quantum",
                    "pr_auc": pr_auc,
                },
                backend={
                    "name": args.backend,
                    "execution_mode": "heron",
                    "shots": args.shots,
                    "noise_model": None,
                },
                metrics={"pr_auc": pr_auc, "roc_auc": roc_auc},
                negative_sampling={
                    "strategy": args.negative_sampling,
                    "ratio": 1.0,
                },
                circuit={
                    "n_qubits": args.qubits,
                    "feature_map": args.feature_map,
                    "feature_map_reps": args.feature_map_reps,
                    "ansatz_reps": args.ansatz_reps if args.model_type.upper() == "VQC" else None,
                    "optimizer": args.optimizer if args.model_type.upper() == "VQC" else None,
                    "max_iter": args.max_iter if args.model_type.upper() == "VQC" else None,
                },
                split={
                    "test_size": 0.20,
                    "random_state": args.random_state,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                },
                notes=f"train_on_heron.py backend={args.backend}",
            )
            logger.info("Provenance registered.")
        except Exception as e:
            logger.warning("Could not register run: %s", e)

    finally:
        try:
            os.unlink(tmp_config)
        except OSError:
            pass


if __name__ == "__main__":
    main()
