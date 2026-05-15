#!/usr/bin/env python3
"""
Deterministic quantum job runner for ml-intern orchestration.

The runner exposes allowlisted recipes for simulator and IBM Heron jobs. It
keeps token handling out of prompts by loading tenant-scoped IBM credentials
from the integration store and injecting them only into subprocess env vars.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_optimized_pipeline.py"
HERON_SCRIPT = PROJECT_ROOT / "scripts" / "train_on_heron.py"
DEFAULT_CHANNEL = "ibm_quantum_platform"


def build_command(
    recipe: str,
    *,
    relation: str = "CtD",
    results_dir: str = "results/ml-intern",
    quantum_config_path: Optional[str] = None,
    embedding_method: str = "RotatE",
    embedding_dim: int = 128,
    embedding_epochs: int = 200,
    qml_dim: int = 16,
    qml_feature_map: str = "Pauli",
    max_entities: Optional[int] = None,
    tenant_id: Optional[str] = None,
    qubits: int = 4,
    shots: int = 100,
    backend: str = "auto",
    model_type: str = "QSVC",
    confirm_hardware: bool = False,
) -> List[str]:
    if recipe == "simulator-smoke":
        return [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--relation",
            relation,
            "--fast_mode",
            "--quantum_config_path",
            quantum_config_path or "config/quantum_config_ideal.yaml",
            "--results_dir",
            results_dir,
        ]

    if recipe == "simulator-full":
        cmd = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--relation",
            relation,
            "--full_graph_embeddings",
            "--embedding_method",
            embedding_method,
            "--embedding_dim",
            str(embedding_dim),
            "--embedding_epochs",
            str(embedding_epochs),
            "--negative_sampling",
            "hard",
            "--quantum_only",
            "--qml_dim",
            str(qml_dim),
            "--qml_feature_map",
            qml_feature_map,
            "--quantum_config_path",
            quantum_config_path or "config/quantum_config_ideal.yaml",
            "--results_dir",
            results_dir,
        ]
        if max_entities is not None:
            cmd.extend(["--max_entities", str(max_entities)])
        return cmd

    if recipe in {"heron-dry-run", "heron-run"}:
        if recipe == "heron-run" and not confirm_hardware:
            raise ValueError("Heron hardware runs require --confirm-hardware.")
        if not tenant_id:
            raise ValueError("Heron recipes require --tenant-id.")
        cmd = [
            sys.executable,
            str(HERON_SCRIPT),
            "--relation",
            relation,
            "--max_entities",
            str(max_entities if max_entities is not None else 100),
            "--qubits",
            str(qubits),
            "--shots",
            str(shots),
            "--backend",
            backend,
            "--model_type",
            model_type,
            "--results_dir",
            results_dir,
        ]
        if recipe == "heron-dry-run":
            cmd.append("--dry_run")
        return cmd

    raise ValueError(f"Unsupported quantum job recipe: {recipe}")


def build_subprocess_env(
    tenant_id: Optional[str],
    *,
    store=None,
    base_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    if not tenant_id:
        return env

    if store is None:
        from middleware.integration_store import integration_store

        store = integration_store

    credentials = store.get_ibm_quantum_credentials(tenant_id)
    if credentials is None:
        raise ValueError(f"No IBM Quantum credentials stored for tenant '{tenant_id}'.")

    token = credentials.get("token")
    if not token:
        raise ValueError(f"Stored IBM Quantum token is empty for tenant '{tenant_id}'.")

    env["IBM_Q_TOKEN"] = token
    env["IBM_QUANTUM_TOKEN"] = token
    env["IBM_QUANTUM_CHANNEL"] = credentials.get("channel") or DEFAULT_CHANNEL
    instance = credentials.get("instance_crn")
    if instance:
        env["IBM_QUANTUM_INSTANCE"] = instance
    else:
        env.pop("IBM_QUANTUM_INSTANCE", None)
    return env


def describe_command(command: List[str]) -> str:
    redacted: List[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        redacted.append(part)
        if part in {"--token", "--api-token", "--api_token"}:
            redact_next = True
    return " ".join(_shell_quote(part) for part in redacted)


def run_recipe(args: argparse.Namespace) -> int:
    command = build_command(
        args.recipe,
        relation=args.relation,
        results_dir=args.results_dir,
        quantum_config_path=args.quantum_config_path,
        embedding_method=args.embedding_method,
        embedding_dim=args.embedding_dim,
        embedding_epochs=args.embedding_epochs,
        qml_dim=args.qml_dim,
        qml_feature_map=args.qml_feature_map,
        max_entities=args.max_entities,
        tenant_id=args.tenant_id,
        qubits=args.qubits,
        shots=args.shots,
        backend=args.backend,
        model_type=args.model_type,
        confirm_hardware=args.confirm_hardware,
    )
    print(f"Quantum job recipe: {args.recipe}")
    print(f"Command: {describe_command(command)}")
    print(f"Results dir: {args.results_dir}")
    if args.dry_run:
        print("Runner dry-run: command was not executed.")
        return 0
    env = build_subprocess_env(args.tenant_id) if args.recipe.startswith("heron-") else dict(os.environ)
    proc = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=False)
    return proc.returncode


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run allowlisted quantum job recipes.")
    parser.add_argument(
        "recipe",
        choices=["simulator-smoke", "simulator-full", "heron-dry-run", "heron-run"],
    )
    parser.add_argument("--relation", default="CtD")
    parser.add_argument("--results-dir", default="results/ml-intern")
    parser.add_argument("--quantum-config-path")
    parser.add_argument("--embedding-method", default="RotatE")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--embedding-epochs", type=int, default=200)
    parser.add_argument("--qml-dim", type=int, default=16)
    parser.add_argument("--qml-feature-map", default="Pauli")
    parser.add_argument("--max-entities", type=int)
    parser.add_argument("--tenant-id")
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model-type", default="QSVC", choices=["QSVC", "VQC"])
    parser.add_argument("--confirm-hardware", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._:-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return run_recipe(parse_args(argv))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
