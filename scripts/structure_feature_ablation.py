#!/usr/bin/env python3
"""Run or print a fast A/B ablation for local structure-derived features.

Default behavior is a dry run. Use ``--execute`` after generating a local
structure feature CSV with ``scripts/build_structure_features.py`` or by passing
``--registry`` so this script can build it first.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PARENT = "results/structure_feature_ablation"
DEFAULT_STRUCTURE_OUT = "results/structure_features"


@dataclass(frozen=True)
class AblationCommand:
    label: str
    command: list[str]
    results_dir: Path

    def shell_line(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


def build_feature_command(
    *,
    registry: str,
    out_dir: str,
    python_executable: str,
) -> list[str]:
    return [
        python_executable,
        "scripts/build_structure_features.py",
        "--registry",
        _command_path(registry),
        "--out-dir",
        _command_path(out_dir),
    ]


def build_ablation_commands(
    *,
    results_parent: str = DEFAULT_RESULTS_PARENT,
    structure_features_path: str = f"{DEFAULT_STRUCTURE_OUT}/target_structure_features.csv",
    python_executable: str = sys.executable,
    relation: str = "CtD",
    pos_edge_sample: int = 100,
    pos_edge_sample_strategy: str = "compound_stratified",
    embedding_method: str = "RotatE",
    embedding_dim: int = 128,
    embedding_epochs: int = 200,
    negative_sampling: str = "hard",
    full_graph_embeddings: bool = True,
    max_entities: int = 0,
    cheap_mode: bool = False,
    include_quantum: bool = False,
    use_cached_embeddings: bool = False,
) -> list[AblationCommand]:
    parent = Path(results_parent)
    common = [
        python_executable,
        "scripts/run_optimized_pipeline.py",
        "--relation",
        relation,
        "--embedding_method",
        embedding_method,
        "--embedding_dim",
        str(embedding_dim),
        "--embedding_epochs",
        str(embedding_epochs),
        "--negative_sampling",
        negative_sampling,
        "--pos_edge_sample",
        str(pos_edge_sample),
        "--pos_edge_sample_strategy",
        pos_edge_sample_strategy,
        "--fast_mode",
    ]
    if full_graph_embeddings:
        common.append("--full_graph_embeddings")
    if max_entities > 0:
        common.extend(["--max_entities", str(max_entities)])
    if cheap_mode:
        common.append("--cheap_mode")
    if use_cached_embeddings:
        common.append("--use_cached_embeddings")
    if not include_quantum:
        common.append("--classical_only")

    baseline_results = parent / "baseline"
    structure_results = parent / "structure"
    baseline = AblationCommand(
        label="A_baseline",
        command=common + ["--results_dir", baseline_results.as_posix()],
        results_dir=baseline_results,
    )
    structure = AblationCommand(
        label="B_structure",
        command=common
        + [
            "--use_structure_features",
            "--structure_features_path",
            _command_path(structure_features_path),
            "--results_dir",
            structure_results.as_posix(),
        ],
        results_dir=structure_results,
    )
    return [baseline, structure]


def _print_command(label: str, command: list[str]) -> None:
    print(f"# {label}")
    print(" ".join(shlex.quote(part) for part in command))
    print()


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _command_path(path: str) -> str:
    return path.replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument("--registry", default=None, help="Optional local structure artifact registry to build first.")
    parser.add_argument("--structure-out-dir", default=DEFAULT_STRUCTURE_OUT)
    parser.add_argument(
        "--structure-features-path",
        default=None,
        help="Existing target_structure_features.csv. Defaults to <structure-out-dir>/target_structure_features.csv.",
    )
    parser.add_argument("--results-parent", default=DEFAULT_RESULTS_PARENT)
    parser.add_argument("--relation", default="CtD")
    parser.add_argument("--pos-edge-sample", type=int, default=100)
    parser.add_argument(
        "--pos-edge-sample-strategy",
        default="compound_stratified",
        choices=["uniform", "compound_stratified"],
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--embedding-epochs", type=int, default=200)
    parser.add_argument("--max-entities", type=int, default=0, help="Forwarded to run_optimized_pipeline.py.")
    parser.add_argument(
        "--task-specific-embeddings",
        action="store_true",
        help="Omit --full_graph_embeddings for a faster smoke run.",
    )
    parser.add_argument(
        "--cheap-mode",
        action="store_true",
        help="Forward --cheap_mode to run_optimized_pipeline.py.",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Bounded wiring smoke: task-specific embeddings, fast-mode epochs, max_entities=300, cheap mode.",
    )
    parser.add_argument("--include-quantum", action="store_true", help="Include quantum path in both A/B runs.")
    parser.add_argument("--use-cached-embeddings", action="store_true")
    args = parser.parse_args()

    full_graph_embeddings = not args.task_specific_embeddings
    embedding_epochs = args.embedding_epochs
    max_entities = args.max_entities
    cheap_mode = args.cheap_mode
    if args.smoke_mode:
        full_graph_embeddings = False
        embedding_epochs = min(embedding_epochs, 50)
        max_entities = max_entities or 300
        cheap_mode = True

    structure_features_path = (
        _command_path(args.structure_features_path)
        if args.structure_features_path
        else (Path(args.structure_out_dir) / "target_structure_features.csv").as_posix()
    )

    print(f"cwd: {REPO_ROOT}")
    print(f"EXECUTE={1 if args.execute else 0}")
    print(f"results_parent={_command_path(args.results_parent)}")
    print(f"structure_features_path={structure_features_path}")
    print()

    if args.registry:
        build_command = build_feature_command(
            registry=args.registry,
            out_dir=args.structure_out_dir,
            python_executable=args.python,
        )
        if args.execute:
            print(f">>> Building structure features from {args.registry}")
            _run(build_command)
        else:
            _print_command("build_structure_features", build_command)

    commands = build_ablation_commands(
        results_parent=args.results_parent,
        structure_features_path=structure_features_path,
        python_executable=args.python,
        relation=args.relation,
        pos_edge_sample=args.pos_edge_sample,
        pos_edge_sample_strategy=args.pos_edge_sample_strategy,
        embedding_epochs=embedding_epochs,
        full_graph_embeddings=full_graph_embeddings,
        max_entities=max_entities,
        cheap_mode=cheap_mode,
        include_quantum=args.include_quantum,
        use_cached_embeddings=args.use_cached_embeddings,
    )

    if args.execute and not Path(structure_features_path).exists():
        raise SystemExit(
            f"Structure feature CSV not found: {structure_features_path}. "
            "Pass --registry to build it first or --structure-features-path to an existing CSV."
        )

    for command in commands:
        if args.execute:
            print(f">>> Running {command.label} -> {command.results_dir}")
            _run(command.command)
        else:
            _print_command(command.label, command.command)

    if not args.execute:
        print("# Dry run only. Add --execute to run the two ablation conditions.")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
