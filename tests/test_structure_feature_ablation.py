from __future__ import annotations

from scripts.structure_feature_ablation import (
    build_ablation_commands,
    build_feature_command,
)


def test_build_feature_command_uses_local_registry_and_out_dir() -> None:
    command = build_feature_command(
        registry="tests/fixtures/structure_artifacts/registry.json",
        out_dir="results/structure_features_fixture",
        python_executable="python",
    )

    assert command == [
        "python",
        "scripts/build_structure_features.py",
        "--registry",
        "tests/fixtures/structure_artifacts/registry.json",
        "--out-dir",
        "results/structure_features_fixture",
    ]


def test_ablation_commands_build_baseline_and_structure_conditions() -> None:
    commands = build_ablation_commands(
        results_parent="results/ablation",
        structure_features_path="results/structure_features/target_structure_features.csv",
        python_executable="python",
        pos_edge_sample=25,
    )

    baseline, structure = commands
    assert baseline.label == "A_baseline"
    assert structure.label == "B_structure"
    assert "--use_structure_features" not in baseline.command
    assert "--use_structure_features" in structure.command
    assert "--structure_features_path" in structure.command
    assert baseline.command[0:2] == ["python", "scripts/run_optimized_pipeline.py"]
    assert "--classical_only" in baseline.command
    assert "--classical_only" in structure.command
    assert baseline.command[baseline.command.index("--pos_edge_sample") + 1] == "25"
    assert baseline.command[baseline.command.index("--results_dir") + 1] == "results/ablation/baseline"
    assert structure.command[structure.command.index("--results_dir") + 1] == "results/ablation/structure"


def test_ablation_commands_can_include_quantum_path() -> None:
    commands = build_ablation_commands(
        results_parent="results/ablation",
        structure_features_path="features.csv",
        python_executable="python",
        include_quantum=True,
        use_cached_embeddings=True,
    )

    for command in commands:
        assert "--classical_only" not in command.command
        assert "--use_cached_embeddings" in command.command


def test_ablation_commands_support_bounded_smoke_mode_shape() -> None:
    commands = build_ablation_commands(
        results_parent="results/ablation",
        structure_features_path="features.csv",
        python_executable="python",
        full_graph_embeddings=False,
        max_entities=300,
        embedding_epochs=5,
        cheap_mode=True,
    )

    baseline = commands[0]
    assert "--full_graph_embeddings" not in baseline.command
    assert baseline.command[baseline.command.index("--embedding_epochs") + 1] == "5"
    assert baseline.command[baseline.command.index("--max_entities") + 1] == "300"
    assert "--cheap_mode" in baseline.command
