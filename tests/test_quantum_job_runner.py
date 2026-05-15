from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from middleware.integration_store import IntegrationStore
from scripts.quantum_job_runner import (
    PROJECT_ROOT as RUNNER_ROOT,
    build_command,
    build_subprocess_env,
    describe_command,
    main as runner_main,
)


def main() -> None:
    simulator = build_command(
        "simulator-smoke",
        relation="CtD",
        results_dir="results/ml-intern/smoke",
    )
    assert simulator[:2] == [
        sys.executable,
        str(RUNNER_ROOT / "scripts" / "run_optimized_pipeline.py"),
    ]
    assert "--fast_mode" in simulator
    assert simulator[simulator.index("--quantum_config_path") + 1] == "config/quantum_config_ideal.yaml"
    assert simulator[simulator.index("--results_dir") + 1] == "results/ml-intern/smoke"

    heron = build_command(
        "heron-dry-run",
        relation="CtD",
        tenant_id="tenant-a",
        max_entities=100,
        qubits=4,
        shots=100,
        backend="ibm_torino",
    )
    assert heron[:2] == [
        sys.executable,
        str(RUNNER_ROOT / "scripts" / "train_on_heron.py"),
    ]
    assert "--dry_run" in heron
    assert heron[heron.index("--backend") + 1] == "ibm_torino"

    auto_heron = build_command(
        "heron-dry-run",
        relation="CtD",
        tenant_id="tenant-a",
    )
    assert auto_heron[auto_heron.index("--backend") + 1] == "auto"

    try:
        build_command("heron-run", relation="CtD", tenant_id="tenant-a")
    except ValueError as exc:
        assert "--confirm-hardware" in str(exc)
    else:
        raise AssertionError("Heron hardware runs must require explicit confirmation.")

    with tempfile.TemporaryDirectory() as tmp:
        store = IntegrationStore(Path(tmp) / "state.db")
        store.save_ibm_quantum_credentials(
            tenant_id="tenant-a",
            token="ibm-token-secret",
            instance_crn="crn:v1:tenant-a",
        )
        env = build_subprocess_env("tenant-a", store=store)
        assert env["IBM_Q_TOKEN"] == "ibm-token-secret"
        assert env["IBM_QUANTUM_TOKEN"] == "ibm-token-secret"
        assert env["IBM_QUANTUM_INSTANCE"] == "crn:v1:tenant-a"
        assert env["IBM_QUANTUM_CHANNEL"] == "ibm_quantum_platform"

    redacted = describe_command(["python", "script.py", "--token", "ibm-token-secret"])
    assert "ibm-token-secret" not in redacted
    assert "[REDACTED]" in redacted

    assert (
        runner_main(
            [
                "heron-dry-run",
                "--tenant-id",
                "missing-tenant",
                "--dry-run",
            ]
        )
        == 0
    )

    script_result = subprocess.run(
        [
            sys.executable,
            "scripts/quantum_job_runner.py",
            "heron-dry-run",
            "--tenant-id",
            "missing-tenant",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert script_result.returncode == 2
    assert "No IBM Quantum credentials stored" in script_result.stderr
    assert "ModuleNotFoundError" not in script_result.stderr


if __name__ == "__main__":
    main()
