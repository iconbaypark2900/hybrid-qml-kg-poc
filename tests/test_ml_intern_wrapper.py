from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/ml_intern_quantum_job.sh",
            "simulator-smoke",
            "--dry-run",
            "--results-dir",
            "results/ml-intern/test",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert "ml-intern dry-run" in result.stdout
    assert "python3 scripts/quantum_job_runner.py simulator-smoke" in result.stdout
    assert "--results-dir results/ml-intern/test" in result.stdout
    assert "Do not print IBM Quantum tokens" in result.stdout


if __name__ == "__main__":
    main()
