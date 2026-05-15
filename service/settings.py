"""Service settings — env-driven via Pydantic Settings (manual since
pydantic-settings isn't pinned). Loads from process env + optional .env file.

Every config value the service reads at runtime is declared here so /status
can fingerprint the loaded config (config_hash) without scraping environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


class Settings(BaseModel):
    """Immutable settings object — built once at app startup."""
    model_config = ConfigDict(frozen=True)

    version: str = "0.1.0"
    git_sha: str = "unknown"
    config_hash: str = "unknown"

    repo_root: Path = DEFAULT_REPO_ROOT
    artifacts_dir: Path = DEFAULT_REPO_ROOT / "artifacts"
    legacy_models_dir: Path = DEFAULT_REPO_ROOT / "models"
    legacy_results_dir: Path = DEFAULT_REPO_ROOT / "results"
    config_dir: Path = DEFAULT_REPO_ROOT / "config"
    tenants_path: Path = DEFAULT_REPO_ROOT / "secrets" / "tenants.yaml"
    tenants_example_path: Path = DEFAULT_REPO_ROOT / "tenants.example.yaml"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    thread_pool_workers: int = 16
    process_pool_workers: int = 4
    job_workers: int = 2

    request_timeout_seconds: float = 30.0
    ibm_probe_timeout_seconds: float = 3.0

    # Quantum/IBM config — read but never echoed in /status
    ibm_quantum_token_env: str = "IBM_QUANTUM_TOKEN"
    ibm_quantum_instance_env: str = "IBM_QUANTUM_INSTANCE"
    quantum_config_path: Path = DEFAULT_REPO_ROOT / "config" / "quantum_config.yaml"

    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "text"

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        env: dict[str, object] = {}
        if (v := _read_env("HETQML_ARTIFACTS_DIR")):
            env["artifacts_dir"] = Path(v)
        if (v := _read_env("HETQML_TENANTS_PATH")):
            env["tenants_path"] = Path(v)
        if (v := _read_env("HETQML_CORS_ORIGINS")):
            env["cors_origins"] = [o.strip() for o in v.split(",") if o.strip()]
        if (v := _read_env("HETQML_THREAD_POOL_WORKERS")):
            env["thread_pool_workers"] = int(v)
        if (v := _read_env("HETQML_PROCESS_POOL_WORKERS")):
            env["process_pool_workers"] = int(v)
        if (v := _read_env("HETQML_JOB_WORKERS")):
            env["job_workers"] = int(v)
        if (v := _read_env("HETQML_LOG_LEVEL")):
            env["log_level"] = v.upper()
        if (v := _read_env("HETQML_LOG_FORMAT")):
            env["log_format"] = v.lower()
        if (v := _read_env("HETQML_QUANTUM_CONFIG")):
            env["quantum_config_path"] = Path(v)

        merged = {**env, **overrides}
        merged["git_sha"] = _git_sha(merged.get("repo_root", DEFAULT_REPO_ROOT))
        merged["config_hash"] = _config_hash(merged)
        return cls(**merged)


def _config_hash(values: dict) -> str:
    """Stable hash of all config values that affect runtime behavior.

    Excludes git_sha (recorded separately) and Path objects are normalized
    to their resolved string form so reorderings don't perturb the hash.
    """
    keys = [k for k in values.keys() if k != "git_sha" and k != "config_hash"]
    payload = {}
    for k in sorted(keys):
        v = values[k]
        if isinstance(v, Path):
            payload[k] = str(v)
        elif isinstance(v, list):
            payload[k] = list(v)
        else:
            payload[k] = v
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
