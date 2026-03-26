"""
In-memory async job manager for pipeline runs.

Each job spawns ``scripts/run_optimized_pipeline.py`` as a subprocess
and tracks its lifecycle (queued -> running -> success | failed).
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_optimized_pipeline.py"
MAX_CONCURRENT_JOBS = 1
JOB_TIMEOUT_SECONDS = 3600


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    success = "success"
    failed = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: float
    flags: Dict[str, Any]
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    log_tail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, flags: Dict[str, Any]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            status=JobStatus.queued,
            created_at=time.time(),
            flags=flags,
        )
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        return list(self._jobs.values())

    def _build_cmd(self, flags: Dict[str, Any]) -> List[str]:
        import sys

        cmd = [sys.executable, str(PIPELINE_SCRIPT)]
        for key, val in flags.items():
            arg = f"--{key}"
            if isinstance(val, bool):
                if val:
                    cmd.append(arg)
            elif val is not None:
                cmd.extend([arg, str(val)])
        return cmd

    def _run(self, job: Job) -> None:
        running = sum(
            1
            for j in self._jobs.values()
            if j.status == JobStatus.running
        )
        while running >= MAX_CONCURRENT_JOBS:
            time.sleep(2)
            running = sum(
                1
                for j in self._jobs.values()
                if j.status == JobStatus.running
            )

        job.status = JobStatus.running
        job.started_at = time.time()
        cmd = self._build_cmd(job.flags)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            tail: List[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                tail.append(line.rstrip("\n"))
                if len(tail) > 200:
                    tail.pop(0)
                job.log_tail = list(tail)

            proc.wait(timeout=JOB_TIMEOUT_SECONDS)
            job.exit_code = proc.returncode
            job.status = (
                JobStatus.success if proc.returncode == 0 else JobStatus.failed
            )
            if proc.returncode != 0:
                job.error = f"Process exited with code {proc.returncode}"
        except subprocess.TimeoutExpired:
            proc.kill()  # type: ignore[possibly-undefined]
            job.status = JobStatus.failed
            job.error = f"Timed out after {JOB_TIMEOUT_SECONDS}s"
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = time.time()


job_manager = JobManager()
