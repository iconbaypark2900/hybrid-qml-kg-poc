"""Liveness probes, structured logging, startup tracker.

The honesty layer: /status reads from real probes here, not hardcoded flags.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Optional

import joblib

from .async_runtime import run_io_bound
from .schemas import (
    ComponentHealth,
    HealthState,
    QuantumMode,
    QuantumStatus,
    StartupProgress,
)

if TYPE_CHECKING:
    from .settings import Settings


# ---------------------------------------------------------------------------
# Correlation ID context (read by the JSON logger)
# ---------------------------------------------------------------------------


request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
tenant_id_ctx: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)


class _ContextFilter(logging.Filter):
    """Inject request_id and tenant_id from contextvars onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        record.tenant_id = tenant_id_ctx.get() or "-"
        return True


def setup_logging(settings: "Settings") -> None:
    """Wire stdout + optional rotating JSONL file + optional HTTP push.

    Sinks:
      - stdout (always)
      - rotating JSONL file at HETQML_LOG_FILE (when set)
      - HTTP POST to HETQML_LOG_HTTP_URL (Loki/Vector/Datadog ingest, when set)

    Format on stdout follows settings.log_format ("json" or "text"). The
    file sink is always JSONL. The HTTP sink batches and ships records in
    JSONL form too.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    json_fmt = _json_formatter()
    text_fmt = _text_formatter()
    ctx_filter = _ContextFilter()

    handlers: list[logging.Handler] = []

    # Stdout
    sh = logging.StreamHandler()
    sh.setFormatter(json_fmt if settings.log_format == "json" else text_fmt)
    sh.addFilter(ctx_filter)
    handlers.append(sh)

    # Rotating JSONL file
    log_file = os.environ.get("HETQML_LOG_FILE")
    if log_file:
        try:
            from logging.handlers import RotatingFileHandler
            max_bytes = int(os.environ.get("HETQML_LOG_FILE_MAX_BYTES", str(50 * 1024 * 1024)))
            backups = int(os.environ.get("HETQML_LOG_FILE_BACKUPS", "5"))
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups)
            fh.setFormatter(json_fmt)
            fh.addFilter(ctx_filter)
            handlers.append(fh)
        except Exception as e:
            sh.handle(logging.LogRecord(
                "service.logging", logging.WARNING, __file__, 0,
                f"failed to attach file log handler: {e}", (), None,
            ))

    # HTTP push (Loki/Vector/Datadog)
    http_url = os.environ.get("HETQML_LOG_HTTP_URL")
    if http_url:
        try:
            hh = _HttpJsonlHandler(
                url=http_url,
                token=os.environ.get("HETQML_LOG_HTTP_TOKEN"),
                batch_size=int(os.environ.get("HETQML_LOG_HTTP_BATCH", "20")),
                flush_interval=float(os.environ.get("HETQML_LOG_HTTP_FLUSH_S", "5.0")),
            )
            hh.setFormatter(json_fmt)
            hh.addFilter(ctx_filter)
            handlers.append(hh)
        except Exception as e:
            sh.handle(logging.LogRecord(
                "service.logging", logging.WARNING, __file__, 0,
                f"failed to attach http log handler: {e}", (), None,
            ))

    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


def _json_formatter() -> logging.Formatter:
    try:
        from pythonjsonlogger import jsonlogger
        return jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(request_id)s %(tenant_id)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
        )
    except Exception:
        return _text_formatter()


class _HttpJsonlHandler(logging.Handler):
    """Best-effort JSONL HTTP shipper. Batches records and POSTs them.

    Compatible with Loki's `/loki/api/v1/push` (with adapter), Vector's
    HTTP source, and Datadog's logs intake (single-record mode). For Loki,
    set HETQML_LOG_HTTP_URL=http://loki:3100/loki/api/v1/push and supply a
    transformer in the future — current shape is one JSON object per line,
    Newline-Delimited JSON.

    Failures don't propagate — log shipping is never allowed to break the
    process. Drops records silently if the queue overflows.
    """

    def __init__(self, url: str, token: Optional[str], batch_size: int, flush_interval: float):
        super().__init__()
        self.url = url
        self.token = token
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.5, flush_interval)
        self._queue: list[str] = []
        self._lock = Lock()
        self._stop = False
        import threading as _t
        self._thread = _t.Thread(target=self._flush_loop, daemon=True, name="hetqml-log-http")
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        with self._lock:
            if len(self._queue) > 5000:
                return  # overflow drop
            self._queue.append(msg)

    def _flush_loop(self) -> None:
        import time as _time
        try:
            import httpx
        except ImportError:
            return
        client = httpx.Client(timeout=5.0)
        while not self._stop:
            _time.sleep(self.flush_interval)
            with self._lock:
                if not self._queue:
                    continue
                batch = self._queue[: self.batch_size]
                del self._queue[: self.batch_size]
            body = "\n".join(batch).encode("utf-8")
            headers = {"Content-Type": "application/x-ndjson"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            try:
                client.post(self.url, content=body, headers=headers)
            except Exception:
                # Drop on the floor; never break the process.
                pass


def _text_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [req=%(request_id)s tenant=%(tenant_id)s] %(message)s"
    )


# ---------------------------------------------------------------------------
# Startup tracker
# ---------------------------------------------------------------------------


class StartupTracker:
    """Tracks lifespan loading progress. /status reads its snapshot."""

    def __init__(self, total_steps: int):
        self.total = total_steps
        self.loaded = 0
        self.current_step = "init"
        self.started_at = time.time()
        self.state: str = "starting"
        self._lock = Lock()

    def step(self, name: str) -> None:
        with self._lock:
            self.current_step = name
            self.loaded += 1

    def mark_ready(self) -> None:
        with self._lock:
            self.state = "ready"
            if self.loaded < self.total:
                self.loaded = self.total

    def mark_failed(self, reason: str) -> None:
        with self._lock:
            self.state = "failed"
            self.current_step = f"failed: {reason}"

    def is_ready(self) -> bool:
        return self.state == "ready"

    def snapshot(self) -> StartupProgress:
        with self._lock:
            state_lit = self.state if self.state in ("starting", "ready", "failed") else "starting"
            return StartupProgress(
                state=state_lit,  # type: ignore[arg-type]
                components_loaded=self.loaded,
                components_total=self.total,
                last_step=self.current_step,
                elapsed_seconds=time.time() - self.started_at,
            )


# ---------------------------------------------------------------------------
# Liveness probes
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.time()


async def probe_entity_mappings(state) -> ComponentHealth:
    name = "entity_mappings"
    res = getattr(state, "entity_resolver", None)
    if res is None:
        return ComponentHealth(name=name, state=HealthState.UNAVAILABLE,
                               detail="resolver not initialized", last_checked=_now())
    drug_count = len(res.drug_ids)
    disease_count = len(res.disease_ids)
    if drug_count == 0 or disease_count == 0:
        return ComponentHealth(name=name, state=HealthState.DEGRADED,
                               detail=f"empty mappings (drugs={drug_count}, diseases={disease_count})",
                               last_checked=_now())
    return ComponentHealth(name=name, state=HealthState.OK,
                           detail=f"drugs={drug_count}, diseases={disease_count}",
                           last_checked=_now())


async def probe_embedder(state) -> ComponentHealth:
    name = "embedder"
    emb = getattr(state, "embedder", None)
    if emb is None:
        return ComponentHealth(name=name, state=HealthState.UNAVAILABLE,
                               detail="embedder not loaded", last_checked=_now())
    # Check that the embedder has vectors
    n = getattr(emb, "num_entities", None) or len(getattr(emb, "entity_to_id", {}) or {})
    if not n:
        return ComponentHealth(name=name, state=HealthState.DEGRADED,
                               detail="embedder has no entities", last_checked=_now())
    return ComponentHealth(name=name, state=HealthState.OK,
                           detail=f"entities={n}", last_checked=_now())


async def probe_classical_model(state) -> ComponentHealth:
    name = "classical_model"
    chain = getattr(state, "manifest_chain", None)
    if chain is None:
        return ComponentHealth(name=name, state=HealthState.UNAVAILABLE,
                               detail="no active manifest chain "
                                      "(run synthesize_manifest_chain)",
                               last_checked=_now())
    settings: "Settings" = state.settings
    path = settings.artifacts_dir / "runs" / chain.model_id / "model.joblib"

    def _check() -> ComponentHealth:
        if not path.exists():
            return ComponentHealth(name=name, state=HealthState.UNAVAILABLE,
                                   detail=f"missing: {path}", last_checked=_now())
        try:
            joblib.load(path)
        except Exception as e:
            return ComponentHealth(name=name, state=HealthState.DEGRADED,
                                   detail=f"corrupt: {type(e).__name__}: {e}",
                                   last_checked=_now())
        return ComponentHealth(name=name, state=HealthState.OK,
                               detail=f"active model: {chain.model_id}",
                               last_checked=_now())

    return await run_io_bound(_check)


async def probe_quantum_model(state) -> ComponentHealth:
    name = "quantum_model"
    quantum_chain = getattr(state, "quantum_chain", None)
    if quantum_chain is None:
        return ComponentHealth(
            name=name, state=HealthState.UNAVAILABLE,
            detail="no quantum manifest active; quantum_strict will 503",
            last_checked=_now(),
        )
    settings: "Settings" = state.settings
    weights = settings.artifacts_dir / "runs" / quantum_chain.model_id / "quantum_weights.npz"
    if not weights.exists():
        return ComponentHealth(
            name=name, state=HealthState.DEGRADED,
            detail=f"quantum manifest active but weights missing at {weights}",
            last_checked=_now(),
        )
    return ComponentHealth(name=name, state=HealthState.OK,
                           detail=f"active model: {quantum_chain.model_id}",
                           last_checked=_now())


async def probe_ibm_runtime(state) -> ComponentHealth:
    """Live IBM Runtime probe with caching.

    Calls QiskitRuntimeService.backends() with a tight timeout (default 3s,
    settable via Settings.ibm_probe_timeout_seconds). Result is cached for
    60s on `state._ibm_probe_cache` so /status doesn't pay network cost on
    every poll.
    """
    name = "ibm_runtime"
    settings: "Settings" = state.settings
    token = os.environ.get(settings.ibm_quantum_token_env)
    if not token:
        return ComponentHealth(name=name, state=HealthState.UNAVAILABLE,
                               detail=f"{settings.ibm_quantum_token_env} not set",
                               last_checked=_now())

    cache = getattr(state, "_ibm_probe_cache", None)
    cache_ttl = 60.0
    if cache is not None and (_now() - cache["ts"]) < cache_ttl:
        return cache["health"]

    timeout = float(getattr(settings, "ibm_probe_timeout_seconds", 3.0))

    def _ping() -> tuple[bool, str]:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            instance = os.environ.get(settings.ibm_quantum_instance_env)
            kw: dict = {"channel": "ibm_quantum", "token": token}
            if instance:
                kw["instance"] = instance
            svc = QiskitRuntimeService(**kw)
            backends = svc.backends()
            return True, f"reachable; {len(list(backends))} backends"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:80]}"

    try:
        reachable, detail = await asyncio.wait_for(
            asyncio.to_thread(_ping), timeout=timeout,
        )
    except asyncio.TimeoutError:
        reachable, detail = False, f"probe timed out (>{timeout:.1f}s)"
    except Exception as e:
        reachable, detail = False, f"probe error: {type(e).__name__}"

    health = ComponentHealth(
        name=name,
        state=HealthState.OK if reachable else HealthState.DEGRADED,
        detail=detail,
        last_checked=_now(),
    )
    state._ibm_probe_cache = {"ts": _now(), "health": health}

    # Mirror the result onto the executor so probe_quantum_status sees it.
    qe = getattr(state, "quantum_executor", None)
    if qe is not None:
        try:
            qe._ibm_reachable = reachable
            qe._ibm_initialized = True
        except Exception:
            pass

    return health


async def probe_manifest_store(state) -> ComponentHealth:
    name = "manifest_store"
    settings: "Settings" = state.settings
    runs_dir = settings.artifacts_dir / "runs"
    if not runs_dir.exists():
        return ComponentHealth(name=name, state=HealthState.DEGRADED,
                               detail=f"no runs directory at {runs_dir}",
                               last_checked=_now())
    chain = getattr(state, "manifest_chain", None)
    if chain is None:
        return ComponentHealth(name=name, state=HealthState.DEGRADED,
                               detail="LATEST.txt missing or unreadable",
                               last_checked=_now())
    return ComponentHealth(name=name, state=HealthState.OK,
                           detail=f"active model: {chain.model_id}",
                           last_checked=_now())


async def probe_all_components(state) -> list[ComponentHealth]:
    """Run every probe. Order is documented; clients shouldn't depend on it."""
    return [
        await probe_entity_mappings(state),
        await probe_embedder(state),
        await probe_classical_model(state),
        await probe_quantum_model(state),
        await probe_ibm_runtime(state),
        await probe_manifest_store(state),
    ]


def derive_overall(tracker: StartupTracker, components: list[ComponentHealth]) -> HealthState:
    if tracker.state == "failed":
        return HealthState.UNAVAILABLE
    if tracker.state == "starting":
        return HealthState.LOADING
    states = {c.state for c in components}
    if HealthState.UNAVAILABLE in states:
        return HealthState.DEGRADED
    if HealthState.DEGRADED in states:
        return HealthState.DEGRADED
    if HealthState.LOADING in states:
        return HealthState.LOADING
    return HealthState.OK


def probe_quantum_status(state, tenant=None) -> QuantumStatus:
    """Build the QuantumStatus payload, filtered by tenant quota.

    Without a tenant, returns the full status (used internally / by tests).
    With a tenant, hides modes the tenant cannot use.
    """
    qe = getattr(state, "quantum_executor", None)
    available: list[QuantumMode] = []
    note: Optional[str] = None

    if qe is None:
        available = [QuantumMode.UNAVAILABLE]
        note = "quantum executor not initialized"
    else:
        if getattr(qe, "_simulator_available", True):
            available.append(QuantumMode.SIMULATOR)
        if getattr(qe, "_gpu_simulator_available", False):
            available.append(QuantumMode.GPU_SIMULATOR)
        if getattr(qe, "_ibm_reachable", False):
            available.append(QuantumMode.IBM_HERON)
        if not available:
            available = [QuantumMode.UNAVAILABLE]

    quantum_chain = getattr(state, "quantum_chain", None)
    if quantum_chain is None:
        note = (note + "; " if note else "") + "no quantum manifest active; quantum_strict will 503"
    else:
        weights = state.settings.artifacts_dir / "runs" / quantum_chain.model_id / "quantum_weights.npz"
        if not weights.exists():
            note = (note + "; " if note else "") + (
                f"quantum manifest active but weights missing at {weights}"
            )

    if tenant is not None:
        if not tenant.quota.can_use_ibm_hardware:
            available = [m for m in available if m != QuantumMode.IBM_HERON]
        if not tenant.quota.can_use_quantum_strict and len(available) > 1:
            # Still show simulator as a fallback target
            pass
        if not available:
            available = [QuantumMode.UNAVAILABLE]

    return QuantumStatus(
        default_mode=available[0],
        available_modes=available,
        ibm_runtime_reachable=QuantumMode.IBM_HERON in available,
        last_calibration_check=getattr(qe, "_last_calibration_ts", None) if qe else None,
        note=note,
    )
