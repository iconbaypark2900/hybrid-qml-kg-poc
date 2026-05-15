"""OpenTelemetry tracing setup. Best-effort: only activates when both
opentelemetry-* packages are installed AND a settings.tracing_enabled flag
is true. Without those, the helpers are no-ops so the service runs the same.

Once the partner ops team is ready, set:
    HETQML_TRACING_ENABLED=1
    HETQML_OTLP_ENDPOINT=https://otel-collector.example.com:4317

and the service emits spans for every HTTP request, every orchestrator
predict, every quantum_executor mode resolution, etc.

Spans automatically inherit the request_id and tenant_id contextvars from
service.observability so traces correlate with logs.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)

_initialized = False
_tracer = None


def is_enabled() -> bool:
    return os.environ.get("HETQML_TRACING_ENABLED", "0") in ("1", "true", "yes")


def setup_tracing(service_name: str = "hetqml-service") -> bool:
    """Initialize OTLP exporter + FastAPI instrumentation. Idempotent.

    Returns True if tracing is wired (and `_tracer` is set), False otherwise.
    """
    global _initialized, _tracer
    if _initialized:
        return _tracer is not None
    _initialized = True

    if not is_enabled():
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError as e:
        log.warning(
            "tracing requested but opentelemetry packages missing: %s. "
            "Install: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-grpc "
            "opentelemetry-instrumentation-fastapi",
            e,
        )
        return False

    endpoint = os.environ.get("HETQML_OTLP_ENDPOINT", "http://localhost:4317")
    insecure = os.environ.get("HETQML_OTLP_INSECURE", "1") in ("1", "true", "yes")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    log.info("tracing enabled, exporting to %s", endpoint)
    return True


def instrument_app(app) -> None:
    """Call after `create_app()` to add automatic FastAPI request spans.
    Safe no-op if tracing isn't installed/enabled."""
    if not is_enabled():
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        log.info("fastapi tracing instrumentation enabled")
    except ImportError:
        pass


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Optional[Any]]:
    """Open a trace span if tracing is wired, else a no-op context.
    Use for orchestrator/predict/quantum-executor sections.

        with span("orchestrator.predict", tenant_id=tid, drug_id=d, disease_id=di):
            ...
    """
    if _tracer is None:
        yield None
        return
    from opentelemetry import trace as _trace  # local import; lib may be absent
    with _tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            try:
                sp.set_attribute(k, v)
            except Exception:
                pass
        # Add the contextvars for log correlation.
        from .observability import request_id_ctx, tenant_id_ctx
        rid = request_id_ctx.get()
        tid = tenant_id_ctx.get()
        if rid:
            sp.set_attribute("hetqml.request_id", rid)
        if tid:
            sp.set_attribute("hetqml.tenant_id", tid)
        yield sp
