from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from tinfer.support.observability import setup_json_logs

_tracer_provider_configured = False


def add_otel_ids(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    span = trace.get_current_span()
    ctx = span.get_span_context()

    if ctx and ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    else:
        event_dict["trace_id"] = None
        event_dict["span_id"] = None

    return event_dict


def setup_observability(
    service_name: str = "tinfer-server",
    environment: str = "dev",
    level: int = logging.INFO,
    force: bool = False,
) -> None:
    global _tracer_provider_configured

    if force:
        structlog.reset_defaults()

    if force or not _tracer_provider_configured:
        resource = Resource.create(
            {
                "service.name": service_name,
                "deployment.environment": environment,
            }
        )
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter(out=sys.stdout)))
        trace.set_tracer_provider(tracer_provider)
        _tracer_provider_configured = True

    GrpcInstrumentorServer().instrument()
    GrpcInstrumentorClient().instrument()
    LoggingInstrumentor().instrument(set_logging_format=False)

    setup_json_logs(level=level, force=force, processors=[add_otel_ids])
