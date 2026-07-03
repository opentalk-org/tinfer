from __future__ import annotations

import logging
import sys
import warnings
from contextlib import contextmanager
from typing import Any

import structlog


def get_logger(name: str):
    return structlog.get_logger(name)


@contextmanager
def start_span(name: str, module_name: str, kind: str | None = None, attributes: dict[str, Any] | None = None):
    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind
    except ImportError:
        yield None
        return

    span_kind = None
    if kind == "server":
        span_kind = SpanKind.SERVER

    tracer = trace.get_tracer(module_name)
    if span_kind is None:
        with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span
    else:
        with tracer.start_as_current_span(name, kind=span_kind, attributes=attributes or {}) as span:
            yield span


def record_span_exception(span: Any, exc: Exception) -> None:
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        return
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def setup_json_logs(
    level: int = logging.INFO,
    force: bool = False,
    processors: list[Any] | None = None,
) -> None:
    pre_chain = [
        structlog.contextvars.merge_contextvars,
    ]
    if processors:
        pre_chain.extend(processors)

    rendering_processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=pre_chain + rendering_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if force:
        root_logger.handlers.clear()
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        for existing_handler in root_logger.handlers:
            existing_handler.setFormatter(formatter)
            existing_handler.setLevel(level)
    root_logger.setLevel(level)

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    structlog.configure(
        processors=pre_chain
        + rendering_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=not force,
    )
