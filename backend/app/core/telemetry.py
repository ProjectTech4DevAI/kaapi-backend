import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

from app.core.config import settings

logger = logging.getLogger(__name__)


def _build_resource() -> Resource:
    return Resource.create(
        {
            SERVICE_NAME: settings.OTEL_SERVICE_NAME,
            "deployment.environment": settings.ENVIRONMENT,
            "service.version": settings.API_VERSION,
        }
    )


def _otlp_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.OTEL_EXPORTER_OTLP_AUTH_HEADER:
        headers["Authorization"] = f"Basic {settings.OTEL_EXPORTER_OTLP_AUTH_HEADER}"
    # Route data to a named stream in OpenObserve (instead of "default")
    headers["stream-name"] = settings.OTEL_SERVICE_NAME
    return headers


def setup_telemetry() -> None:
    """Initialize OpenTelemetry tracing and log export."""
    if not settings.OTEL_ENABLED:
        logger.info("[setup_telemetry] OTEL_ENABLED is False, skipping")
        return

    resource = _build_resource()
    headers = _otlp_headers()
    base_endpoint = settings.OTEL_OTLP_BASE_ENDPOINT

    # --- Traces ---
    trace_exporter = OTLPSpanExporter(
        endpoint=f"{base_endpoint}/v1/traces",
        headers=headers,
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    # --- Logs ---
    log_exporter = OTLPLogExporter(
        endpoint=f"{base_endpoint}/v1/logs",
        headers=headers,
    )
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

    # Attach OTEL handler to root logger — existing Python logging flows to OpenObserve
    otel_handler = LoggingHandler(
        level=logging.INFO,
        logger_provider=logger_provider,
    )
    logging.getLogger().addHandler(otel_handler)

    # --- Auto-instrumentation ---
    # Inject trace/span IDs into log records so they correlate with traces
    LoggingInstrumentor().instrument(set_logging_format=False)

    # Celery task tracing
    CeleryInstrumentor().instrument()

    # Outbound HTTP calls via httpx
    HTTPXClientInstrumentor().instrument()

    logger.info(
        "[setup_telemetry] OpenTelemetry initialized, stream: %s, endpoint: %s",
        settings.OTEL_SERVICE_NAME,
        base_endpoint,
    )


def instrument_app(app: object) -> None:
    """Instrument the FastAPI app. Call after app is created."""
    if not settings.OTEL_ENABLED:
        return
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]
    logger.info("[instrument_app] FastAPI instrumented with OpenTelemetry")


def instrument_db_engine(engine: object) -> None:
    """Instrument a SQLAlchemy engine. Call after engine is created."""
    if not settings.OTEL_ENABLED:
        return
    SQLAlchemyInstrumentor().instrument(engine=engine)  # type: ignore[arg-type]
    logger.info("[instrument_db_engine] SQLAlchemy engine instrumented")
