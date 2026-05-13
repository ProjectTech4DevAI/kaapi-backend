import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

from asgi_correlation_id import correlation_id

from app.core.config import settings

LOG_DIR = settings.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = (
    "%(asctime)s - [%(service_name)s] - [%(correlation_id)s] - "
    "%(levelname)s - %(name)s - %(message)s"
)
_service_name_context: ContextVar[str | None] = ContextVar(
    "kaapi_service_name", default=None
)


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get() or "N/A"
        if record.name == "uvicorn.error":
            record.name = "uvicorn"
        return True


class ServiceNameFilter(logging.Filter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def set_default_service_name(self, service_name: str) -> None:
        self._service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.service_name = _service_name_context.get() or self._service_name
        return True


@contextmanager
def log_service_name(service_name: str) -> Iterator[None]:
    token = _service_name_context.set(service_name)
    try:
        yield
    finally:
        _service_name_context.reset(token)


def _set_service_name_on_existing_filters(
    root_logger: logging.Logger, service_name: str
) -> None:
    for handler in root_logger.handlers:
        for handler_filter in handler.filters:
            if isinstance(handler_filter, ServiceNameFilter):
                handler_filter.set_default_service_name(service_name)


def _configure_uvicorn_loggers() -> None:
    for logger_name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.disabled = False

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True


logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("celery.worker.consumer.connection").setLevel(logging.WARNING)
logging.getLogger("celery.worker.consumer.mingle").setLevel(logging.WARNING)
logging.getLogger("celery.apps.worker").setLevel(logging.WARNING)


def configure_logging(service_name: str | None = None) -> None:
    root_logger = logging.getLogger()
    resolved_service_name = service_name or settings.BACKEND_SERVICE_NAME

    if getattr(root_logger, "_kaapi_logging_configured", False):
        _set_service_name_on_existing_filters(root_logger, resolved_service_name)
        root_logger._kaapi_service_name = resolved_service_name
        _configure_uvicorn_loggers()
        return

    root_logger.setLevel(LOGGING_LEVEL)

    formatter = logging.Formatter(LOGGING_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(CorrelationIdFilter())
    stream_handler.addFilter(ServiceNameFilter(resolved_service_name))

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(CorrelationIdFilter())
    file_handler.addFilter(ServiceNameFilter(resolved_service_name))

    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger._kaapi_logging_configured = True
    root_logger._kaapi_service_name = resolved_service_name
    _configure_uvicorn_loggers()
