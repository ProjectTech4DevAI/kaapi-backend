import logging
import os
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


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get() or "N/A"
        return True


class ServiceNameFilter(logging.Filter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.service_name = self._service_name
        return True


logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("celery.worker.consumer.connection").setLevel(logging.WARNING)
logging.getLogger("celery.worker.consumer.mingle").setLevel(logging.WARNING)
logging.getLogger("celery.apps.worker").setLevel(logging.WARNING)


def configure_logging(service_name: str | None = None) -> None:
    root_logger = logging.getLogger()
    if getattr(root_logger, "_kaapi_logging_configured", False):
        return

    root_logger.setLevel(LOGGING_LEVEL)

    formatter = logging.Formatter(LOGGING_FORMAT)
    resolved_service_name = service_name or settings.OTEL_SERVICE_NAME

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
