import logging

import sentry_sdk
from asgi_correlation_id.middleware import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from app.api.docs.openapi_config import customize_openapi_schema, tags_metadata
from app.api.main import api_router, api_v2_router
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logger import configure_logging
from app.core.middleware import StripTrailingSlashMiddleware, http_request_logger
from app.core.sentry_filters import before_send_transaction_filter
from app.core.telemetry import instrument_app, setup_telemetry
from app.load_env import load_environment

# Load environment variables
load_environment()
configure_logging(service_name=settings.BACKEND_SERVICE_NAME)


if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=str(settings.SENTRY_DSN),
        environment=settings.ENVIRONMENT,
        release=settings.API_VERSION,
        instrumenter="otel",
        traces_sample_rate=1.0,
        enable_logs=True,
        before_send_transaction=before_send_transaction_filter,
        integrations=[
            LoggingIntegration(
                level=logging.INFO,
                sentry_logs_level=logging.INFO,
            ),
        ],
        disabled_integrations=[
            FastApiIntegration(),
            StarletteIntegration(),
            SqlalchemyIntegration(),
            CeleryIntegration(),
            HttpxIntegration(),
        ],
    )

setup_telemetry(service_name=settings.BACKEND_SERVICE_NAME)


def custom_generate_unique_id(route: APIRoute) -> str:
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    description="**Responsible AI for the development sector**",
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=settings.API_VERSION,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
    )

    openapi_schema = customize_openapi_schema(openapi_schema)

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.middleware("http")(http_request_logger)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(StripTrailingSlashMiddleware)

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_v2_router, prefix=settings.API_V2_STR)

register_exception_handlers(app)

instrument_app(app)


# health check endpoint for uptime monitoring
@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str | float]:
    return {
        "status": "ok",
    }
