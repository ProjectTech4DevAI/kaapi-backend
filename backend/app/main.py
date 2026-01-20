import sentry_sdk

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.openapi.utils import get_openapi
from asgi_correlation_id.middleware import CorrelationIdMiddleware
from app.api.main import api_router
from app.api.docs.openapi_config import tags_metadata, customize_openapi_schema
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.middleware import http_request_logger

from app.load_env import load_environment

# Load environment variables
load_environment()


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


if settings.SENTRY_DSN and settings.ENVIRONMENT != "development":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

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

app.include_router(api_router, prefix=settings.API_V1_STR)

register_exception_handlers(app)
