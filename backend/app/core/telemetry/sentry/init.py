"""sentry_sdk.init() with shared sampling, profiling, PII and before_send filters."""

import logging

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.logging import LoggingIntegration

from app.core.config import settings
from app.core.telemetry.sentry.filters import (
    before_send_error_filter,
    before_send_log_filter,
    before_send_transaction_filter,
    genai_privacy_integrations,
)


def init_sentry(
    *,
    integrations: list[Integration],
    disabled_integrations: list[Integration],
) -> None:
    """Initialise the Sentry SDK with the shared sampling, privacy and filter config."""
    if not settings.SENTRY_DSN:
        return
    sentry_sdk.init(
        dsn=str(settings.SENTRY_DSN),
        environment=settings.ENVIRONMENT,
        release=settings.API_VERSION,
        instrumenter="otel",
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        sample_rate=settings.SENTRY_ERROR_SAMPLE_RATE,
        profile_session_sample_rate=settings.SENTRY_PROFILE_SESSION_SAMPLE_RATE,
        profile_lifecycle=settings.SENTRY_PROFILE_LIFECYCLE,
        send_default_pii=settings.SENTRY_SEND_DEFAULT_PII,
        enable_logs=True,
        include_local_variables=False,
        max_request_body_size="never",
        before_send=before_send_error_filter,
        before_send_transaction=before_send_transaction_filter,
        before_send_log=before_send_log_filter,
        integrations=[
            LoggingIntegration(level=logging.INFO, sentry_logs_level=logging.INFO),
            *genai_privacy_integrations(),
            *integrations,
        ],
        disabled_integrations=disabled_integrations,
    )
