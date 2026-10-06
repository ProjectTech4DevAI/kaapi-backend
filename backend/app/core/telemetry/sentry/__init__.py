"""Sentry SDK wiring: init_sentry() shared by API and Celery worker."""

from app.core.telemetry.sentry.init import init_sentry

__all__ = ["init_sentry"]
