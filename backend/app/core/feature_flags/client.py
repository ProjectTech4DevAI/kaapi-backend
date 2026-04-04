"""Unleash client singleton.

Initialised once at app startup via ``init_unleash()``.  All other code
accesses the client through ``get_client()``.

When Unleash is not configured (empty URL / key), a no-op stub is used
so the app can run without an Unleash server — every flag defaults to its
fallback value.
"""

import logging
from typing import Any, Protocol

from UnleashClient import UnleashClient

from app.core.config import settings

logger = logging.getLogger(__name__)


class _FeatureFlagClient(Protocol):
    """Minimal interface both the real SDK and the stub satisfy."""

    def is_enabled(
        self,
        feature_name: str,
        context: dict[str, str] | None = None,
        fallback_function: Any = None,
    ) -> bool:
        ...

    def destroy(self) -> None:
        ...


class _NoOpClient:
    """Stub used when Unleash is not configured."""

    def is_enabled(
        self,
        feature_name: str,
        context: dict[str, str] | None = None,
        fallback_function: Any = None,
    ) -> bool:
        if fallback_function is not None:
            return fallback_function(feature_name, context)
        return False

    def destroy(self) -> None:
        pass


_client: _FeatureFlagClient | None = None


def init_unleash() -> None:
    """Initialise the Unleash SDK (call once at app startup)."""
    global _client

    url = settings.UNLEASH_URL.rstrip("/") if settings.UNLEASH_URL else ""
    api_key = settings.UNLEASH_API_KEY
    app_name = settings.UNLEASH_APP_NAME

    if not url or not api_key:
        logger.warning(
            "[init_unleash] UNLEASH_URL or UNLEASH_API_KEY not set — "
            "using no-op client (all flags default to fallback)"
        )
        _client = _NoOpClient()
        return

    _client = UnleashClient(
        url=url,
        app_name=app_name,
        custom_headers={"Authorization": api_key},
    )
    _client.initialize_client()
    logger.info(f"[init_unleash] Unleash initialised | url={url} | app={app_name}")


def get_client() -> _FeatureFlagClient:
    """Return the initialised Unleash client."""
    if _client is None:
        raise RuntimeError(
            "Unleash client not initialised. Call init_unleash() at startup."
        )
    return _client


def shutdown_unleash() -> None:
    """Gracefully shut down the Unleash client."""
    global _client
    if _client is not None:
        _client.destroy()
        _client = None
        logger.info("[shutdown_unleash] Unleash client destroyed")
