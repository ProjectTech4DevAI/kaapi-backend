import logging
import re
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import httpx
import requests
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.retry import retry_base

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Providers return errors as strings tagged `(code: NNN ...)` instead of raising.
_PROVIDER_5XX = re.compile(r"\(code: 5\d{2}\b")


def is_transient(error: BaseException | str | None) -> bool:
    """5xx from upstream, or a connection that never reached it."""
    if isinstance(error, str):
        return _PROVIDER_5XX.search(error) is not None
    if isinstance(error, (httpx.HTTPStatusError, requests.HTTPError)):
        return error.response is not None and error.response.status_code >= 500
    return isinstance(
        error, (httpx.ConnectError, httpx.ConnectTimeout, requests.ConnectionError)
    )


def retry_thrice(condition: retry_base) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """First attempt plus three retries; on exhaustion returns the last result or re-raises."""
    return retry(
        retry=condition,
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry_error_callback=lambda state: state.outcome.result(),
    )
