import re

import httpx
import requests

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
