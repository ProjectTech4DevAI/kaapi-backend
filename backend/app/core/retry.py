import httpx
import requests


def is_transient(error: BaseException) -> bool:
    """5xx from upstream, or a connection that never reached it."""
    if isinstance(error, (httpx.HTTPStatusError, requests.HTTPError)):
        return error.response is not None and error.response.status_code >= 500
    return isinstance(
        error, (httpx.ConnectError, httpx.ConnectTimeout, requests.ConnectionError)
    )
