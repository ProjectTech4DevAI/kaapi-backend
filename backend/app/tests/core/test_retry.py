import httpx
import pytest

from app.core.retry import is_transient

_REQUEST = httpx.Request("POST", "https://upstream.example/v1")


def _status_error(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        "upstream", request=_REQUEST, response=httpx.Response(code, request=_REQUEST)
    )


@pytest.mark.parametrize(
    "error, expected",
    [
        (_status_error(500), True),
        (_status_error(404), False),
        (httpx.ConnectError("refused"), True),
        (httpx.ConnectTimeout("slow handshake"), True),
        # The request may have reached upstream, so a retry could double-execute.
        (httpx.ReadTimeout("no response"), False),
    ],
)
def test_is_transient(error: BaseException, expected: bool) -> None:
    assert is_transient(error) is expected
