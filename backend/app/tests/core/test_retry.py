import httpx
import pytest
from tenacity import retry_if_exception_type, retry_if_result

from app.core.retry import is_transient, retry_thrice

pytestmark = pytest.mark.usefixtures("no_retry_backoff")

_REQUEST = httpx.Request("POST", "https://upstream.example/v1")


def _status_error(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        "upstream", request=_REQUEST, response=httpx.Response(code, request=_REQUEST)
    )


@pytest.mark.parametrize(
    "error, expected",
    [
        ("OpenAI error (code: 503, Service Unavailable)", True),
        ("Gemini error (code: 500 INTERNAL)", True),
        ("Anthropic overloaded (code: 529, overloaded_error)", True),
        ("rate limited (code: 429, too many requests)", False),
        ("weird (code: 5000)", False),
        ("timed out (code: APITimeoutError)", False),
        ("plain failure", False),
        (None, False),
        (_status_error(500), True),
        (_status_error(404), False),
        (httpx.ConnectError("refused"), True),
        (httpx.ConnectTimeout("slow handshake"), True),
        # The request may have reached upstream, so a retry could double-execute.
        (httpx.ReadTimeout("no response"), False),
    ],
)
def test_is_transient(error: BaseException | str | None, expected: bool) -> None:
    assert is_transient(error) is expected


def test_retry_thrice_returns_last_result_after_four_attempts() -> None:
    calls: list[int] = []

    @retry_thrice(retry_if_result(lambda r: r is None or r < 0))
    def flaky() -> int | None:
        calls.append(1)
        return None if len(calls) < 4 else -1

    assert flaky() == -1
    assert len(calls) == 4


def test_retry_thrice_reraises_original_exception_on_exhaustion() -> None:
    calls: list[int] = []

    @retry_thrice(retry_if_exception_type(ConnectionError))
    def always_down() -> None:
        calls.append(1)
        raise ConnectionError(f"attempt {len(calls)}")

    with pytest.raises(ConnectionError, match="attempt 4"):
        always_down()
    assert len(calls) == 4
