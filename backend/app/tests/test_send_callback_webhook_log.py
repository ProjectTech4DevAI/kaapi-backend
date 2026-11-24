from unittest.mock import MagicMock, patch

import pytest
import requests
from sqlmodel import Session, select

from app.models.webhook_log import (
    WebhookDeliveryStatusEnum,
    WebhookLog,
    WebhookResourceTypeEnum,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.utils import random_lower_string
from app.utils import send_callback


class _NonClosingSession:
    def __init__(self, session: Session):
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture(autouse=True)
def no_webhook_log() -> None:
    """Override conftest's no-op so the real webhook_log write runs here."""


@pytest.fixture
def callback_url(db: Session, no_retry_backoff: None):
    # _save_webhook_log opens its own Session(engine); route it into the
    # transactional db fixture so rows are visible here and rolled back after.
    with (
        patch("app.utils.Session", side_effect=lambda _e: _NonClosingSession(db)),
        patch("app.utils.validate_callback_url"),
    ):
        yield f"https://hooks.example.com/{random_lower_string()}"


def _response(status_code: int, text: str) -> MagicMock:
    return MagicMock(status_code=status_code, text=text)


def _http_error(status_code: int, text: str) -> requests.HTTPError:
    return requests.HTTPError(
        f"{status_code} Server Error", response=_response(status_code, text)
    )


def _logs(db: Session, url: str) -> list[WebhookLog]:
    return list(db.exec(select(WebhookLog).where(WebhookLog.url == url)).all())


def test_rejected_url_records_rejected_row(db: Session, callback_url: str) -> None:
    with (
        patch("app.utils.validate_callback_url", side_effect=ValueError("private IP")),
        patch("app.utils._post_callback") as post,
    ):
        assert send_callback(callback_url, {"x": 1}) is False

    post.assert_not_called()  # SSRF guard: no request may leave
    [log] = _logs(db, callback_url)
    assert log.status == WebhookDeliveryStatusEnum.REJECTED
    assert log.attempt_count == 0
    assert log.status_code is None
    assert log.latency is None
    assert log.error == "private IP"


def test_delivered_first_try(db: Session, callback_url: str) -> None:
    with patch("app.utils._post_callback", return_value=_response(200, "ok")):
        assert send_callback(callback_url, {"x": 1}) is True

    [log] = _logs(db, callback_url)
    assert log.status == WebhookDeliveryStatusEnum.DELIVERED
    assert log.attempt_count == 1
    assert log.status_code == 200
    assert log.latency is not None and log.latency >= 0
    assert log.response_body == "ok"
    assert log.error is None


def test_delivered_after_retries_counts_attempts(
    db: Session, callback_url: str
) -> None:
    outcomes = [
        _http_error(503, "busy"),
        requests.ConnectionError("reset"),
        _response(200, "ok"),
    ]
    with patch("app.utils._post_callback", side_effect=outcomes):
        assert send_callback(callback_url, {"x": 1}) is True

    [log] = _logs(db, callback_url)
    assert log.status == WebhookDeliveryStatusEnum.DELIVERED
    assert log.attempt_count == 3
    assert log.status_code == 200


def test_exhausted_http_error_records_failed_with_truncated_body(
    db: Session, callback_url: str
) -> None:
    with patch(
        "app.utils._post_callback", side_effect=_http_error(500, "e" * 2500)
    ) as post:
        assert send_callback(callback_url, {"x": 1}) is False

    assert post.call_count == 4
    [log] = _logs(db, callback_url)
    assert log.status == WebhookDeliveryStatusEnum.FAILED
    assert log.attempt_count == 4
    assert log.status_code == 500
    assert log.response_body == "e" * 2000
    assert log.error == "500 Server Error"


def test_exhausted_connection_error_has_no_status_code(
    db: Session, callback_url: str
) -> None:
    with patch(
        "app.utils._post_callback", side_effect=requests.ConnectionError("refused")
    ):
        assert send_callback(callback_url, {"x": 1}) is False

    [log] = _logs(db, callback_url)
    assert log.status == WebhookDeliveryStatusEnum.FAILED
    assert log.attempt_count == 4
    assert log.status_code is None
    assert log.latency is not None and log.latency >= 0
    assert log.response_body is None
    assert "refused" in log.error


@pytest.mark.parametrize("secret, signed", [("s3cret", True), (None, False)])
def test_attribution_and_request_metadata_stored(
    db: Session,
    callback_url: str,
    user_api_key: TestAuthContext,
    secret: str | None,
    signed: bool,
) -> None:
    payload = {"success": True, "data": {"answer": "hi"}}
    with patch("app.utils._post_callback", return_value=_response(200, "")):
        send_callback(
            callback_url,
            payload,
            secret,
            resource_type=WebhookResourceTypeEnum.LLM_CALL,
            resource_id="job-123",
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            request_metadata={"conversation_id": "c-1"},
        )

    [log] = _logs(db, callback_url)
    assert log.signed is signed
    assert log.resource_type == WebhookResourceTypeEnum.LLM_CALL
    assert log.resource_id == "job-123"
    assert log.project_id == user_api_key.project_id
    assert log.organization_id == user_api_key.organization_id
    assert log.request_metadata == {"conversation_id": "c-1"}
    # compact JSON: {"success":true,"data":{"answer":"hi"}}
    assert log.request_bytes == 39


def test_request_metadata_null_when_omitted(db: Session, callback_url: str) -> None:
    with patch("app.utils._post_callback", return_value=_response(200, "ok")):
        send_callback(callback_url, {"request_metadata": {"leak": True}})

    [log] = _logs(db, callback_url)
    # payload's own request_metadata key must not be picked up; only the kwarg counts
    assert log.request_metadata is None


def test_payload_is_not_persisted() -> None:
    assert "request_body" not in WebhookLog.__table__.columns
    assert not hasattr(WebhookLog, "request_body")


def test_log_write_failure_does_not_change_outcome(
    db: Session, callback_url: str
) -> None:
    with (
        patch("app.utils._post_callback", return_value=_response(200, "ok")),
        patch(
            "app.crud.webhook_log.create_webhook_log",
            side_effect=RuntimeError("db down"),
        ),
    ):
        assert send_callback(callback_url, {"x": 1}) is True

    assert _logs(db, callback_url) == []
