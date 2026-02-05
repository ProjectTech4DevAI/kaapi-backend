import uuid
from unittest.mock import MagicMock, patch

import pytest
import httpx

from app.services.llm.guardrails import call_guardrails
from app.core.config import settings


TEST_JOB_ID = uuid.uuid4()
TEST_TEXT = "hello world"
TEST_CONFIG = [{"type": "pii_remover"}]


@patch("app.services.llm.guardrails.httpx.Client")
def test_call_guardrails_success(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"success": True}
    mock_response.raise_for_status.return_value = None

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = call_guardrails(TEST_TEXT, TEST_CONFIG, TEST_JOB_ID)

    assert result == {"success": True}
    mock_client.post.assert_called_once()

    args, kwargs = mock_client.post.call_args

    assert kwargs["json"]["input"] == TEST_TEXT
    assert kwargs["json"]["validators"] == TEST_CONFIG
    assert kwargs["json"]["request_id"] == str(TEST_JOB_ID)

    assert kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert kwargs["headers"]["Content-Type"] == "application/json"


@patch("app.services.llm.guardrails.httpx.Client")
def test_call_guardrails_http_error_bypasses(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad", request=None, response=None
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = call_guardrails(TEST_TEXT, TEST_CONFIG, TEST_JOB_ID)

    assert result["success"] is False
    assert result["bypassed"] is True
    assert result["data"]["safe_text"] == TEST_TEXT


@patch("app.services.llm.guardrails.httpx.Client")
def test_call_guardrails_network_failure_bypasses(mock_client_cls) -> None:
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ConnectError("failed")
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = call_guardrails(TEST_TEXT, TEST_CONFIG, TEST_JOB_ID)

    assert result["bypassed"] is True
    assert result["data"]["safe_text"] == TEST_TEXT


@patch("app.services.llm.guardrails.httpx.Client")
def test_call_guardrails_timeout_bypasses(mock_client_cls) -> None:
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.TimeoutException("timeout")
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = call_guardrails(TEST_TEXT, TEST_CONFIG, TEST_JOB_ID)

    assert result["bypassed"] is True


@patch("app.services.llm.guardrails.httpx.Client")
def test_call_guardrails_uses_settings(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"ok": True}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    call_guardrails(TEST_TEXT, TEST_CONFIG, TEST_JOB_ID)

    _, kwargs = mock_client.post.call_args

    assert (
        kwargs["headers"]["Authorization"] == f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}"
    )
