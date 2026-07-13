"""Tests for GeminiFileSearchStoreCrud — File Search store create/import/delete,
LRO polling, and the [GEMINI]-tagged error wrapping for google-genai SDK errors.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors, types

from app.crud.rag.gemini import GeminiFileSearchStoreCrud

STORE_NAME = "fileSearchStores/store-abc"
FILE_NAME = "files/file-xyz"


def _fast_time() -> MagicMock:
    """A `time` stand-in whose clock never advances and whose sleep is a no-op."""
    mock_time = MagicMock()
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep.return_value = None
    return mock_time


def _client_error(
    code: int, message: str, status: str = "STATUS"
) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"message": message, "status": status}})


def _server_error(code: int, message: str) -> errors.ServerError:
    return errors.ServerError(
        code, {"error": {"message": message, "status": "UNAVAILABLE"}}
    )


class TestGeminiFileSearchStoreCrudInit:
    def test_none_client_raises(self):
        with pytest.raises(ValueError):
            GeminiFileSearchStoreCrud(client=None)


class TestCreate:
    def test_returns_store_resource_name(self):
        client = MagicMock()
        client.file_search_stores.create.return_value = SimpleNamespace(name=STORE_NAME)

        assert GeminiFileSearchStoreCrud(client).create() == STORE_NAME
        _, kwargs = client.file_search_stores.create.call_args
        assert isinstance(kwargs["config"], types.CreateFileSearchStoreConfig)


class TestImportDocument:
    def test_polls_until_operation_done(self):
        client = MagicMock()
        client.file_search_stores.import_file.return_value = SimpleNamespace(done=False)
        client.operations.get.return_value = SimpleNamespace(done=True, error=None)

        with patch("app.crud.rag.gemini.time", _fast_time()):
            GeminiFileSearchStoreCrud(client).import_document(STORE_NAME, FILE_NAME)

        client.operations.get.assert_called_once()

    def test_operation_error_raises(self):
        client = MagicMock()
        client.file_search_stores.import_file.return_value = SimpleNamespace(
            done=True, error={"code": 3, "message": "corrupt file"}
        )

        with patch("app.crud.rag.gemini.time", _fast_time()):
            with pytest.raises(InterruptedError, match=r"\[GEMINI\].*corrupt file"):
                GeminiFileSearchStoreCrud(client).import_document(STORE_NAME, FILE_NAME)

    def test_timeout_raises_interrupted_error(self):
        client = MagicMock()
        client.file_search_stores.import_file.return_value = SimpleNamespace(done=False)
        client.operations.get.return_value = SimpleNamespace(done=False, error=None)

        mock_time = MagicMock()
        mock_time.monotonic.side_effect = [0.0, 10_000.0]
        with patch("app.crud.rag.gemini.time", mock_time):
            with pytest.raises(InterruptedError, match="import-timeout"):
                GeminiFileSearchStoreCrud(client).import_document(STORE_NAME, FILE_NAME)


class TestDelete:
    def test_forces_deletion(self):
        client = MagicMock()

        GeminiFileSearchStoreCrud(client).delete(STORE_NAME)

        _, kwargs = client.file_search_stores.delete.call_args
        assert kwargs["name"] == STORE_NAME
        assert kwargs["config"].force is True


class TestErrorWrapping:
    """Every google-genai SDK error surfaces as an InterruptedError whose message
    is [GEMINI]-tagged and carries the upstream code and original message."""

    @pytest.mark.parametrize(
        "error_factory, expected_code, original_message",
        [
            (lambda: _client_error(429, "quota exceeded"), 429, "quota exceeded"),
            (lambda: _client_error(403, "permission denied"), 403, "permission denied"),
            (lambda: _client_error(404, "store gone"), 404, "store gone"),
            (lambda: _client_error(400, "bad shape"), 400, "bad shape"),
            (lambda: _server_error(503, "overloaded"), 503, "overloaded"),
        ],
    )
    def test_create_wraps_sdk_errors(
        self, error_factory, expected_code, original_message
    ):
        client = MagicMock()
        client.file_search_stores.create.side_effect = error_factory()

        with pytest.raises(InterruptedError) as exc_info:
            GeminiFileSearchStoreCrud(client).create()

        msg = str(exc_info.value)
        assert msg.startswith("[GEMINI]"), msg
        assert f"code: {expected_code}" in msg, msg
        assert original_message in msg, msg
