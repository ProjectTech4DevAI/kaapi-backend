"""
Tests for OpenAIVectorStoreCrud.update — focused on the failure-path error
enrichment (per-file reasons + category-prefixed messages for each specific
OpenAI exception type).
"""

from unittest.mock import MagicMock

import openai
import pytest

from app.crud.rag.open_ai import OpenAIVectorStoreCrud


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def crud(mock_client):
    return OpenAIVectorStoreCrud(client=mock_client)


def _make_doc(file_id: str, fname: str) -> MagicMock:
    doc = MagicMock()
    doc.file_id = {"openai": file_id}
    doc.fname = fname
    return doc


@pytest.fixture
def docs():
    return [
        _make_doc("file-1", "file1.pdf"),
        _make_doc("file-2", "file2.pdf"),
    ]


def _batch_result(*, completed: int, failed: int, batch_id: str = "batch_abc"):
    """Mock the return of vector_stores.file_batches.upload_and_poll."""
    counts = MagicMock(completed=completed, failed=failed)
    return MagicMock(id=batch_id, file_counts=counts)


def _failed_file(file_id: str, error_message: str | None):
    """Build one failed-file row. last_error=None means 'no reason recorded'."""
    f = MagicMock()
    f.id = file_id
    f.last_error = MagicMock(message=error_message) if error_message else None
    return f


class TestOpenAIVectorStoreCrudUpdateSuccess:
    def test_completes_when_all_files_complete(self, crud, mock_client, docs):
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=2, failed=0)
        )

        crud.update("vs_1", docs)

        _, kwargs = mock_client.vector_stores.file_batches.upload_and_poll.call_args
        assert kwargs["vector_store_id"] == "vs_1"
        assert kwargs["file_ids"] == ["file-1", "file-2"]
        # list_files should not have been called on the happy path
        mock_client.vector_stores.file_batches.list_files.assert_not_called()

    def test_skips_upload_when_no_docs(self, crud, mock_client):
        crud.update("vs_1", [])
        mock_client.vector_stores.file_batches.upload_and_poll.assert_not_called()


class TestOpenAIVectorStoreCrudUpdatePartialFailure:
    """Failed files -> RuntimeError with per-file reasons labelled by fname."""

    def test_includes_failed_fnames_and_messages(self, crud, mock_client, docs):
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=1, failed=1)
        )
        mock_client.vector_stores.file_batches.list_files.return_value = [
            _failed_file("file-1", "Unsupported file type"),
            _failed_file("file-2", "File too large"),
        ]

        with pytest.raises(RuntimeError) as exc_info:
            crud.update("vs_1", docs)

        msg = str(exc_info.value)
        assert "file1.pdf: Unsupported file type" in msg
        assert "file2.pdf: File too large" in msg

    def test_reports_no_error_detail_when_last_error_missing(
        self, crud, mock_client, docs
    ):
        """A failed file with no `last_error` shouldn't drop out of the
        summary — it gets 'no error detail' so the user sees that something
        was wrong with that file even if OpenAI didn't tell us what."""
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=1, failed=1)
        )
        mock_client.vector_stores.file_batches.list_files.return_value = [
            _failed_file("file-1", None)
        ]

        with pytest.raises(RuntimeError, match="file1.pdf: no error detail"):
            crud.update("vs_1", docs)

    def test_falls_back_to_file_id_label_for_unknown_file(
        self, crud, mock_client, docs
    ):
        """A failed file ID not matching any doc is labelled by its file ID."""
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=1, failed=1)
        )
        mock_client.vector_stores.file_batches.list_files.return_value = [
            _failed_file("file-unknown", "parse error")
        ]

        with pytest.raises(RuntimeError, match="file-unknown: parse error"):
            crud.update("vs_1", docs)

    def test_reraises_when_list_files_errors(self, crud, mock_client, docs):
        """If the follow-up list_files lookup itself raises, the OpenAI error
        propagates instead of masking the real upload problem."""
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=0, failed=2)
        )
        mock_client.vector_stores.file_batches.list_files.side_effect = (
            openai.OpenAIError("list failed")
        )

        with pytest.raises(openai.OpenAIError, match="list failed"):
            crud.update("vs_1", docs)


class TestOpenAIVectorStoreCrudUpdateOpenAIExceptions:
    """`upload_and_poll` raising each specific OpenAI exception type maps to
    `InterruptedError` with a category-prefixed message that includes the
    upstream status code and a remediation hint.

    Assertions are deliberately structural (prefix + code + original message)
    rather than exact-string equality so future tweaks to the remediation
    wording don't break the suite.
    """

    @pytest.mark.parametrize(
        "exception_factory, expected_prefix, expected_status, original_message",
        [
            (
                lambda: openai.RateLimitError(
                    message="quota exceeded",
                    response=MagicMock(
                        status_code=429, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "[OPENAI] Rate limit exceeded",
                429,
                "quota exceeded",
            ),
            (
                lambda: openai.AuthenticationError(
                    message="bad api key",
                    response=MagicMock(
                        status_code=401, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "[OPENAI] Authentication failed",
                401,
                "bad api key",
            ),
            (
                lambda: openai.NotFoundError(
                    message="missing resource",
                    response=MagicMock(
                        status_code=404, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "[OPENAI] Resource not found",
                404,
                "missing resource",
            ),
            (
                lambda: openai.BadRequestError(
                    message="invalid file",
                    response=MagicMock(
                        status_code=400, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "[OPENAI] Bad request",
                400,
                "invalid file",
            ),
            (
                lambda: openai.UnprocessableEntityError(
                    message="cannot process",
                    response=MagicMock(
                        status_code=422, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "[OPENAI] Unprocessable entity",
                422,
                "cannot process",
            ),
            (
                lambda: openai.InternalServerError(
                    message="upstream boom",
                    response=MagicMock(
                        status_code=500, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "[OPENAI] Server error",
                500,
                "upstream boom",
            ),
        ],
    )
    def test_specific_openai_exception_maps_to_category_prefix(
        self,
        crud,
        mock_client,
        docs,
        exception_factory,
        expected_prefix,
        expected_status,
        original_message,
    ):
        mock_client.vector_stores.file_batches.upload_and_poll.side_effect = (
            exception_factory()
        )

        with pytest.raises(InterruptedError) as exc_info:
            crud.update("vs_1", docs)
        msg = str(exc_info.value)
        assert msg.startswith(expected_prefix), msg
        assert f"code: {expected_status}" in msg, msg
        assert original_message in msg, msg

    def test_api_timeout_error(self, crud, mock_client, docs):
        """APITimeoutError doesn't expose .message — handler interpolates str(e)."""
        mock_client.vector_stores.file_batches.upload_and_poll.side_effect = (
            openai.APITimeoutError(request=MagicMock())
        )

        with pytest.raises(InterruptedError) as exc_info:
            crud.update("vs_1", docs)
        assert str(exc_info.value).startswith("[KAAPI] OpenAI request timed out")

    def test_generic_openai_error_falls_through(self, crud, mock_client, docs):
        """Any OpenAIError subclass without a dedicated handler lands in the
        bottom-most `except openai.OpenAIError` block — prefixed with the
        generic "OpenAI error" tag but still carrying the original message.
        """
        mock_client.vector_stores.file_batches.upload_and_poll.side_effect = (
            openai.OpenAIError("something else")
        )

        with pytest.raises(InterruptedError) as exc_info:
            crud.update("vs_1", docs)
        msg = str(exc_info.value)
        assert msg.startswith("[OPENAI] SDK error:"), msg
        assert "something else" in msg, msg


class TestOpenAIVectorStoreCrudInit:
    """The base OpenAICrud init rejects a None client; subclasses inherit it."""

    def test_none_client_raises(self):
        with pytest.raises(ValueError):
            OpenAIVectorStoreCrud(client=None)
