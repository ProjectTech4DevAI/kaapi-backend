"""
Tests for OpenAIVectorStoreCrud.update — focused on the failure-path error
enrichment (per-file reasons + category-prefixed messages for each specific
OpenAI exception type).
"""

from unittest.mock import MagicMock, patch

import openai
import pytest
from tenacity import stop_after_attempt, wait_none

from app.crud.rag.open_ai import (
    BATCH_INDEX_MAX_ATTEMPTS,
    BATCH_POLL_INTERVAL_SECONDS,
    OpenAIVectorStoreCrud,
    _provider_file_id,
)
from app.tests.utils.openai import get_mock_openai_client_with_vector_store


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def crud(mock_client):
    return OpenAIVectorStoreCrud(client=mock_client)


@pytest.fixture(autouse=True)
def _single_attempt_batch_retry():
    """_create_and_index_batch is tenacity-retried; default tests to a single
    attempt with no backoff. TestBatchRetry re-enables retries explicitly."""
    retrying = OpenAIVectorStoreCrud._create_and_index_batch.retry
    stop, wait = retrying.stop, retrying.wait
    retrying.stop, retrying.wait = stop_after_attempt(1), wait_none()
    yield
    retrying.stop, retrying.wait = stop, wait


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


REAL_BATCH_ID = "vsfb_real123"
CORRUPT_POLL_ID = "vs_corrupt999"


def _wire_batch(mock_client, *, completed: int, failed: int) -> None:
    """create() gives the real vsfb_ id; retrieve() gives the corrupt vs_ id."""
    mock_client.vector_stores.file_batches.create.return_value = MagicMock(
        id=REAL_BATCH_ID
    )
    counts = MagicMock(completed=completed, failed=failed, in_progress=0)
    mock_client.vector_stores.file_batches.retrieve.return_value = MagicMock(
        id=CORRUPT_POLL_ID, status="completed", file_counts=counts
    )


def _failed_file(file_id: str, error_message: str | None):
    """Build one failed-file row. last_error=None means 'no reason recorded'."""
    f = MagicMock()
    f.id = file_id
    f.last_error = MagicMock(message=error_message) if error_message else None
    return f


class TestOpenAIVectorStoreCrudUpdateSuccess:
    def test_completes_when_all_files_complete(self, crud, mock_client, docs):
        _wire_batch(mock_client, completed=2, failed=0)

        crud.update("vs_1", docs)

        _, kwargs = mock_client.vector_stores.file_batches.create.call_args
        assert kwargs["vector_store_id"] == "vs_1"
        assert kwargs["file_ids"] == ["file-1", "file-2"]
        # list_files should not have been called on the happy path
        mock_client.vector_stores.file_batches.list_files.assert_not_called()

    def test_polls_with_the_create_batch_id(self, crud, mock_client, docs):
        _wire_batch(mock_client, completed=2, failed=0)

        crud.update("vs_1", docs)

        args, kwargs = mock_client.vector_stores.file_batches.retrieve.call_args
        assert args[0] == REAL_BATCH_ID
        assert kwargs["vector_store_id"] == "vs_1"

    def test_skips_upload_when_no_docs(self, crud, mock_client):
        crud.update("vs_1", [])
        mock_client.vector_stores.file_batches.create.assert_not_called()


class TestOpenAIVectorStoreCrudUpdatePartialFailure:
    """Failed files -> RuntimeError with per-file reasons labelled by fname."""

    def test_includes_failed_fnames_and_messages(self, crud, mock_client, docs):
        _wire_batch(mock_client, completed=1, failed=1)
        mock_client.vector_stores.file_batches.list_files.return_value = [
            _failed_file("file-1", "Unsupported file type"),
            _failed_file("file-2", "File too large"),
        ]

        with pytest.raises(RuntimeError) as exc_info:
            crud.update("vs_1", docs)

        msg = str(exc_info.value)
        assert "file1.pdf: Unsupported file type" in msg
        assert "file2.pdf: File too large" in msg

    def test_looks_up_failures_with_create_batch_id_not_poll_id(
        self, crud, mock_client, docs
    ):
        """Regression: poll()'s return .id is the vs_ id, which list_files rejects."""
        _wire_batch(mock_client, completed=1, failed=1)
        mock_client.vector_stores.file_batches.list_files.return_value = [
            _failed_file("file-1", "Unsupported file type")
        ]

        with pytest.raises(RuntimeError):
            crud.update("vs_1", docs)

        _, kwargs = mock_client.vector_stores.file_batches.list_files.call_args
        assert kwargs["batch_id"] == REAL_BATCH_ID
        assert kwargs["batch_id"] != CORRUPT_POLL_ID

    def test_reports_no_error_detail_when_last_error_missing(
        self, crud, mock_client, docs
    ):
        """A failed file with no `last_error` shouldn't drop out of the
        summary — it gets 'no error detail' so the user sees that something
        was wrong with that file even if OpenAI didn't tell us what."""
        _wire_batch(mock_client, completed=1, failed=1)
        mock_client.vector_stores.file_batches.list_files.return_value = [
            _failed_file("file-1", None)
        ]

        with pytest.raises(RuntimeError, match="file1.pdf: no error detail"):
            crud.update("vs_1", docs)

    def test_falls_back_to_file_id_label_for_unknown_file(
        self, crud, mock_client, docs
    ):
        """A failed file ID not matching any doc is labelled by its file ID."""
        _wire_batch(mock_client, completed=1, failed=1)
        mock_client.vector_stores.file_batches.list_files.return_value = [
            _failed_file("file-unknown", "parse error")
        ]

        with pytest.raises(RuntimeError, match="file-unknown: parse error"):
            crud.update("vs_1", docs)

    def test_surfaces_when_list_files_errors(self, crud, mock_client, docs):
        """If the follow-up list_files lookup itself raises, the OpenAI error is
        surfaced (mapped to InterruptedError) instead of being masked as success."""
        _wire_batch(mock_client, completed=0, failed=2)
        mock_client.vector_stores.file_batches.list_files.side_effect = (
            openai.OpenAIError("list failed")
        )

        with pytest.raises(InterruptedError, match="list failed"):
            crud.update("vs_1", docs)


class TestOpenAIVectorStoreCrudUpdateOpenAIExceptions:
    """`create` raising each specific OpenAI exception type maps to
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
        mock_client.vector_stores.file_batches.create.side_effect = exception_factory()

        with pytest.raises(InterruptedError) as exc_info:
            crud.update("vs_1", docs)
        msg = str(exc_info.value)
        assert msg.startswith(expected_prefix), msg
        assert f"code: {expected_status}" in msg, msg
        assert original_message in msg, msg

    def test_api_timeout_error(self, crud, mock_client, docs):
        """APITimeoutError doesn't expose .message — handler interpolates str(e)."""
        mock_client.vector_stores.file_batches.create.side_effect = (
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
        mock_client.vector_stores.file_batches.create.side_effect = openai.OpenAIError(
            "something else"
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


class TestPollFileBatch:
    """Poll loop sleeps while in_progress and returns once the batch settles. A batch
    that never finishes is bounded by the Celery soft time limit, not an internal one.
    """

    def test_polls_until_not_in_progress(
        self, crud: OpenAIVectorStoreCrud, mock_client: MagicMock
    ) -> None:
        done = MagicMock(status="completed")
        mock_client.vector_stores.file_batches.retrieve.side_effect = [
            MagicMock(status="in_progress"),
            done,
        ]

        with patch("app.crud.rag.open_ai.time.sleep") as mock_sleep:
            result = crud._poll_file_batch("vsfb_x", "vs_1")

        assert result is done
        mock_sleep.assert_called_once_with(BATCH_POLL_INTERVAL_SECONDS)


class TestUpdateTerminalStatus:
    """A batch ending 'cancelled'/'failed' with no per-file failures must raise, not
    slip through the failed-count check as success."""

    def test_cancelled_batch_raises(
        self, crud: OpenAIVectorStoreCrud, mock_client: MagicMock, docs: list[MagicMock]
    ) -> None:
        mock_client.vector_stores.file_batches.create.return_value = MagicMock(
            id=REAL_BATCH_ID
        )
        counts = MagicMock(completed=0, failed=0, cancelled=2, in_progress=0)
        mock_client.vector_stores.file_batches.retrieve.return_value = MagicMock(
            status="cancelled", file_counts=counts
        )

        with pytest.raises(RuntimeError, match="cancelled"):
            crud.update("vs_1", docs)


class TestGetMockOpenAIClientWithVectorStore:
    """Contract test for the repo test fixture: exercises the whole helper and
    pins the wiring the callers that use it depend on."""

    def test_wiring_contract(self) -> None:
        client = get_mock_openai_client_with_vector_store()

        assert client.vector_stores.create.return_value.id == "mock_vector_store_id"

        batch = client.vector_stores.file_batches.create.return_value
        assert batch.id == "vsfb_mock"
        assert batch.file_counts.failed == 0
        assert batch.file_counts.completed == 2
        # _poll_file_batch polls retrieve, not poll — pin the endpoint production uses.
        assert client.vector_stores.file_batches.retrieve.return_value is batch

        assert client.beta.assistants.create.return_value.id == "mock_assistant_id"


class TestProviderFileIdValidation:
    """A doc missing its OpenAI file id is a deterministic local failure: it must
    raise a non-retryable ValueError before any batch request is issued, even with
    retries enabled (ValueError is outside the retried exception set)."""

    @pytest.mark.parametrize("file_id", [None, {"gemini": "file-g"}])
    def test_missing_openai_file_id_raises_without_batch_request(
        self, crud: OpenAIVectorStoreCrud, mock_client: MagicMock, file_id
    ) -> None:
        TestBatchRetry._enable_retries()
        doc = MagicMock()
        doc.file_id = file_id

        with pytest.raises(ValueError):
            crud.update("vs_1", [doc])

        mock_client.vector_stores.file_batches.create.assert_not_called()


class TestBatchRetry:
    """_create_and_index_batch retries the whole create+poll+validate unit on any
    OpenAI/indexing failure, up to BATCH_INDEX_MAX_ATTEMPTS (tenacity)."""

    @staticmethod
    def _enable_retries() -> None:
        retrying = OpenAIVectorStoreCrud._create_and_index_batch.retry
        retrying.stop = stop_after_attempt(BATCH_INDEX_MAX_ATTEMPTS)
        retrying.wait = wait_none()

    def test_retries_then_succeeds(
        self, crud: OpenAIVectorStoreCrud, mock_client: MagicMock
    ) -> None:
        self._enable_retries()
        mock_client.vector_stores.file_batches.create.return_value = MagicMock(
            id=REAL_BATCH_ID
        )
        completed = MagicMock(
            status="completed",
            file_counts=MagicMock(completed=1, failed=0, in_progress=0),
        )
        mock_client.vector_stores.file_batches.retrieve.side_effect = [
            openai.APIConnectionError(request=MagicMock()),
            completed,
        ]

        crud.update("vs_1", [_make_doc("file-1", "f1.pdf")])

        assert mock_client.vector_stores.file_batches.create.call_count == 2

    def test_gives_up_after_max_attempts(
        self, crud: OpenAIVectorStoreCrud, mock_client: MagicMock
    ) -> None:
        self._enable_retries()
        mock_client.vector_stores.file_batches.create.return_value = MagicMock(
            id=REAL_BATCH_ID
        )
        mock_client.vector_stores.file_batches.retrieve.return_value = MagicMock(
            status="cancelled",
            file_counts=MagicMock(completed=0, failed=0, cancelled=1, in_progress=0),
        )

        with pytest.raises(RuntimeError):
            crud.update("vs_1", [_make_doc("file-1", "f1.pdf")])

        assert (
            mock_client.vector_stores.file_batches.create.call_count
            == BATCH_INDEX_MAX_ATTEMPTS
        )


class TestProviderFileId:
    def test_returns_openai_file_id(self) -> None:
        assert _provider_file_id(_make_doc("file-1", "f1.pdf")) == "file-1"

    def test_missing_file_id_raises(self) -> None:
        doc = MagicMock()
        doc.file_id = None
        with pytest.raises(ValueError, match="no OpenAI file id"):
            _provider_file_id(doc)

    def test_absent_openai_provider_raises(self) -> None:
        doc = MagicMock()
        doc.file_id = {"google-aistudio": "files/x"}
        with pytest.raises(ValueError, match="no OpenAI file id"):
            _provider_file_id(doc)


class TestDeleteRetriesValidation:
    def test_retries_below_one_raises(
        self, crud: OpenAIVectorStoreCrud, mock_client: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="Retries must be greater-than 1"):
            crud.delete("vs_1", retries=0)
