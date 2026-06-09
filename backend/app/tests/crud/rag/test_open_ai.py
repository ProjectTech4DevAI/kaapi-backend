"""
Tests for OpenAIVectorStoreCrud.update — focused on the failure-path error
enrichment added in this PR (per-file reasons + category-prefixed messages
for each specific OpenAI exception type).
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


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get.return_value = b"file content"
    return storage


@pytest.fixture
def docs_batch():
    """One batch with two documents. update() loops `for docs in documents`."""
    doc1 = MagicMock(object_store_url="s3://bucket/file1.pdf", fname="file1.pdf")
    doc2 = MagicMock(object_store_url="s3://bucket/file2.pdf", fname="file2.pdf")
    return [[doc1, doc2]]


def _batch_result(*, completed: int, total: int, batch_id: str = "batch_abc"):
    """Mock the return of vector_stores.file_batches.upload_and_poll."""
    counts = MagicMock(completed=completed, total=total)
    return MagicMock(id=batch_id, file_counts=counts)


def _failed_file_page(files, has_more: bool = False):
    """Mock the iterable returned by list_files(filter='failed')."""
    page = MagicMock()
    page.__iter__ = MagicMock(return_value=iter(files))
    page.has_more = has_more
    return page


def _failed_file(file_id: str, error_message: str | None):
    """Build one failed-file row. last_error=None means 'no reason recorded'."""
    f = MagicMock()
    f.id = file_id
    f.last_error = MagicMock(message=error_message) if error_message else None
    return f


class TestOpenAIVectorStoreCrudUpdateSuccess:
    def test_yields_docs_when_all_files_complete(
        self, crud, mock_client, mock_storage, docs_batch
    ):
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=2, total=2)
        )

        yielded = list(crud.update("vs_1", mock_storage, docs_batch))

        # update yields from the inner docs list on success
        assert len(yielded) == 2
        # list_files should not have been called on the happy path
        mock_client.vector_stores.file_batches.list_files.assert_not_called()


class TestOpenAIVectorStoreCrudUpdatePartialFailure:
    """Partial completion -> InterruptedError with enriched per-file reasons."""

    def test_includes_failed_file_ids_and_messages(
        self, crud, mock_client, mock_storage, docs_batch
    ):
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=1, total=3)
        )
        mock_client.vector_stores.file_batches.list_files.return_value = (
            _failed_file_page(
                [
                    _failed_file("file-abc", "Unsupported file type"),
                    _failed_file("file-xyz", "File too large"),
                ]
            )
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))

        msg = str(exc_info.value)
        assert "OpenAI document processing error" in msg
        assert "1/3 files completed" in msg
        assert "Failed files:" in msg
        assert "file-abc (Unsupported file type)" in msg
        assert "file-xyz (File too large)" in msg

    def test_reports_unknown_error_when_last_error_missing(
        self, crud, mock_client, mock_storage, docs_batch
    ):
        """A failed file with no `last_error` shouldn't drop out of the
        summary — it gets 'Unknown error' so the user sees that something
        was wrong with that file even if OpenAI didn't tell us what."""
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=0, total=1)
        )
        mock_client.vector_stores.file_batches.list_files.return_value = (
            _failed_file_page([_failed_file("file-noerr", None)])
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))

        assert "file-noerr (Unknown error)" in str(exc_info.value)

    def test_appends_ellipsis_when_has_more_results(
        self, crud, mock_client, mock_storage, docs_batch
    ):
        """When OpenAI returns has_more=True we cap at the first 10 entries
        and signal truncation with a trailing ', ...' so callers know more
        failures exist beyond what's shown."""
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=0, total=100)
        )
        mock_client.vector_stores.file_batches.list_files.return_value = (
            _failed_file_page(
                [_failed_file(f"file-{i}", "err") for i in range(10)],
                has_more=True,
            )
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))

        assert str(exc_info.value).endswith(", ...")

    def test_truncates_summary_when_over_600_chars(
        self, crud, mock_client, mock_storage, docs_batch
    ):
        """Long error blobs are truncated at 597 chars + '...' so callback
        payloads stay bounded regardless of what OpenAI returns."""
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=0, total=10)
        )
        # Inflate per-file strings to push the joined summary past 600 chars.
        mock_client.vector_stores.file_batches.list_files.return_value = (
            _failed_file_page(
                [
                    _failed_file(f"file-{'x' * 80}-{i}", "long error " * 10)
                    for i in range(10)
                ]
            )
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))

        msg = str(exc_info.value)
        marker = "Failed files: "
        summary = msg[msg.index(marker) + len(marker) :]
        assert len(summary) == 600
        assert summary.endswith("...")

    def test_falls_back_to_count_only_when_list_files_errors(
        self, crud, mock_client, mock_storage, docs_batch
    ):
        """If the follow-up list_files lookup itself raises, the update
        still raises InterruptedError but with the original count-only
        message — no 'Failed files:' suffix, no transient list_files crash
        masking the real upload problem."""
        mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
            _batch_result(completed=0, total=3)
        )
        mock_client.vector_stores.file_batches.list_files.side_effect = (
            openai.OpenAIError("list failed")
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))

        msg = str(exc_info.value)
        assert "0/3 files completed" in msg
        assert "Failed files:" not in msg


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
        mock_storage,
        docs_batch,
        exception_factory,
        expected_prefix,
        expected_status,
        original_message,
    ):
        mock_client.vector_stores.file_batches.upload_and_poll.side_effect = (
            exception_factory()
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))
        msg = str(exc_info.value)
        assert msg.startswith(expected_prefix), msg
        assert f"code: {expected_status}" in msg, msg
        assert original_message in msg, msg

    def test_api_timeout_error(self, crud, mock_client, mock_storage, docs_batch):
        """APITimeoutError doesn't expose .message — handler interpolates str(e)."""
        mock_client.vector_stores.file_batches.upload_and_poll.side_effect = (
            openai.APITimeoutError(request=MagicMock())
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))
        assert str(exc_info.value).startswith("[KAAPI] OpenAI request timed out")

    def test_generic_openai_error_falls_through(
        self, crud, mock_client, mock_storage, docs_batch
    ):
        """Any OpenAIError subclass without a dedicated handler lands in the
        bottom-most `except openai.OpenAIError` block — prefixed with the
        generic "OpenAI error" tag but still carrying the original message.
        """
        mock_client.vector_stores.file_batches.upload_and_poll.side_effect = (
            openai.OpenAIError("something else")
        )

        with pytest.raises(InterruptedError) as exc_info:
            list(crud.update("vs_1", mock_storage, docs_batch))
        msg = str(exc_info.value)
        assert msg.startswith("[OPENAI] SDK error:"), msg
        assert "something else" in msg, msg


class TestOpenAIVectorStoreCrudInit:
    """The base OpenAICrud init rejects a None client; subclasses inherit it."""

    def test_none_client_raises(self):
        with pytest.raises(ValueError):
            OpenAIVectorStoreCrud(client=None)
