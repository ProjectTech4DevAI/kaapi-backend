from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from openai import OpenAIError
from tenacity import stop_after_attempt, wait_none

from app.crud.rag.open_ai import OpenAIVectorStoreCrud
from app.services.collections.providers.openai import OpenAIProvider
from app.models.collection import Collection
from app.services.collections.helpers import get_service_name
from app.tests.utils.llm_provider import (
    generate_openai_id,
    get_mock_openai_client_with_vector_store,
)


@pytest.fixture(autouse=True)
def _single_attempt_batch_retry():
    """update() retries batch create+index via tenacity; run one attempt, no backoff."""
    retrying = OpenAIVectorStoreCrud._create_and_index_batch.retry
    stop, wait = retrying.stop, retrying.wait
    retrying.stop, retrying.wait = stop_after_attempt(1), wait_none()
    yield
    retrying.stop, retrying.wait = stop, wait


def test_create_vector_store_returns_id() -> None:
    client = get_mock_openai_client_with_vector_store()
    provider = OpenAIProvider(client=client)
    vector_store_id = generate_openai_id("vs_")

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud_cls.return_value.create.return_value = MagicMock(
            id=vector_store_id
        )

        assert provider.create_vector_store() == vector_store_id


def test_create_vector_store_propagates_exception() -> None:
    provider = OpenAIProvider(client=MagicMock())

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud_cls.return_value.create.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            provider.create_vector_store()


def test_create_openai_vector_store() -> None:
    client = get_mock_openai_client_with_vector_store()
    provider = OpenAIProvider(client=client)

    documents = [
        SimpleNamespace(file_size_kb=10),
        SimpleNamespace(file_size_kb=20),
    ]
    vector_store_id = generate_openai_id("vs_")

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud = vector_store_crud_cls.return_value
        vector_store_crud.update.return_value = None

        collection = provider.create(documents, vector_store_id=vector_store_id)

    vector_store_crud.update.assert_called_once_with(vector_store_id, documents)
    assert isinstance(collection, Collection)
    assert collection.knowledge_base_id == vector_store_id
    assert collection.knowledge_base_provider == get_service_name("openai")


def test_delete_openai_vector_store() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)

    collection = Collection(
        knowledge_base_id=generate_openai_id("vs_"),
        knowledge_base_provider=get_service_name("openai"),
    )

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud = vector_store_crud_cls.return_value
        provider.delete(collection)

    vector_store_crud.delete.assert_called_once_with(collection.knowledge_base_id)


# ---------------------------------------------------------------------------
# upload_files
# ---------------------------------------------------------------------------


def _make_doc(*, file_id=None, file_size_kb=None):
    return SimpleNamespace(
        id=uuid4(),
        fname="test.md",
        object_store_url="s3://bucket/test.md",
        file_id=file_id,
        file_size_kb=file_size_kb,
    )


def _make_storage(content: bytes = b"content") -> MagicMock:
    storage = MagicMock()
    storage.stream.return_value = BytesIO(content)
    return storage


def _patch_session_and_crud():
    """Patches Session and DocumentCrud used inside upload_files."""
    session_patcher = patch("app.services.collections.providers.openai.Session")
    crud_patcher = patch("app.services.collections.providers.openai.DocumentCrud")
    return session_patcher, crud_patcher


def test_upload_files_skips_doc_with_existing_openai_file_id() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)
    storage = _make_storage()
    doc = _make_doc(file_id={"openai": "file-already-exists"}, file_size_kb=10.0)

    session_p, crud_p = _patch_session_and_crud()
    with session_p, crud_p:
        provider.upload_files(storage, [doc], project_id=1)

    storage.stream.assert_not_called()
    client.files.create.assert_not_called()


def test_upload_files_uploads_doc_and_sets_file_id() -> None:
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-new-abc")
    provider = OpenAIProvider(client=client)

    storage = _make_storage(b"file content")
    doc = _make_doc(file_size_kb=10.0)

    mock_crud = MagicMock()
    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p as MockDocCrud:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False
        MockDocCrud.return_value = mock_crud

        provider.upload_files(storage, [doc], project_id=1)

    assert doc.file_id == {"openai": "file-new-abc"}
    client.files.create.assert_called_once()
    _, kwargs = client.files.create.call_args
    assert kwargs.get("purpose") == "assistants"
    mock_crud.update.assert_called_once()


def test_upload_files_sets_file_size_kb_when_none() -> None:
    """file_size_kb should be computed from content length if not already set."""
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-xyz")
    provider = OpenAIProvider(client=client)

    content = b"x" * 2048  # 2 KB
    storage = _make_storage(content)
    doc = _make_doc(file_size_kb=None)

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False

        provider.upload_files(storage, [doc], project_id=1)

    assert doc.file_size_kb == round(len(content) / 1024, 2)


def test_upload_files_preserves_existing_file_size_kb() -> None:
    """file_size_kb should not be overwritten if already set."""
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-xyz")
    provider = OpenAIProvider(client=client)

    storage = _make_storage(b"x" * 4096)
    doc = _make_doc(file_size_kb=99.0)

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False

        provider.upload_files(storage, [doc], project_id=1)

    assert doc.file_size_kb == 99.0


def test_upload_files_updates_db_with_file_id_and_size() -> None:
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-db-check")
    provider = OpenAIProvider(client=client)

    storage = _make_storage()
    doc = _make_doc(file_size_kb=5.0)
    mock_db_doc = MagicMock()
    mock_crud = MagicMock()
    mock_crud.read_one.return_value = mock_db_doc

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p as MockDocCrud:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False
        MockDocCrud.return_value = mock_crud

        provider.upload_files(storage, [doc], project_id=42)

    MockDocCrud.assert_called_once_with(
        MockSession.return_value.__enter__.return_value, 42
    )
    mock_crud.read_one.assert_called_once_with(doc.id)
    assert mock_db_doc.file_id == {"openai": "file-db-check"}
    assert mock_db_doc.file_size_kb == 5.0
    mock_crud.update.assert_called_once_with(mock_db_doc)


def test_upload_files_raises_on_storage_failure() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)

    storage = MagicMock()
    storage.stream.side_effect = RuntimeError("S3 error")

    doc = _make_doc()

    session_p, crud_p = _patch_session_and_crud()
    with session_p, crud_p:
        with pytest.raises(RuntimeError, match="S3 error"):
            provider.upload_files(storage, [doc], project_id=1)

    client.files.create.assert_not_called()


def test_upload_files_raises_on_openai_failure() -> None:
    client = MagicMock()
    client.files.create.side_effect = RuntimeError("OpenAI error")
    provider = OpenAIProvider(client=client)

    storage = _make_storage()
    doc = _make_doc()

    session_p, crud_p = _patch_session_and_crud()
    with session_p, crud_p:
        with pytest.raises(RuntimeError, match="OpenAI error"):
            provider.upload_files(storage, [doc], project_id=1)


def test_upload_files_mixed_skips_uploaded_uploads_new() -> None:
    """Docs with an existing openai file ID are skipped; others are uploaded."""
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-new")
    provider = OpenAIProvider(client=client)

    storage = _make_storage()
    already_uploaded = _make_doc(file_id={"openai": "file-exists"}, file_size_kb=5.0)
    new_doc = _make_doc(file_size_kb=5.0)

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False

        provider.upload_files(storage, [already_uploaded, new_doc], project_id=1)

    assert already_uploaded.file_id == {"openai": "file-exists"}
    assert new_doc.file_id == {"openai": "file-new"}
    client.files.create.assert_called_once()
    storage.stream.assert_called_once_with(new_doc.object_store_url)


def test_upload_files_empty_docs_is_noop() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)
    storage = MagicMock()

    session_p, crud_p = _patch_session_and_crud()
    with session_p, crud_p:
        provider.upload_files(storage, [], project_id=1)

    storage.stream.assert_not_called()
    client.files.create.assert_not_called()


def test_upload_files_file_name_matches_doc_fname() -> None:
    """The file tuple passed to OpenAI must carry the original filename."""
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-abc")
    provider = OpenAIProvider(client=client)

    storage = _make_storage(b"data")
    doc = _make_doc(file_size_kb=1.0)
    doc.fname = "report.pdf"

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False
        provider.upload_files(storage, [doc], project_id=1)

    _, kwargs = client.files.create.call_args
    fname, _ = kwargs["file"]
    assert fname == "report.pdf"


def test_upload_files_raises_on_db_update_failure() -> None:
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-ok")
    provider = OpenAIProvider(client=client)

    storage = _make_storage()
    doc = _make_doc(file_size_kb=1.0)
    mock_crud = MagicMock()
    mock_crud.read_one.return_value = MagicMock()
    mock_crud.update.side_effect = RuntimeError("DB write failed")

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p as MockDocCrud:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False
        MockDocCrud.return_value = mock_crud

        with pytest.raises(RuntimeError, match="DB write failed"):
            provider.upload_files(storage, [doc], project_id=1)

    client.files.delete.assert_called_once_with("file-ok")
    assert "openai" not in doc.file_id


def test_upload_files_db_failure_rollback_delete_error_still_raises_original() -> None:
    """If both DB persistence and the rollback delete fail, the original DB error propagates."""
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-ok")
    client.files.delete.side_effect = RuntimeError("delete failed")
    provider = OpenAIProvider(client=client)

    storage = _make_storage()
    doc = _make_doc(file_size_kb=1.0)
    mock_crud = MagicMock()
    mock_crud.read_one.return_value = MagicMock()
    mock_crud.update.side_effect = RuntimeError("DB write failed")

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p as MockDocCrud:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False
        MockDocCrud.return_value = mock_crud

        with pytest.raises(RuntimeError, match="DB write failed"):
            provider.upload_files(storage, [doc], project_id=1)

    client.files.delete.assert_called_once_with("file-ok")
    assert "openai" not in doc.file_id


def test_upload_files_first_failure_stops_remaining_docs() -> None:
    """If the first doc raises, subsequent docs are never attempted."""
    client = MagicMock()
    client.files.create.side_effect = RuntimeError("quota exceeded")
    provider = OpenAIProvider(client=client)

    storage = _make_storage()
    doc1 = _make_doc(file_size_kb=1.0)
    doc2 = _make_doc(file_size_kb=1.0)

    session_p, crud_p = _patch_session_and_crud()
    with session_p, crud_p:
        with pytest.raises(RuntimeError, match="quota exceeded"):
            provider.upload_files(storage, [doc1, doc2], project_id=1)

    client.files.create.assert_called_once()
    assert storage.stream.call_count == 1


# ---------------------------------------------------------------------------
# OpenAIVectorStoreCrud.update
# ---------------------------------------------------------------------------


def _wire_batch(client: MagicMock, completed: int, failed: int) -> None:
    """create() gives the real vsfb_ id; retrieve() gives the corrupt vs_ id."""
    client.vector_stores.file_batches.create.return_value = MagicMock(id="vsfb_real")
    batch = MagicMock(id="vs_corrupt")
    batch.status = "completed"
    batch.file_counts.completed = completed
    batch.file_counts.failed = failed
    batch.file_counts.in_progress = 0
    client.vector_stores.file_batches.retrieve.return_value = batch


def _make_openai_doc(file_id: str = "file-abc", fname: str = "doc.pdf") -> MagicMock:
    doc = MagicMock()
    doc.file_id = {"openai": file_id}
    doc.fname = fname
    return doc


def test_vector_store_update_skips_when_no_docs() -> None:
    client = MagicMock()
    crud = OpenAIVectorStoreCrud(client)
    crud.update("vs_123", [])
    client.vector_stores.file_batches.create.assert_not_called()


def test_vector_store_update_succeeds_with_no_failures() -> None:
    client = MagicMock()
    _wire_batch(client, completed=3, failed=0)
    crud = OpenAIVectorStoreCrud(client)
    crud.update("vs_123", [_make_openai_doc() for _ in range(3)])
    client.vector_stores.file_batches.create.assert_called_once()


def test_vector_store_update_raises_on_openai_error() -> None:
    client = MagicMock()
    client.vector_stores.file_batches.create.side_effect = OpenAIError("rate limit")
    crud = OpenAIVectorStoreCrud(client)

    with pytest.raises(InterruptedError, match="rate limit"):
        crud.update("vs_123", [_make_openai_doc()])


def _make_failed_file(message: str) -> MagicMock:
    f = MagicMock()
    f.last_error.message = message
    return f


def test_vector_store_update_raises_on_partial_failures() -> None:
    client = MagicMock()
    _wire_batch(client, completed=2, failed=1)
    client.vector_stores.file_batches.list_files.return_value = [
        _make_failed_file("unsupported file type")
    ]
    crud = OpenAIVectorStoreCrud(client)

    with pytest.raises(RuntimeError, match="unsupported file type"):
        crud.update("vs_123", [_make_openai_doc() for _ in range(3)])


def test_vector_store_update_raises_on_all_failures() -> None:
    client = MagicMock()
    _wire_batch(client, completed=0, failed=2)
    client.vector_stores.file_batches.list_files.return_value = [
        _make_failed_file("invalid pdf"),
        _make_failed_file("parse error"),
    ]
    crud = OpenAIVectorStoreCrud(client)

    with pytest.raises(RuntimeError, match="invalid pdf"):
        crud.update("vs_123", [_make_openai_doc() for _ in range(2)])


def test_vector_store_update_passes_file_ids_to_openai() -> None:
    client = MagicMock()
    _wire_batch(client, completed=2, failed=0)
    crud = OpenAIVectorStoreCrud(client)
    docs = [_make_openai_doc("file-1"), _make_openai_doc("file-2")]

    crud.update("vs_abc", docs)

    _, kwargs = client.vector_stores.file_batches.create.call_args
    assert kwargs["vector_store_id"] == "vs_abc"
    assert kwargs["file_ids"] == ["file-1", "file-2"]


# ---------------------------------------------------------------------------
# create (existing tests below)
# ---------------------------------------------------------------------------


def test_create_propagates_exception() -> None:
    provider = OpenAIProvider(client=MagicMock())

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud_cls.return_value.update.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            provider.create([SimpleNamespace(file_size_kb=10)], vector_store_id="vs_x")
