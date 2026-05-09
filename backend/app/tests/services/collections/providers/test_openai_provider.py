from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.services.collections.providers.openai import OpenAIProvider
from app.models.collection import Collection
from app.services.collections.helpers import get_service_name
from app.tests.utils.llm_provider import (
    generate_openai_id,
    get_mock_openai_client_with_vector_store,
)


def test_create_openai_vector_store_only() -> None:
    client = get_mock_openai_client_with_vector_store()
    provider = OpenAIProvider(client=client)

    collection_request = SimpleNamespace(
        documents=["doc1", "doc2"],
        model=None,
        instructions=None,
        temperature=None,
    )

    storage = MagicMock()
    documents = [
        SimpleNamespace(file_size_kb=10),
        SimpleNamespace(file_size_kb=20),
    ]
    vector_store_id = generate_openai_id("vs_")

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud = vector_store_crud_cls.return_value
        vector_store_crud.create.return_value = MagicMock(id=vector_store_id)
        vector_store_crud.update.return_value = None

        collection = provider.create(
            collection_request,
            storage,
            documents,
        )

    assert isinstance(collection, Collection)
    assert collection.llm_service_id == vector_store_id
    assert collection.llm_service_name == get_service_name("openai")


def test_create_openai_with_assistant() -> None:
    client = get_mock_openai_client_with_vector_store()
    provider = OpenAIProvider(client=client)

    collection_request = SimpleNamespace(
        documents=["doc1"],
        model="gpt-4o",
        instructions="You are helpful",
        temperature=0.7,
    )

    storage = MagicMock()
    documents = [SimpleNamespace(file_size_kb=10)]
    vector_store_id = generate_openai_id("vs_")
    assistant_id = generate_openai_id("asst_")

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls, patch(
        "app.services.collections.providers.openai.OpenAIAssistantCrud"
    ) as assistant_crud_cls:
        vector_store_crud = vector_store_crud_cls.return_value
        vector_store_crud.create.return_value = MagicMock(id=vector_store_id)
        vector_store_crud.update.return_value = None

        assistant_crud = assistant_crud_cls.return_value
        assistant_crud.create.return_value = MagicMock(id=assistant_id)

        collection = provider.create(
            collection_request,
            storage,
            documents,
        )

    assert collection.llm_service_id == assistant_id
    assert collection.llm_service_name == "gpt-4o"


def test_delete_openai_assistant() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)

    collection = Collection(
        llm_service_id=generate_openai_id("asst_"),
        llm_service_name="gpt-4o",
        provider="openai",
        project_id=1,
    )

    with patch(
        "app.services.collections.providers.openai.OpenAIAssistantCrud"
    ) as assistant_crud_cls:
        assistant_crud = assistant_crud_cls.return_value
        provider.delete(collection)

    assistant_crud.delete.assert_called_once_with(collection.llm_service_id)


def test_delete_openai_vector_store() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)

    collection = Collection(
        llm_service_id=generate_openai_id("vs_"),
        llm_service_name=get_service_name("openai"),
    )

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud = vector_store_crud_cls.return_value
        provider.delete(collection)

    vector_store_crud.delete.assert_called_once_with(collection.llm_service_id)


# ---------------------------------------------------------------------------
# upload_files
# ---------------------------------------------------------------------------


def _make_doc(*, openai_file_id=None, file_size_kb=None):
    return SimpleNamespace(
        id=uuid4(),
        fname="test.md",
        object_store_url="s3://bucket/test.md",
        openai_file_id=openai_file_id,
        file_size_kb=file_size_kb,
    )


def _patch_session_and_crud():
    """Patches Session and DocumentCrud used inside upload_files."""
    session_patcher = patch("app.services.collections.providers.openai.Session")
    crud_patcher = patch("app.services.collections.providers.openai.DocumentCrud")
    return session_patcher, crud_patcher


def test_upload_files_skips_doc_with_existing_openai_file_id() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)
    storage = MagicMock()
    doc = _make_doc(openai_file_id="file-already-exists", file_size_kb=10.0)

    session_p, crud_p = _patch_session_and_crud()
    with session_p, crud_p:
        provider.upload_files(storage, [doc], project_id=1)

    storage.get.assert_not_called()
    client.files.create.assert_not_called()


def test_upload_files_uploads_doc_and_sets_openai_file_id() -> None:
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-new-abc")
    provider = OpenAIProvider(client=client)

    storage = MagicMock()
    storage.get.return_value = b"file content"

    doc = _make_doc(file_size_kb=10.0)

    mock_crud = MagicMock()
    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p as MockDocCrud:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False
        MockDocCrud.return_value = mock_crud

        provider.upload_files(storage, [doc], project_id=1)

    assert doc.openai_file_id == "file-new-abc"
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
    storage = MagicMock()
    storage.get.return_value = content

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

    storage = MagicMock()
    storage.get.return_value = b"x" * 4096

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

    storage = MagicMock()
    storage.get.return_value = b"content"

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
    assert mock_db_doc.openai_file_id == "file-db-check"
    assert mock_db_doc.file_size_kb == 5.0
    mock_crud.update.assert_called_once_with(mock_db_doc)


def test_upload_files_raises_on_storage_failure() -> None:
    client = MagicMock()
    provider = OpenAIProvider(client=client)

    storage = MagicMock()
    storage.get.side_effect = RuntimeError("S3 error")

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

    storage = MagicMock()
    storage.get.return_value = b"content"

    doc = _make_doc()

    session_p, crud_p = _patch_session_and_crud()
    with session_p, crud_p:
        with pytest.raises(RuntimeError, match="OpenAI error"):
            provider.upload_files(storage, [doc], project_id=1)


def test_upload_files_mixed_skips_uploaded_uploads_new() -> None:
    """Docs with openai_file_id are skipped; others are uploaded."""
    client = MagicMock()
    client.files.create.return_value = MagicMock(id="file-new")
    provider = OpenAIProvider(client=client)

    storage = MagicMock()
    storage.get.return_value = b"content"

    already_uploaded = _make_doc(openai_file_id="file-exists", file_size_kb=5.0)
    new_doc = _make_doc(file_size_kb=5.0)

    session_p, crud_p = _patch_session_and_crud()
    with session_p as MockSession, crud_p:
        MockSession.return_value.__enter__.return_value = MagicMock()
        MockSession.return_value.__exit__.return_value = False

        provider.upload_files(storage, [already_uploaded, new_doc], project_id=1)

    assert already_uploaded.openai_file_id == "file-exists"
    assert new_doc.openai_file_id == "file-new"
    client.files.create.assert_called_once()
    storage.get.assert_called_once_with(new_doc.object_store_url)


# ---------------------------------------------------------------------------
# create (existing tests below)
# ---------------------------------------------------------------------------


def test_create_propagates_exception() -> None:
    provider = OpenAIProvider(client=MagicMock())

    collection_request = SimpleNamespace(
        documents=["doc1"],
        model=None,
        instructions=None,
        temperature=None,
    )

    with patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud_cls.return_value.create.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            provider.create(
                collection_request,
                MagicMock(),
                [SimpleNamespace(file_size_kb=10)],
            )
