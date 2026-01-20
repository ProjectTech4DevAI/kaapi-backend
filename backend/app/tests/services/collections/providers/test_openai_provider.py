from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        batch_size=1,
        model=None,
        instructions=None,
        temperature=None,
    )

    storage = MagicMock()
    document_crud = MagicMock()

    fake_batches = [["doc1"], ["doc2"]]
    vector_store_id = generate_openai_id("vs_")

    with patch(
        "app.services.collections.providers.openai.batch_documents",
        return_value=fake_batches,
    ), patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls:
        vector_store_crud = vector_store_crud_cls.return_value
        vector_store_crud.create.return_value = MagicMock(id=vector_store_id)
        vector_store_crud.update.return_value = iter([None])

        collection = provider.create(
            collection_request,
            storage,
            document_crud,
        )

    assert isinstance(collection, Collection)
    assert collection.llm_service_id == vector_store_id
    assert collection.llm_service_name == get_service_name("openai")


def test_create_openai_with_assistant() -> None:
    client = get_mock_openai_client_with_vector_store()
    provider = OpenAIProvider(client=client)

    collection_request = SimpleNamespace(
        documents=["doc1"],
        batch_size=1,
        model="gpt-4o",
        instructions="You are helpful",
        temperature=0.7,
    )

    storage = MagicMock()
    document_crud = MagicMock()

    fake_batches = [["doc1"]]
    vector_store_id = generate_openai_id("vs_")
    assistant_id = generate_openai_id("asst_")

    with patch(
        "app.services.collections.providers.openai.batch_documents",
        return_value=fake_batches,
    ), patch(
        "app.services.collections.providers.openai.OpenAIVectorStoreCrud"
    ) as vector_store_crud_cls, patch(
        "app.services.collections.providers.openai.OpenAIAssistantCrud"
    ) as assistant_crud_cls:
        vector_store_crud = vector_store_crud_cls.return_value
        vector_store_crud.create.return_value = MagicMock(id=vector_store_id)
        vector_store_crud.update.return_value = iter([None])

        assistant_crud = assistant_crud_cls.return_value
        assistant_crud.create.return_value = MagicMock(id=assistant_id)

        collection = provider.create(
            collection_request,
            storage,
            document_crud,
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


def test_create_propagates_exception() -> None:
    provider = OpenAIProvider(client=MagicMock())

    collection_request = SimpleNamespace(
        documents=["doc1"],
        batch_size=1,
        model=None,
        instructions=None,
        temperature=None,
    )

    with patch(
        "app.services.collections.providers.openai.batch_documents",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError):
            provider.create(
                collection_request,
                MagicMock(),
                MagicMock(),
            )
