from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlmodel import Session
from fastapi import HTTPException

from app.services.collections import helpers
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_vector_store_collection
from app.services.collections.helpers import (
    ensure_unique_name,
    get_service_name,
    to_collection_public,
)
from app.models import Collection, ProviderType
from app.core.util import now


def test_extract_error_message_parses_json_and_strips_prefix() -> None:
    payload = {"error": {"message": "Inner JSON message"}}
    err = Exception(f"Error code: 400 - {json.dumps(payload)}")
    msg = helpers.extract_error_message(err)
    assert msg == "Inner JSON message"


def test_extract_error_message_parses_python_dict_repr() -> None:
    payload = {"error": {"message": "Dict-repr message"}}
    err = Exception(str(payload))
    msg = helpers.extract_error_message(err)
    assert msg == "Dict-repr message"


def test_extract_error_message_falls_back_to_clean_text_and_truncates() -> None:
    long_text = "x" * 1500
    err = Exception(long_text)
    msg = helpers.extract_error_message(err)
    assert len(msg) == 1000
    assert msg == long_text[:1000]


def test_extract_error_message_handles_non_matching_bodies() -> None:
    err = Exception("some random error without structure")
    msg = helpers.extract_error_message(err)
    assert msg == "some random error without structure"


# batch documents


def create_fake_documents(
    count: int, file_size_kb: float | None = 1
) -> list[SimpleNamespace]:
    """Create fake document objects for testing.

    Args:
        count: Number of documents to create
        file_size_kb: Size in KB for each document (default 1 KB)

    Returns:
        List of SimpleNamespace objects mimicking Document objects
    """
    return [
        SimpleNamespace(
            id=uuid4(),
            fname=f"doc_{i}.txt",
            object_store_url=f"s3://bucket/doc_{i}.txt",
            file_size_kb=file_size_kb,
        )
        for i in range(count)
    ]


def test_batch_documents_small_files_single_batch() -> None:
    """Test that small files all fit in one batch (under 30 MB and under 200 docs)."""
    docs = create_fake_documents(6, file_size_kb=1)  # 1 KB per file
    batches = helpers.batch_documents(docs)

    # All 6 small files should fit in one batch
    assert len(batches) == 1
    assert len(batches[0]) == 6
    assert [d.id for d in batches[0]] == [d.id for d in docs]


def test_batch_documents_size_based_batching() -> None:
    """Test that large files trigger size-based batching (30 MB limit)."""
    # Each file is 20 MB (20480 KB), so max 1 file per batch (since 2 * 20 MB > 30 MB)
    docs = create_fake_documents(3, file_size_kb=20 * 1024)
    batches = helpers.batch_documents(docs)

    # Should create 3 batches, one for each 20 MB file
    assert len(batches) == 3
    assert len(batches[0]) == 1
    assert len(batches[1]) == 1
    assert len(batches[2]) == 1


def test_batch_documents_count_based_batching() -> None:
    """Test that document count triggers batching (200 docs limit)."""
    docs = create_fake_documents(250, file_size_kb=0.1)  # Small files
    batches = helpers.batch_documents(docs)

    # Should create 2 batches: 200 + 50
    assert len(batches) == 2
    assert len(batches[0]) == 200
    assert len(batches[1]) == 50


def test_batch_documents_mixed_size_batching() -> None:
    """Test batching with files that fit multiple per batch but hit 30 MB limit."""
    # Each file is 15 MB (15360 KB), so 2 files = 30 MB (at limit), 3 files > 30 MB
    docs = create_fake_documents(5, file_size_kb=15 * 1024)
    batches = helpers.batch_documents(docs)

    # Should create 3 batches: [2 files, 2 files, 1 file]
    assert len(batches) == 3
    assert len(batches[0]) == 2  # 30 MB total
    assert len(batches[1]) == 2  # 30 MB total
    assert len(batches[2]) == 1  # 15 MB total


def test_batch_documents_zero_size_files_batches_by_count() -> None:
    """Zero-size docs contribute nothing to size, so only the 200-doc count limit applies."""
    docs = create_fake_documents(250, file_size_kb=0)
    batches = helpers.batch_documents(docs)

    assert len(batches) == 2
    assert len(batches[0]) == 200
    assert len(batches[1]) == 50


def test_batch_documents_doc_exactly_at_size_limit_stays_in_same_batch() -> None:
    """A doc whose size exactly equals MAX_BATCH_SIZE_KB should not trigger a new batch
    on its own — the split only happens when adding it would *exceed* the limit."""
    from app.services.collections.helpers import MAX_BATCH_SIZE_KB

    docs = create_fake_documents(1, file_size_kb=MAX_BATCH_SIZE_KB)
    batches = helpers.batch_documents(docs)

    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_batch_documents_with_none_file_size_raises() -> None:
    """Test that documents with None file_size raise TypeError — sizes must be backfilled before batching."""
    docs = create_fake_documents(10, file_size_kb=None)

    with pytest.raises(TypeError):
        helpers.batch_documents(docs)


def test_batch_documents_empty_input() -> None:
    """Test that empty input returns empty batches."""
    batches = helpers.batch_documents([])
    assert batches == []


def test_ensure_unique_name_success(db: Session) -> None:
    requested_name = "new_collection_name"

    project = get_project(db)

    result = ensure_unique_name(
        session=db,
        project_id=project.id,
        requested_name=requested_name,
    )

    assert result == requested_name


def test_ensure_unique_name_conflict_with_vector_store_collection(db: Session) -> None:
    existing_name = "vector_collection"
    project = get_project(db)

    collection = get_vector_store_collection(
        db=db,
        project=project,
    )

    collection.name = existing_name
    db.commit()

    with pytest.raises(HTTPException) as exc:
        ensure_unique_name(
            session=db,
            project_id=project.id,
            requested_name=existing_name,
        )

    assert exc.value.status_code == 409
    assert "already exists" in exc.value.detail


# get_service_name


def test_get_service_name_openai() -> None:
    """Test that OpenAI provider returns correct service name."""
    result = get_service_name("openai")
    assert result == "openai vector store"


def test_get_service_name_case_insensitive() -> None:
    """Test that provider name is case-insensitive."""
    assert get_service_name("OpenAI") == "openai vector store"
    assert get_service_name("OPENAI") == "openai vector store"
    assert get_service_name("OpEnAi") == "openai vector store"


def test_get_service_name_unknown_provider() -> None:
    """Test that unknown providers return empty string."""
    assert get_service_name("unknown") == ""
    assert get_service_name("bedrock") == ""  # Commented out in the mapping
    assert get_service_name("gemini") == ""  # Commented out in the mapping
    assert get_service_name("") == ""


# to_collection_public


def test_to_collection_public_vector_store() -> None:
    """Test conversion of vector store collection to public model."""
    collection = Collection(
        id=uuid4(),
        project_id=1,
        provider=ProviderType.openai,
        knowledge_base_id="vs_123",
        knowledge_base_provider="openai vector store",  # Matches get_service_name("openai")
        name="Test Collection",
        description="Test description",
        inserted_at=now(),
        updated_at=now(),
        deleted_at=None,
    )

    result = to_collection_public(collection)

    assert result.id == collection.id
    assert result.knowledge_base_id == "vs_123"
    assert result.knowledge_base_provider == "openai vector store"
    assert result.project_id == 1
    assert result.inserted_at == collection.inserted_at
    assert result.updated_at == collection.updated_at
    assert result.deleted_at is None


def test_to_collection_public_with_deleted_at() -> None:
    """Test that deleted_at field is properly included when set."""
    deleted_time = now()
    collection = Collection(
        id=uuid4(),
        project_id=3,
        provider=ProviderType.openai,
        knowledge_base_id="vs_789",
        knowledge_base_provider="openai vector store",
        name="Deleted Collection",
        description="Deleted",
        inserted_at=now(),
        updated_at=now(),
        deleted_at=deleted_time,
    )

    result = to_collection_public(collection)

    assert result.deleted_at == deleted_time
