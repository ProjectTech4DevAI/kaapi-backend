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
from app.services.collections.helpers import ensure_unique_name


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


def test_batch_documents_with_none_file_size() -> None:
    """Test that documents with None file_size are treated as 0 bytes."""
    docs = create_fake_documents(10, file_size_kb=None)
    batches = helpers.batch_documents(docs)

    # All files with None/0 size should fit in one batch (under both limits)
    assert len(batches) == 1
    assert len(batches[0]) == 10


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
