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


class FakeDocumentCrud:
    def __init__(self, file_size_per_doc=1024):
        """
        Args:
            file_size_per_doc: Size in bytes for each fake document (default 1 KB)
        """
        self.calls = []
        self.file_size_per_doc = file_size_per_doc
        self.documents = {}

    def read_one(self, doc_id):
        """Simulate reading a single document by ID."""
        if doc_id not in self.documents:
            self.documents[doc_id] = SimpleNamespace(
                id=doc_id,
                fname=f"{doc_id}.txt",
                object_store_url=f"s3://bucket/{doc_id}.txt",
                file_size=self.file_size_per_doc,
            )
        self.calls.append(doc_id)
        return self.documents[doc_id]


def test_batch_documents_small_files_single_batch() -> None:
    """Test that small files all fit in one batch (under 30 MB and under 200 docs)."""
    crud = FakeDocumentCrud(file_size_per_doc=1024)  # 1 KB per file
    ids = [uuid4() for _ in range(6)]
    batches = helpers.batch_documents(crud, ids)

    # All 6 small files should fit in one batch
    assert len(batches) == 1
    assert len(batches[0]) == 6
    assert [d.id for d in batches[0]] == ids


def test_batch_documents_size_based_batching() -> None:
    """Test that large files trigger size-based batching (30 MB limit)."""
    # Each file is 20 MB, so max 1 file per batch (since 2 * 20 MB > 30 MB)
    crud = FakeDocumentCrud(file_size_per_doc=20 * 1024 * 1024)
    ids = [uuid4() for _ in range(3)]
    batches = helpers.batch_documents(crud, ids)

    # Should create 3 batches, one for each 20 MB file
    assert len(batches) == 3
    assert len(batches[0]) == 1
    assert len(batches[1]) == 1
    assert len(batches[2]) == 1


def test_batch_documents_count_based_batching() -> None:
    """Test that document count triggers batching (200 docs limit)."""
    crud = FakeDocumentCrud(file_size_per_doc=100)  # Small files
    ids = [uuid4() for _ in range(250)]
    batches = helpers.batch_documents(crud, ids)

    # Should create 2 batches: 200 + 50
    assert len(batches) == 2
    assert len(batches[0]) == 200
    assert len(batches[1]) == 50


def test_batch_documents_mixed_size_batching() -> None:
    """Test batching with files that fit multiple per batch but hit 30 MB limit."""
    # Each file is 15 MB, so 2 files = 30 MB (at limit), 3 files > 30 MB
    crud = FakeDocumentCrud(file_size_per_doc=15 * 1024 * 1024)
    ids = [uuid4() for _ in range(5)]
    batches = helpers.batch_documents(crud, ids)

    # Should create 3 batches: [2 files, 2 files, 1 file]
    assert len(batches) == 3
    assert len(batches[0]) == 2  # 30 MB total
    assert len(batches[1]) == 2  # 30 MB total
    assert len(batches[2]) == 1  # 15 MB total


def test_batch_documents_with_none_file_size() -> None:
    """Test that documents with None file_size are treated as 0 bytes."""
    crud = FakeDocumentCrud(file_size_per_doc=None)
    ids = [uuid4() for _ in range(10)]
    batches = helpers.batch_documents(crud, ids)

    # All files with None/0 size should fit in one batch (under both limits)
    assert len(batches) == 1
    assert len(batches[0]) == 10


def test_batch_documents_empty_input() -> None:
    """Test that empty input returns empty batches."""
    crud = FakeDocumentCrud()
    batches = helpers.batch_documents(crud, [])
    assert batches == []
    assert crud.calls == []


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
