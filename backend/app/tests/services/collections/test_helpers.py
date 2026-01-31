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
    def __init__(self):
        self.calls = []

    def read_each(self, ids):
        self.calls.append(list(ids))
        return [
            SimpleNamespace(
                id=i, fname=f"{i}.txt", object_store_url=f"s3://bucket/{i}.txt"
            )
            for i in ids
        ]


def test_batch_documents_even_chunks() -> None:
    crud = FakeDocumentCrud()
    ids = [uuid4() for _ in range(6)]
    batches = helpers.batch_documents(crud, ids, batch_size=3)

    # read_each called with chunks [0:3], [3:6]
    assert crud.calls == [ids[0:3], ids[3:6]]
    # output mirrors what read_each returned
    assert len(batches) == 2
    assert [d.id for d in batches[0]] == ids[0:3]
    assert [d.id for d in batches[1]] == ids[3:6]


def test_batch_documents_ragged_last_chunk() -> None:
    crud = FakeDocumentCrud()
    ids = [uuid4() for _ in range(5)]
    batches = helpers.batch_documents(crud, ids, batch_size=2)

    assert crud.calls == [ids[0:2], ids[2:4], ids[4:5]]
    assert [d.id for d in batches[0]] == ids[0:2]
    assert [d.id for d in batches[1]] == ids[2:4]
    assert [d.id for d in batches[2]] == ids[4:5]


def test_batch_documents_empty_input() -> None:
    crud = FakeDocumentCrud()
    batches = helpers.batch_documents(crud, [], batch_size=3)
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
