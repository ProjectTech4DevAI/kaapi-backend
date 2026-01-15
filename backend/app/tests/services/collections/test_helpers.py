from __future__ import annotations

import json
from typing import Any
from types import SimpleNamespace
from uuid import uuid4

from app.services.collections import helpers


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


# _backout


def test_backout_calls_delete_and_swallows_openai_error(monkeypatch: Any) -> None:
    class Crud:
        def __init__(self):
            self.calls = 0

        def delete(self, resource_id: str):
            self.calls += 1

    crud = Crud()
    helpers._backout(crud, "rsrc_1")
    assert crud.calls == 1

    class DummyOpenAIError(Exception):
        pass

    monkeypatch.setattr(helpers, "OpenAIError", DummyOpenAIError)

    class FailingCrud:
        def delete(self, resource_id: str):
            raise DummyOpenAIError("nope")

    helpers._backout(FailingCrud(), "rsrc_2")
