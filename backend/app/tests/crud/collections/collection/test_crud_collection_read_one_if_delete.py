"""Deleting an already-deleted collection is a conflict, not bad input."""

import uuid

import pytest
from sqlmodel import Session

from fastapi import HTTPException

from app.core.util import now
from app.crud import CollectionCrud
from app.models import Collection, ProviderType
from app.tests.utils.utils import get_project


def _make_collection(db: Session, project_id: int, deleted: bool) -> Collection:
    collection = Collection(
        project_id=project_id,
        knowledge_base_id="vs_test",
        knowledge_base_provider="openai vector store",
        provider=ProviderType.openai,
        deleted_at=now() if deleted else None,
    )
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return collection


def test_already_deleted_raises_409(db: Session) -> None:
    project = get_project(db)
    collection = _make_collection(db, project.id, deleted=True)

    with pytest.raises(HTTPException) as exc:
        CollectionCrud(db, project.id).read_one_if_delete(collection.id)

    assert exc.value.status_code == 409
    assert "already deleted" in exc.value.detail


def test_active_collection_is_returned(db: Session) -> None:
    project = get_project(db)
    collection = _make_collection(db, project.id, deleted=False)

    assert (
        CollectionCrud(db, project.id).read_one_if_delete(collection.id).id
        == collection.id
    )


def test_missing_collection_still_raises_404(db: Session) -> None:
    project = get_project(db)

    with pytest.raises(HTTPException) as exc:
        CollectionCrud(db, project.id).read_one_if_delete(uuid.uuid4())

    assert exc.value.status_code == 404
