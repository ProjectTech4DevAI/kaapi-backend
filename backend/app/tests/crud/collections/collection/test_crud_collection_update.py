from uuid import uuid4
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.crud import CollectionCrud
from app.crud.collection.collection import CollectionNameConflictError
from app.models import CollectionUpdate
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_vector_store_collection


class TestCollectionCrudUpdate:
    def test_update_name_and_description(self, db: Session) -> None:
        project = get_project(db, "Dalgo")
        collection = get_vector_store_collection(db, project)

        crud = CollectionCrud(db, project.id)
        updated = crud.update(
            collection.id,
            CollectionUpdate(name="renamed", description="new desc"),
        )

        assert updated.name == "renamed"
        assert updated.description == "new desc"

    def test_update_only_name_leaves_description_untouched(self, db: Session) -> None:
        project = get_project(db, "Dalgo")
        collection = get_vector_store_collection(db, project)
        crud = CollectionCrud(db, project.id)

        crud.update(collection.id, CollectionUpdate(description="initial"))
        updated = crud.update(collection.id, CollectionUpdate(name="only-name"))

        assert updated.name == "only-name"
        assert updated.description == "initial"

    def test_update_with_same_name_is_noop_no_conflict(self, db: Session) -> None:
        project = get_project(db, "Dalgo")
        collection = get_vector_store_collection(db, project)
        crud = CollectionCrud(db, project.id)

        crud.update(collection.id, CollectionUpdate(name="same"))
        again = crud.update(collection.id, CollectionUpdate(name="same"))

        assert again.name == "same"

    def test_update_rename_to_existing_name_raises_conflict(self, db: Session) -> None:
        project = get_project(db, "Dalgo")
        crud = CollectionCrud(db, project.id)

        existing = get_vector_store_collection(db, project)
        crud.update(existing.id, CollectionUpdate(name="taken"))

        target = get_vector_store_collection(db, project)

        with pytest.raises(CollectionNameConflictError) as excinfo:
            crud.update(target.id, CollectionUpdate(name="taken"))

        assert excinfo.value.name == "taken"

    def test_update_nonexistent_collection_raises_404(self, db: Session) -> None:
        project = get_project(db, "Dalgo")
        crud = CollectionCrud(db, project.id)

        with pytest.raises(HTTPException) as excinfo:
            crud.update(uuid4(), CollectionUpdate(name="anything"))

        assert excinfo.value.status_code == 404

    def test_update_integrity_error_raises_conflict(self, db: Session) -> None:
        """
        Simulate a concurrent insert: pre-check passes, but the DB commit
        races and raises IntegrityError on the unique index. The CRUD should
        catch it, roll back, and raise the domain CollectionNameConflictError
        (the route is responsible for translating it into HTTP 409).
        """
        project = get_project(db, "Dalgo")
        collection = get_vector_store_collection(db, project)
        crud = CollectionCrud(db, project.id)

        with patch.object(
            crud,
            "_update",
            side_effect=IntegrityError("stmt", {}, Exception("duplicate")),
        ):
            with pytest.raises(CollectionNameConflictError) as excinfo:
                crud.update(collection.id, CollectionUpdate(name="race-condition"))

        assert excinfo.value.name == "race-condition"
