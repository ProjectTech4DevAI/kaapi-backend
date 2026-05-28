import logging
import functools as ft
from uuid import UUID
from typing import Optional
import logging

from fastapi import HTTPException
from sqlmodel import Session, select, and_
from sqlalchemy.exc import IntegrityError

from app.models import Document, Collection, CollectionUpdate, DocumentCollection
from app.core.util import now
from app.crud.document_collection import DocumentCollectionCrud

logger = logging.getLogger(__name__)


class CollectionNameConflictError(Exception):
    """Raised when a collection name conflicts with an existing active collection."""

    def __init__(self, name: str | None) -> None:
        self.name = name
        super().__init__(f"Collection name '{name}' already exists")


class CollectionCrud:
    def __init__(self, session: Session, project_id: int):
        self.session = session
        self.project_id = project_id

    def _update(self, collection: Collection):
        self.session.add(collection)
        self.session.commit()
        self.session.refresh(collection)
        logger.info(
            f"[CollectionCrud._update] Collection updated successfully | {{'collection_id': '{collection.id}'}}"
        )

        return collection

    def _exists(self, collection: Collection) -> bool:
        stmt = select(Collection.id).where(
            (Collection.project_id == self.project_id)
            & (Collection.llm_service_id == collection.llm_service_id)
            & (Collection.llm_service_name == collection.llm_service_name)
        )
        present = self.session.exec(stmt).scalar_one_or_none() is not None

        return present

    def create(
        self, collection: Collection, documents: list[Document] | None = None
    ) -> Collection:
        self.session.add(collection)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            return self.read_one(collection.id)
        self.session.refresh(collection)

        if documents:
            DocumentCollectionCrud(self.session).create(collection, documents)

        return collection

    def read_one(self, collection_id: UUID) -> Collection:
        statement = select(Collection).where(
            and_(
                Collection.project_id == self.project_id,
                Collection.id == collection_id,
                Collection.deleted_at.is_(None),
            )
        )

        collection = self.session.exec(statement).one_or_none()
        if collection is None:
            logger.warning(
                "[CollectionCrud.read_one] Collection not found | "
                f"{{'project_id': '{self.project_id}', 'collection_id': '{collection_id}'}}"
            )
            raise HTTPException(
                status_code=404,
                detail="Collection not found",
            )

        logger.info(
            "[CollectionCrud.read_one] Retrieved collection | "
            f"{{'project_id': '{self.project_id}', 'collection_id': '{collection_id}'}}"
        )
        return collection

    def read_one_if_delete(self, collection_id: UUID) -> Collection:
        statement = select(Collection).where(
            and_(
                Collection.project_id == self.project_id,
                Collection.id == collection_id,
            )
        )

        collection = self.session.exec(statement).one_or_none()
        if collection is None:
            logger.warning(
                "[CollectionCrud.read_one_if_delete] Collection not found | "
                f"{{'project_id': '{self.project_id}', 'collection_id': '{collection_id}'}}"
            )
            raise HTTPException(status_code=404, detail="Collection not found")

        if collection.deleted_at is not None:
            logger.warning(
                "[CollectionCrud.read_one_if_delete] Collection already deleted | "
                f"{{'project_id': '{self.project_id}', 'collection_id': '{collection_id}'}}"
            )
            raise HTTPException(status_code=400, detail="Collection already deleted")

        return collection

    def read_all(self):
        statement = select(Collection).where(
            and_(
                Collection.project_id == self.project_id,
                Collection.deleted_at.is_(None),
            )
        )

        collections = self.session.exec(statement).all()
        return collections

    def update(self, collection_id: UUID, patch: CollectionUpdate) -> Collection:
        """Update editable fields of a collection (name, description).

        Raises:
            CollectionNameConflictError: when the requested name is already taken
                by another active collection in the same project (caught either by
                the pre-check or by a unique-index IntegrityError on commit).
        """
        collection = self.read_one(collection_id)

        changes = patch.model_dump(exclude_unset=True, exclude_none=True)

        if "name" in changes and changes["name"] != collection.name:
            if self.exists_by_name(changes["name"]):
                raise CollectionNameConflictError(changes["name"])

        for field, value in changes.items():
            setattr(collection, field, value)

        collection.updated_at = now()
        try:
            return self._update(collection)
        except IntegrityError:
            self.session.rollback()
            raise CollectionNameConflictError(changes.get("name"))

    def exists_by_name(self, collection_name: str) -> bool:
        statement = (
            select(Collection.id)
            .where(Collection.project_id == self.project_id)
            .where(Collection.name == collection_name)
            .where(Collection.deleted_at.is_(None))
        )
        result = self.session.exec(statement).first()
        return result is not None

    def delete_by_id(self, collection_id: UUID) -> Collection:
        coll = self.read_one(collection_id)
        coll.deleted_at = now()

        return self._update(coll)

    @ft.singledispatchmethod
    def delete(self, model, remote):  # remote should be an OpenAICrud
        try:
            raise TypeError(type(model))
        except TypeError as err:
            logger.error(
                f"[CollectionCrud.delete] Invalid model type | {{'model_type': '{type(model).__name__}'}}",
                exc_info=True,
            )
            raise

    @delete.register
    def _(self, model: Collection, remote):
        if model.deleted_at is not None:
            logger.info(
                f"[CollectionCrud.delete] Collection already deleted | {{'collection_id': '{model.id}'}}"
            )
            return model
        remote.delete(model.llm_service_id)
        model.deleted_at = now()
        collection = self._update(model)
        logger.info(
            f"[CollectionCrud.delete] Collection deleted successfully | {{'collection_id': '{model.id}'}}"
        )
        return collection

    @delete.register
    def _(self, model: Document, remote):
        statement = (
            select(Collection)
            .join(
                DocumentCollection,
                DocumentCollection.collection_id == Collection.id,
            )
            .where(
                DocumentCollection.document_id == model.id,
                Collection.deleted_at.is_(None),
            )
            .distinct()
        )

        for coll in self.session.exec(statement):
            self.delete(coll, remote)
        self.session.refresh(model)
        logger.info(
            f"[CollectionCrud.delete] Document deletion from collections completed | {{'document_id': '{model.id}'}}"
        )

        return model
