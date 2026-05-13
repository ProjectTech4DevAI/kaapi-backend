import logging
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models import Document, Collection, DocumentCollection

logger = logging.getLogger(__name__)


class DocumentCollectionCrud:
    def __init__(self, session: Session):
        self.session = session

    def create(self, collection: Collection, documents: list[Document]) -> None:
        document_collection = []
        for d in documents:
            dc = DocumentCollection(
                document_id=d.id,
                collection_id=collection.id,
            )
            document_collection.append(dc)

        try:
            self.session.bulk_save_objects(document_collection)
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            logger.warning(
                "[DocumentCollectionCrud.create] Duplicate document-collection link | "
                f"{{'collection_id': '{collection.id}'}}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=409,
                detail="Duplicate document-collection link",
            )
        self.session.refresh(collection)

    def read(
        self,
        collection: Collection,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        statement = (
            select(Document)
            .join(
                DocumentCollection,
                DocumentCollection.document_id == Document.id,
            )
            .where(DocumentCollection.collection_id == collection.id)
        )
        if skip is not None:
            if skip < 0:
                raise ValueError(f"Negative skip: {skip}")
            statement = statement.offset(skip)
        if limit is not None:
            if limit < 0:
                raise ValueError(f"Negative limit: {limit}")
            statement = statement.limit(limit)

        return self.session.exec(statement).all()
