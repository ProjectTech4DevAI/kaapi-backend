import logging
from uuid import UUID

from sqlmodel import Session, and_, select

from app.core.exception_handlers import HTTPException
from app.core.util import now
from app.models import Document

logger = logging.getLogger(__name__)


class DocumentCrud:
    def __init__(self, session: Session, project_id: int):
        self.session = session
        self.project_id = project_id

    def read_one(self, doc_id: UUID) -> Document:
        statement = select(Document).where(
            and_(
                Document.id == doc_id,
                Document.project_id == self.project_id,
                Document.is_deleted.is_(False),
            )
        )

        result = self.session.exec(statement).one_or_none()
        if result is None:
            logger.warning(
                f"[DocumentCrud.read_one] Document not found | {{'doc_id': '{doc_id}', 'project_id': {self.project_id}}}"
            )
            raise HTTPException(status_code=404, detail="Document not found")

        return result

    def read_many(
        self,
        skip: int | None = None,
        limit: int | None = None,
    ) -> tuple[list[Document], bool]:
        statement = select(Document).where(
            and_(
                Document.project_id == self.project_id,
                Document.is_deleted.is_(False),
            )
        )
        statement = statement.order_by(Document.inserted_at.desc())

        if skip is not None:
            if skip < 0:
                try:
                    raise ValueError(f"Negative skip: {skip}")
                except ValueError as err:
                    logger.error(
                        f"[DocumentCrud.read_many] Invalid skip value | {{'project_id': {self.project_id}, 'skip': {skip}, 'error': '{str(err)}'}}",
                        exc_info=True,
                    )
                    raise
            statement = statement.offset(skip)

        if limit is not None:
            if limit < 0:
                try:
                    raise ValueError(f"Negative limit: {limit}")
                except ValueError as err:
                    logger.error(
                        f"[DocumentCrud.read_many] Invalid limit value | {{'project_id': {self.project_id}, 'limit': {limit}, 'error': '{str(err)}'}}",
                        exc_info=True,
                    )
                    raise
            statement = statement.limit(limit + 1)

        documents = self.session.exec(statement).all()

        has_more = False
        if limit is not None and len(documents) > limit:
            has_more = True
            documents = documents[:limit]

        return documents, has_more

    def read_each(self, doc_ids: list[UUID]):
        statement = select(Document).where(
            and_(
                Document.project_id == self.project_id,
                Document.id.in_(doc_ids),
                Document.is_deleted.is_(False),
            )
        )
        results = self.session.exec(statement).all()

        (retrieved_count, requested_count) = map(len, (results, doc_ids))
        if retrieved_count != requested_count:
            try:
                raise ValueError(
                    f"Requested atleast {requested_count} document retrieved {retrieved_count}"
                )
            except ValueError:
                logger.error(
                    f"[DocumentCrud.read_each] Mismatch in retrieved documents | {{'project_id': {self.project_id}, 'requested_count': {requested_count}, 'retrieved_count': {retrieved_count}}}",
                    exc_info=True,
                )
                raise

        return results

    def update(self, document: Document):
        if not document.project_id:
            document.project_id = self.project_id
        elif document.project_id != self.project_id:
            error = f"Invalid document ownership: project={self.project_id} attempter={document.project_id}"
            try:
                raise PermissionError(error)
            except PermissionError as err:
                logger.error(
                    f"[DocumentCrud.update] Permission error | {{'doc_id': '{document.id}', 'error': '{str(err)}'}}",
                    exc_info=True,
                )
                raise
        document.updated_at = now()

        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        logger.info(
            f"[DocumentCrud.update] Document updated successfully | {{'doc_id': '{document.id}', 'project_id': {self.project_id}}}"
        )

        return document

    def delete(self, doc_id: UUID):
        document = self.read_one(doc_id)
        document.is_deleted = True
        document.deleted_at = now()
        document.updated_at = now()

        updated_document = self.update(document)
        logger.info(
            f"[DocumentCrud.delete] Document deleted successfully | {{'doc_id': '{doc_id}', 'project_id': {self.project_id}}}"
        )
        return updated_document
