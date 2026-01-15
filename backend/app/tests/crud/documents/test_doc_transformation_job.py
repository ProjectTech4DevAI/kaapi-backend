import pytest
from sqlmodel import Session
from sqlalchemy.exc import IntegrityError

from app.crud.document.doc_transformation_job import DocTransformationJobCrud
from app.models import (
    TransformationStatus,
    DocTransformJobCreate,
    DocTransformJobUpdate,
)
from app.core.exception_handlers import HTTPException
from app.tests.utils.document import DocumentStore
from app.tests.utils.utils import get_project, SequentialUuidGenerator
from app.tests.utils.test_data import create_test_project


@pytest.fixture
def store(db: Session) -> DocumentStore:
    project = get_project(db)
    return DocumentStore(db, project.id)


@pytest.fixture
def crud(db: Session, store: DocumentStore) -> DocTransformationJobCrud:
    return DocTransformationJobCrud(db, store.project.id)


class TestDocTransformationJobCrudCreate:
    def test_can_create_job_with_valid_document(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        document = store.put()

        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        assert job.id is not None
        assert job.source_document_id == document.id
        assert job.status == TransformationStatus.PENDING
        assert job.error_message is None
        assert job.transformed_document_id is None
        assert job.inserted_at is not None
        assert job.updated_at is not None

    def test_cannot_create_job_with_invalid_document(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        """With FK enforced, creating with a non-existent document should fail at commit."""
        invalid_id = next(SequentialUuidGenerator())

        with pytest.raises(IntegrityError):
            crud.create(DocTransformJobCreate(source_document_id=invalid_id))

    def test_cannot_create_job_with_deleted_document(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        """
        Creation itself will succeed (FK exists), but later reads should treat it as not found
        because read filters out deleted documents.
        """
        document = store.put()
        document.is_deleted = True
        db.add(document)
        db.commit()

        job = crud.create(DocTransformJobCreate(source_document_id=document.id))
        # read_one should 404 due to is_deleted=True on joined document
        with pytest.raises(HTTPException) as exc_info:
            crud.read_one(job.id)
        assert exc_info.value.status_code == 404
        assert "Transformation job not found" in str(exc_info.value.detail)


class TestDocTransformationJobCrudReadOne:
    def test_can_read_existing_job(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        result = crud.read_one(job.id)

        assert result.id == job.id
        assert result.source_document_id == document.id
        assert result.status == TransformationStatus.PENDING

    def test_cannot_read_nonexistent_job(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        invalid_id = next(SequentialUuidGenerator())

        with pytest.raises(HTTPException) as exc_info:
            crud.read_one(invalid_id)

        assert exc_info.value.status_code == 404
        assert "Transformation job not found" in str(exc_info.value.detail)

    def test_cannot_read_job_with_deleted_document(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        document.is_deleted = True
        db.add(document)
        db.commit()

        with pytest.raises(HTTPException) as exc_info:
            crud.read_one(job.id)

        assert exc_info.value.status_code == 404
        assert "Transformation job not found" in str(exc_info.value.detail)

    def test_cannot_read_job_from_different_project(
        self, db: Session, store: DocumentStore
    ) -> None:
        document = store.put()
        job_crud = DocTransformationJobCrud(db, store.project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        other_project = create_test_project(db)
        other_crud = DocTransformationJobCrud(db, other_project.id)

        with pytest.raises(HTTPException) as exc_info:
            other_crud.read_one(job.id)

        assert exc_info.value.status_code == 404
        assert "Transformation job not found" in str(exc_info.value.detail)


class TestDocTransformationJobCrudReadEach:
    def test_can_read_multiple_existing_jobs(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        documents = store.fill(3)
        jobs = [
            crud.create(DocTransformJobCreate(source_document_id=doc.id))
            for doc in documents
        ]
        job_ids = {job.id for job in jobs}

        results = crud.read_each(job_ids)

        assert len(results) == 3
        result_ids = {job.id for job in results}
        assert result_ids == job_ids

    def test_read_partial_existing_jobs(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        documents = store.fill(2)
        jobs = [
            crud.create(DocTransformJobCreate(source_document_id=doc.id))
            for doc in documents
        ]
        job_ids = {job.id for job in jobs}
        job_ids.add(next(SequentialUuidGenerator()))  # non-existent

        results = crud.read_each(job_ids)

        assert len(results) == 2
        result_ids = {job.id for job in results}
        assert result_ids == {job.id for job in jobs}

    def test_read_empty_job_set(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        results = crud.read_each(set())
        assert len(results) == 0

    def test_cannot_read_jobs_from_different_project(
        self, db: Session, store: DocumentStore
    ) -> None:
        document = store.put()
        job_crud = DocTransformationJobCrud(db, store.project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        other_project = get_project(db, name="Dalgo")
        other_crud = DocTransformationJobCrud(db, other_project.id)

        results = other_crud.read_each({job.id})
        assert len(results) == 0


class TestDocTransformationJobCrudUpdateStatus:
    def test_can_update_status_to_processing(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        updated = crud.update(
            job.id,
            DocTransformJobUpdate(status=TransformationStatus.PROCESSING),
        )

        assert updated.id == job.id
        assert updated.status == TransformationStatus.PROCESSING
        assert updated.updated_at >= job.updated_at

    def test_can_update_status_to_completed_with_result(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        source_document = store.put()
        transformed_document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=source_document.id))

        updated = crud.update(
            job.id,
            DocTransformJobUpdate(
                status=TransformationStatus.COMPLETED,
                transformed_document_id=transformed_document.id,
            ),
        )

        assert updated.status == TransformationStatus.COMPLETED
        assert updated.transformed_document_id == transformed_document.id
        assert updated.error_message is None

    def test_can_update_status_to_failed_with_error(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))
        error_msg = "Transformation failed due to invalid format"

        updated = crud.update(
            job.id,
            DocTransformJobUpdate(
                status=TransformationStatus.FAILED,
                error_message=error_msg,
            ),
        )

        assert updated.status == TransformationStatus.FAILED
        assert updated.error_message == error_msg
        assert updated.transformed_document_id is None

    def test_cannot_update_nonexistent_job(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        invalid_id = next(SequentialUuidGenerator())

        with pytest.raises(HTTPException) as exc_info:
            crud.update(
                invalid_id,
                DocTransformJobUpdate(status=TransformationStatus.PROCESSING),
            )

        assert exc_info.value.status_code == 404
        assert "Transformation job not found" in str(exc_info.value.detail)

    def test_update_preserves_existing_fields(
        self, db: Session, store: DocumentStore, crud: DocTransformationJobCrud
    ) -> None:
        """Fields not present in the patch must be preserved by `update`."""
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        crud.update(
            job.id,
            DocTransformJobUpdate(
                status=TransformationStatus.FAILED, error_message="Initial error"
            ),
        )

        updated = crud.update(
            job.id, DocTransformJobUpdate(status=TransformationStatus.PROCESSING)
        )

        assert updated.status == TransformationStatus.PROCESSING
        assert updated.error_message == "Initial error"
