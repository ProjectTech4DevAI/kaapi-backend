from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud.document.doc_transformation_job import DocTransformationJobCrud
from app.models import (
    TransformationStatus,
    DocTransformJobCreate,
    DocTransformJobUpdate,
)
from app.tests.utils.document import DocumentStore
from app.tests.utils.auth import TestAuthContext


class TestGetTransformationJob:
    def test_get_existing_job_success(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        document = DocumentStore(db, user_api_key.project_id).put()
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        created_job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        resp = client.get(
            f"{settings.API_V1_STR}/documents/transformation/{created_job.id}",
            headers={"X-API-KEY": user_api_key.key},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]

        assert data["job_id"] is not None
        assert data["source_document_id"] == str(document.id)
        assert data["status"] == TransformationStatus.PENDING
        assert data["error_message"] is None
        assert data["transformed_document"] is None

    def test_get_nonexistent_job_404(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test getting a non-existent transformation job returns 404."""
        fake_uuid = "00000000-0000-0000-0000-000000000001"

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/{fake_uuid}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 404

    def test_get_job_invalid_uuid_422(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        resp = client.get(
            f"{settings.API_V1_STR}/documents/transformation/not-a-uuid",
            headers={"X-API-KEY": user_api_key.key},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body and "valid UUID" in body["error"]

    def test_get_job_different_project_404(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        superuser_api_key: TestAuthContext,
    ) -> None:
        """Test that jobs from different projects are not accessible."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/{job.id}",
            headers={"X-API-KEY": superuser_api_key.key},
        )

        assert response.status_code == 404

    def test_get_completed_job_with_result(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test getting a completed job with transformation result."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        source_document = store.put()
        transformed_document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=source_document.id))

        # Update job to completed status
        crud.update(
            job.id,
            DocTransformJobUpdate(
                status=TransformationStatus.COMPLETED,
                transformed_document_id=transformed_document.id,
            ),
        )

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/{job.id}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == TransformationStatus.COMPLETED
        assert data["data"]["transformed_document"]["id"] == str(
            transformed_document.id
        )

    def test_get_failed_job_with_error(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test getting a failed job with error message."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))
        error_msg = "Transformation failed due to invalid format"

        # Update job to failed status
        crud.update(
            job.id,
            DocTransformJobUpdate(
                status=TransformationStatus.FAILED, error_message=error_msg
            ),
        )

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/{job.id}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == TransformationStatus.FAILED
        assert data["data"]["error_message"] == error_msg


class TestGetMultipleTransformationJobs:
    def test_get_multiple_jobs_success(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test successfully retrieving multiple transformation jobs."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        documents = store.fill(3)
        jobs = [
            crud.create(DocTransformJobCreate(source_document_id=doc.id))
            for doc in documents
        ]
        job_ids_params = "&".join(f"job_ids={job.id}" for job in jobs)

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?{job_ids_params}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]["jobs"]) == 3
        assert len(data["data"]["jobs_not_found"]) == 0

        returned_ids = {job["job_id"] for job in data["data"]["jobs"]}
        expected_ids = {str(job.id) for job in jobs}
        assert returned_ids == expected_ids

    def test_get_mixed_existing_nonexisting_jobs(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test retrieving a mix of existing and non-existing jobs."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        documents = store.fill(2)
        jobs = [
            crud.create(DocTransformJobCreate(source_document_id=doc.id))
            for doc in documents
        ]
        fake_uuid = "00000000-0000-0000-0000-000000000001"

        job_ids_params = (
            f"job_ids={jobs[0].id}&job_ids={jobs[1].id}&job_ids={fake_uuid}"
        )

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?{job_ids_params}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["jobs"]) == 2
        assert len(data["data"]["jobs_not_found"]) == 1
        assert data["data"]["jobs_not_found"][0] == fake_uuid

    def test_get_jobs_with_empty_string(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        """Test retrieving jobs with empty job_ids parameter."""
        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?job_ids=",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["data"] is None
        assert "error" in body
        assert "valid UUID" in body["error"] or "expected length" in body["error"]
        assert "job_ids" in body["error"]

    def test_get_jobs_with_whitespace_only(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        """Test retrieving jobs with whitespace-only job_ids."""
        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?job_ids=   ",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["data"] is None
        assert "error" in body
        assert "valid UUID" in body["error"]

    def test_get_jobs_invalid_uuid_format_422(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        """Invalid UUID format should return 422 (validation error)."""
        invalid_uuid = "not-a-uuid"

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?job_ids={invalid_uuid}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["data"] is None
        assert "error" in body
        assert "valid UUID" in body["error"] or "expected length" in body["error"]
        assert "job_ids" in body["error"]

    def test_get_jobs_mixed_valid_invalid_uuid_422(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Mixed valid/invalid UUIDs should return 422."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        job_ids_params = f"job_ids={job.id}&job_ids=not-a-uuid"

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?{job_ids_params}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["data"] is None
        assert "error" in body
        assert "job_ids" in body["error"]
        assert (
            "valid UUID" in body["error"]
            or "invalid character" in body["error"]
            or "invalid length" in body["error"]
        )

    def test_get_jobs_missing_parameter_422(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        """Missing job_ids parameter should 422 (Query(min=1))."""
        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["data"] is None
        assert "error" in body
        assert "Field required" in body["error"]
        assert "job_ids" in body["error"]

    def test_get_jobs_different_project_not_found(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        superuser_api_key: TestAuthContext,
    ) -> None:
        """Jobs from different projects are not returned."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        document = store.put()
        job = crud.create(DocTransformJobCreate(source_document_id=document.id))

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?job_ids={job.id}",
            headers={"X-API-KEY": superuser_api_key.key},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["jobs"]) == 0
        assert len(data["data"]["jobs_not_found"]) == 1
        assert data["data"]["jobs_not_found"][0] == str(job.id)

    def test_get_jobs_with_various_statuses(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test retrieving jobs with different statuses."""
        store = DocumentStore(db, user_api_key.project_id)
        crud = DocTransformationJobCrud(db, user_api_key.project_id)
        documents = store.fill(4)
        jobs = [
            crud.create(DocTransformJobCreate(source_document_id=doc.id))
            for doc in documents
        ]

        crud.update(
            jobs[1].id, DocTransformJobUpdate(status=TransformationStatus.PROCESSING)
        )
        crud.update(
            jobs[2].id,
            DocTransformJobUpdate(
                status=TransformationStatus.COMPLETED,
                transformed_document_id=documents[2].id,
            ),
        )
        crud.update(
            jobs[3].id,
            DocTransformJobUpdate(
                status=TransformationStatus.FAILED,
                error_message="Test error",
            ),
        )

        job_ids_params = "&".join(f"job_ids={job.id}" for job in jobs)

        response = client.get(
            f"{settings.API_V1_STR}/documents/transformation/?{job_ids_params}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["jobs"]) == 4

        statuses = {job["status"] for job in data["data"]["jobs"]}
        expected_statuses = {
            TransformationStatus.PENDING,
            TransformationStatus.PROCESSING,
            TransformationStatus.COMPLETED,
            TransformationStatus.FAILED,
        }
        assert statuses == expected_statuses
