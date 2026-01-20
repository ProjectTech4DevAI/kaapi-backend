from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlmodel import Session

from app.services.doctransform.job import start_job
from app.services.doctransform.registry import TRANSFORMERS
from app.core.exception_handlers import HTTPException
from app.crud import DocTransformationJobCrud
from app.models import (
    Document,
    DocTransformationJob,
    Project,
    TransformationStatus,
    AuthContext,
    DocTransformJobCreate,
)
from app.tests.services.doctransformer.test_job.utils import (
    DocTransformTestBase,
    MockTestTransformer,
)


class TestStartJob(DocTransformTestBase):
    """Test cases for the start_job function."""

    def _create_job(self, db: Session, project_id: int, source_document_id):
        job = DocTransformJobCreate(source_document_id=source_document_id)
        job = DocTransformationJobCrud(db, project_id=project_id).create(job)
        return job

    def test_start_job_success(
        self,
        db: Session,
        current_user: AuthContext,
        test_document: tuple[Document, Project],
    ) -> None:
        """start_job should enqueue execute_job with correct kwargs and return the same job id."""
        document, _project = test_document

        job = self._create_job(db, current_user.project.id, document.id)

        with patch(
            "app.services.doctransform.job.start_low_priority_job"
        ) as mock_schedule:
            mock_schedule.return_value = "fake-task-id"

            returned_job_id = start_job(
                db=db,
                project_id=current_user.project.id,
                job_id=job.id,
                transformer_name="test-transformer",
                target_format="markdown",
                callback_url=None,
            )

        assert returned_job_id == job.id

        job = db.get(DocTransformationJob, job.id)
        assert job is not None
        assert job.source_document_id == document.id
        assert job.status == TransformationStatus.PENDING
        assert job.error_message is None
        assert job.transformed_document_id is None

        mock_schedule.assert_called_once()
        kwargs = mock_schedule.call_args.kwargs
        assert kwargs["function_path"] == "app.services.doctransform.job.execute_job"
        assert kwargs["project_id"] == current_user.project.id
        assert kwargs["job_id"] == str(job.id)
        assert kwargs["source_document_id"] == str(job.source_document_id)
        assert kwargs["transformer_name"] == "test-transformer"
        assert kwargs["target_format"] == "markdown"
        assert kwargs["callback_url"] is None

    def test_start_job_with_nonexistent_document(
        self,
        db: Session,
        current_user: AuthContext,
    ) -> None:
        """
        Previously: start_job validated document and raised 404.
        Now: start_job expects an existing job; the equivalent negative case is a non-existent JOB.
        """
        nonexistent_job_id = uuid4()

        with pytest.raises(HTTPException):
            with patch(
                "app.services.doctransform.job.start_low_priority_job"
            ) as mock_schedule:
                mock_schedule.return_value = "fake-task-id"
                start_job(
                    db=db,
                    project_id=current_user.project.id,
                    job_id=nonexistent_job_id,
                    transformer_name="test-transformer",
                    target_format="markdown",
                    callback_url=None,
                )

    def test_start_job_with_different_formats(
        self,
        db: Session,
        current_user: AuthContext,
        test_document: tuple[Document, Project],
        monkeypatch,
    ) -> None:
        """Ensure start_job passes target_format through to the scheduler."""
        monkeypatch.setitem(TRANSFORMERS, "test", MockTestTransformer)

        document, _ = test_document
        formats = ["markdown", "text", "html"]

        with patch(
            "app.services.doctransform.job.start_low_priority_job"
        ) as mock_schedule:
            mock_schedule.return_value = "fake-task-id"

            for target_format in formats:
                job = self._create_job(db, current_user.project.id, document.id)

                returned_job_id = start_job(
                    db=db,
                    project_id=current_user.project.id,
                    job_id=job.id,
                    transformer_name="test",
                    target_format=target_format,
                    callback_url=None,
                )

                # job row still PENDING
                job = db.get(DocTransformationJob, job.id)
                assert job is not None
                assert job.status == TransformationStatus.PENDING

                # scheduler called with correct kwargs
                kwargs = mock_schedule.call_args.kwargs
                assert kwargs["target_format"] == target_format
                assert (
                    kwargs["function_path"]
                    == "app.services.doctransform.job.execute_job"
                )
                assert kwargs["project_id"] == current_user.project.id
                assert kwargs["job_id"] == str(job.id)
                assert kwargs["source_document_id"] == str(job.source_document_id)
                assert kwargs["transformer_name"] == "test"
                assert returned_job_id == job.id  # new start_job returns the same UUID

    @pytest.mark.parametrize("transformer_name", ["test"])
    def test_start_job_with_different_transformers(
        self,
        db: Session,
        current_user: AuthContext,
        test_document: tuple[Document, Project],
        transformer_name: str,
        monkeypatch,
    ) -> None:
        """Ensure start_job passes transformer_name through to the scheduler."""
        monkeypatch.setitem(TRANSFORMERS, "test", MockTestTransformer)

        document, _ = test_document
        job = self._create_job(db, current_user.project.id, document.id)

        with patch(
            "app.services.doctransform.job.start_low_priority_job"
        ) as mock_schedule:
            mock_schedule.return_value = "fake-task-id"

            returned_job_id = start_job(
                db=db,
                project_id=current_user.project.id,
                job_id=job.id,
                transformer_name=transformer_name,
                target_format="markdown",
                callback_url=None,
            )

        job = db.get(DocTransformationJob, job.id)
        assert job is not None
        assert job.status == TransformationStatus.PENDING

        kwargs = mock_schedule.call_args.kwargs
        assert kwargs["transformer_name"] == transformer_name
        assert kwargs["target_format"] == "markdown"
        assert kwargs["function_path"] == "app.services.doctransform.job.execute_job"
        assert kwargs["project_id"] == current_user.project.id
        assert kwargs["job_id"] == str(job.id)
        assert kwargs["source_document_id"] == str(job.source_document_id)
        assert returned_job_id == job.id
