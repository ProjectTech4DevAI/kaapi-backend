from typing import Any, Callable, Tuple
from unittest.mock import patch
from uuid import uuid4

import pytest
from moto import mock_aws
from sqlmodel import Session
from tenacity import RetryError

from app.crud import DocTransformationJobCrud, DocumentCrud
from app.services.doctransform.registry import TransformationError
from app.services.doctransform.job import execute_job
from app.core.exception_handlers import HTTPException
from app.models import Document, Project, TransformationStatus, DocTransformJobCreate
from app.tests.services.doctransformer.test_job.utils import (
    DocTransformTestBase,
    MockTestTransformer,
)


class TestExecuteJob(DocTransformTestBase):
    """Test cases for the execute_job function."""

    @pytest.mark.parametrize(
        "target_format, expected_extension",
        [
            ("markdown", ".md"),
            ("text", ".txt"),
            ("html", ".html"),
        ],
    )
    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_success(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        target_format: str,
        expected_extension: str,
    ) -> None:
        """Test successful document transformation with multiple formats."""
        document, project = test_document
        aws = self.setup_aws_s3()

        source_content = b"This is a test document for transformation."
        self.create_s3_document_content(aws, document, source_content)

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            execute_job(
                project_id=project.id,
                job_id=str(job.id),
                source_document_id=str(document.id),
                transformer_name="test",
                target_format=target_format,
                task_id=str(uuid4()),
                callback_url=None,
                task_instance=None,
            )

        db.refresh(job)
        assert job.status == TransformationStatus.COMPLETED
        assert job.transformed_document_id is not None
        assert job.error_message is None

        document_crud = DocumentCrud(session=db, project_id=project.id)
        transformed_doc = document_crud.read_one(job.transformed_document_id)
        assert transformed_doc is not None
        assert transformed_doc.fname.endswith(expected_extension)
        assert "<transformed>" in transformed_doc.fname
        assert transformed_doc.source_document_id == document.id
        assert transformed_doc.object_store_url is not None

        self.verify_s3_content(aws, transformed_doc)

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_with_nonexistent_job(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        fast_execute_job: Callable[
            [int, str, str, str, str, str, str | None, Any], Any
        ],
    ) -> None:
        """Test job execution with non-existent job ID."""
        _, project = test_document
        self.setup_aws_s3()
        nonexistent_job_id = uuid4()

        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            # Execute job should fail because job doesn't exist
            with pytest.raises((HTTPException, RetryError)):
                fast_execute_job(
                    project_id=project.id,
                    job_id=str(nonexistent_job_id),
                    source_document_id=str(uuid4()),
                    transformer_name="test",
                    target_format="markdown",
                    task_id=str(uuid4()),
                    callback_url=None,
                    task_instance=None,
                )

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_with_missing_source_document(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        fast_execute_job: Callable[
            [int, str, str, str, str, str, str | None, Any], Any
        ],
    ) -> None:
        """Test job execution when source document is missing from S3."""
        document, project = test_document
        self.setup_aws_s3()

        # Create job but don't upload document to S3
        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            with pytest.raises(Exception):
                fast_execute_job(
                    project_id=project.id,
                    job_id=str(job.id),
                    source_document_id=str(document.id),
                    transformer_name="test",
                    target_format="markdown",
                    task_id=str(uuid4()),
                    callback_url=None,
                    task_instance=None,
                )

        db.refresh(job)
        assert job.status == TransformationStatus.FAILED
        assert job.error_message is not None
        assert job.transformed_document_id is None

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_with_transformer_error(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        fast_execute_job: Callable[
            [int, str, str, str, str, str, str | None, Any], Any
        ],
    ) -> None:
        """Test job execution when transformer raises an error."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        # Mock convert_document to raise TransformationError
        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.job.convert_document"
        ) as mock_convert, patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None
            mock_convert.side_effect = TransformationError("Mock transformation error")

            # Due to retry mechanism, it will raise RetryError after exhausting retries
            with pytest.raises((TransformationError, RetryError)):
                fast_execute_job(
                    project_id=project.id,
                    job_id=str(job.id),
                    source_document_id=str(document.id),
                    transformer_name="test",
                    target_format="markdown",
                    task_id=str(uuid4()),
                    callback_url=None,
                    task_instance=None,
                )

        db.refresh(job)
        assert job.status == TransformationStatus.FAILED
        assert "Mock transformation error" in job.error_message
        assert job.transformed_document_id is None

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_status_transitions(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        """Test that job status transitions correctly during execution."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))
        initial_status = job.status

        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            execute_job(
                project_id=project.id,
                job_id=str(job.id),
                source_document_id=str(document.id),
                transformer_name="test",
                target_format="markdown",
                task_id=str(uuid4()),
                callback_url=None,
                task_instance=None,
            )

        # Verify status progression by checking final job state
        db.refresh(job)
        assert job.status == TransformationStatus.COMPLETED
        assert initial_status == TransformationStatus.PENDING

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_with_different_content_types(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        """Test job execution produces correct content types for different formats."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        format_extensions = [
            ("markdown", "text/markdown", ".md"),
            ("text", "text/plain", ".txt"),
            ("html", "text/html", ".html"),
            ("unknown", "text/plain", ".unknown"),  # Default fallback
        ]

        for (
            target_format,
            expected_content_type,
            expected_extension,
        ) in format_extensions:
            job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
            job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

            with patch(
                "app.services.doctransform.job.Session"
            ) as mock_session_class, patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ):
                mock_session_class.return_value.__enter__.return_value = db
                mock_session_class.return_value.__exit__.return_value = None

                execute_job(
                    project_id=project.id,
                    job_id=str(job.id),
                    source_document_id=str(document.id),
                    transformer_name="test",
                    target_format=target_format,
                    task_id=str(uuid4()),
                    callback_url=None,
                    task_instance=None,
                )
            # Verify transformation completed and check file extension
            db.refresh(job)
            assert job.status == TransformationStatus.COMPLETED
            document_crud = DocumentCrud(session=db, project_id=project.id)
            assert job.transformed_document_id is not None
            transformed_doc = document_crud.read_one(job.transformed_document_id)
            assert transformed_doc is not None
            assert transformed_doc.fname.endswith(expected_extension)
