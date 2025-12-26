from io import BytesIO
from uuid import uuid4
from typing import Any, Callable, Tuple
from unittest.mock import patch

import pytest
from moto import mock_aws
from sqlmodel import Session

from app.crud import DocTransformationJobCrud
from app.models import Document, Project, TransformationStatus, DocTransformJobCreate
from app.tests.services.doctransformer.test_job.utils import (
    DocTransformTestBase,
    MockTestTransformer,
)
from app.tests.services.doctransformer.test_job.utils import (
    create_failing_convert_document,
    create_persistent_failing_convert_document,
)


class TestExecuteJobRetryAndErrors(DocTransformTestBase):
    """Test cases for retry mechanisms and error handling in execute_job function."""

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_with_storage_error(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        fast_execute_job: Callable[
            [int, str, str, str, str, str, str | None, Any], Any
        ],
    ) -> None:
        """Test job execution when S3 upload fails."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        # Mock storage.put to raise an error
        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.job.get_cloud_storage"
        ) as mock_storage_class, patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            mock_storage = mock_storage_class.return_value
            mock_storage.stream.return_value = BytesIO(b"test content")
            mock_storage.put.side_effect = Exception("S3 upload failed")

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

        # Verify job was marked as failed
        db.refresh(job)
        assert job.status == TransformationStatus.FAILED
        assert "S3 upload failed" in job.error_message
        assert job.transformed_document_id is None

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_retry_mechanism(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        fast_execute_job: Callable[
            [int, str, str, str, str, str, str | None, Any], Any
        ],
    ) -> None:
        """Test that retry mechanism works for transient failures."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        # Create a side effect that fails once then succeeds (fast retry will only try 2 times)
        failing_convert_document = create_failing_convert_document(fail_count=1)

        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.job.convert_document",
            side_effect=failing_convert_document,
        ), patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

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
        # Verify the function was retried and eventually succeeded
        db.refresh(job)
        assert job.status == TransformationStatus.COMPLETED

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_exhausted_retries(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        fast_execute_job: Callable[
            [int, str, str, str, str, str, str | None, Any], Any
        ],
    ) -> None:
        """Test behavior when all retry attempts are exhausted."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        # Mock convert_document to always fail
        persistent_failing_convert_document = (
            create_persistent_failing_convert_document("Persistent error")
        )

        with patch(
            "app.services.doctransform.job.Session"
        ) as mock_session_class, patch(
            "app.services.doctransform.job.convert_document",
            side_effect=persistent_failing_convert_document,
        ), patch(
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
        # Verify job was marked as failed after retries
        db.refresh(job)
        assert job.status == TransformationStatus.FAILED
        assert "Persistent error" in job.error_message

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_database_error_during_completion(
        self,
        db: Session,
        test_document: Tuple[Document, Project],
        fast_execute_job: Callable[
            [int, str, str, str, str, str, str | None, Any], Any
        ],
    ) -> None:
        """Test handling of database errors when updating job completion."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

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

            # Mock DocumentCrud.update to fail when creating the transformed document
            with patch(
                "app.services.doctransform.job.DocumentCrud"
            ) as mock_doc_crud_class:
                mock_doc_crud_instance = mock_doc_crud_class.return_value
                mock_doc_crud_instance.read_one.return_value = (
                    document  # Return valid document for source
                )
                mock_doc_crud_instance.update.side_effect = Exception(
                    "Database error during document creation"
                )

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

        # Verify job was marked as failed
        db.refresh(job)
        assert job.status == TransformationStatus.FAILED
        assert "Database error during document creation" in job.error_message
