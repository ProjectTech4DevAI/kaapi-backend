from typing import Tuple
from uuid import uuid4
from unittest.mock import patch

import pytest
from moto import mock_aws
from sqlmodel import Session

from app.crud import DocTransformationJobCrud, DocumentCrud
from app.services.doctransform.job import execute_job, start_job
from app.models import (
    Document,
    Project,
    TransformationStatus,
    DocTransformJobCreate,
)
from app.tests.services.doctransformer.test_job.utils import (
    DocTransformTestBase,
    MockTestTransformer,
)


class TestExecuteJobIntegration(DocTransformTestBase):
    """Integration tests for execute_job function with minimal mocking."""

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_end_to_end_workflow(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        """Test complete end-to-end workflow from start_job to execute_job."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))

        with patch(
            "app.services.doctransform.job.start_low_priority_job",
            return_value="fake-task-id",
        ), patch("app.services.doctransform.job.Session") as mock_session_class, patch(
            "app.services.doctransform.registry.TRANSFORMERS",
            {"test": MockTestTransformer},
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            returned_job_id = start_job(
                db=db,
                project_id=project.id,
                job_id=job.id,
                transformer_name="test",
                target_format="markdown",
                callback_url=None,
            )
            assert job.id == returned_job_id

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

        db.refresh(job)
        assert job.status == TransformationStatus.COMPLETED
        assert job.transformed_document_id is not None

        document_crud = DocumentCrud(session=db, project_id=project.id)
        transformed_doc = document_crud.read_one(job.transformed_document_id)
        assert transformed_doc.source_document_id == document.id
        assert "<transformed>" in transformed_doc.fname

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_execute_job_concurrent_jobs(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        """Test multiple concurrent job executions don't interfere with each other."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        # Create multiple jobs
        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        jobs = []
        for i in range(3):
            job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))
            jobs.append(job)

        for job in jobs:
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

        for job in jobs:
            db.refresh(job)
            assert job.status == TransformationStatus.COMPLETED
            assert job.transformed_document_id is not None

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_multiple_format_transformations(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        """Test transforming the same document to multiple formats."""
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        formats = ["markdown", "text", "html"]
        jobs = []

        job_crud = DocTransformationJobCrud(session=db, project_id=project.id)
        for target_format in formats:
            job = job_crud.create(DocTransformJobCreate(source_document_id=document.id))
            jobs.append((job, target_format))

        for job, target_format in jobs:
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

        document_crud = DocumentCrud(session=db, project_id=project.id)
        for i, (job, target_format) in enumerate(jobs):
            db.refresh(job)
            assert job.status == TransformationStatus.COMPLETED
            assert job.transformed_document_id is not None

            transformed_doc = document_crud.read_one(job.transformed_document_id)
            assert transformed_doc is not None
            if target_format == "markdown":
                assert transformed_doc.fname.endswith(".md")
            elif target_format == "text":
                assert transformed_doc.fname.endswith(".txt")
            elif target_format == "html":
                assert transformed_doc.fname.endswith(".html")
