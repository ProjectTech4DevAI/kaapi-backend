"""
Tests for doctransform execute_job: callbacks, payload builders, signed URL, and tmp dir cleanup.

All existing tests pass callback_url=None. This file covers the gaps:
- success / failure callbacks (payload structure, single send, webhook secret)
- build_success_payload / build_failure_payload
- tmp dir cleaned up in both success and failure paths
- signed URL included when storage supports it; exception swallowed when it doesn't
"""
import shutil
from datetime import datetime
from io import BytesIO
from typing import Tuple
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from moto import mock_aws
from sqlmodel import Session

from app.crud import DocTransformationJobCrud
from app.models import (
    Document,
    DocTransformJobCreate,
    Project,
    TransformationStatus,
    TransformedDocumentPublic,
)
from app.services.doctransform.job import (
    build_failure_payload,
    build_success_payload,
    execute_job,
)
from app.tests.services.doctransformer.test_job.utils import (
    DocTransformTestBase,
    MockTestTransformer,
)


def _make_transformed_doc(document: Document) -> TransformedDocumentPublic:
    return TransformedDocumentPublic(
        id=uuid4(),
        project_id=document.project_id,
        fname="output.md",
        object_store_url="s3://bucket/key",
        source_document_id=document.id,
        inserted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Payload builders — pure logic, no S3
# ---------------------------------------------------------------------------


class TestBuildPayloads:
    def test_success_payload_structure(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )
        payload = build_success_payload(job, _make_transformed_doc(document))

        assert payload["success"] is True
        assert payload["error"] is None
        assert "error_message" not in payload["data"]
        assert payload["data"]["transformed_document"]["fname"] == "output.md"

    def test_failure_payload_structure(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )
        payload = build_failure_payload(job, "conversion crashed")

        assert payload["success"] is False
        assert "conversion crashed" in payload["error"]
        assert "error_message" not in payload["data"]
        assert payload["data"]["transformed_document"] is None


# ---------------------------------------------------------------------------
# Callback — success path
# ---------------------------------------------------------------------------


class TestCallbacksSuccess(DocTransformTestBase):
    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_success_sends_callback_once_with_correct_payload(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )
        callback_url = "https://example.com/webhook"

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback") as mock_send,
            patch(
                "app.services.doctransform.job.get_webhook_secret", return_value=None
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            execute_job(
                project_id=project.id,
                job_id=str(job.id),
                source_document_id=str(document.id),
                transformer_name="test",
                target_format="markdown",
                task_id=str(uuid4()),
                callback_url=callback_url,
                task_instance=None,
            )

        mock_send.assert_called_once()
        url_arg, payload_arg = mock_send.call_args.args
        assert url_arg == callback_url
        assert payload_arg["success"] is True
        assert payload_arg["data"]["status"] == TransformationStatus.COMPLETED

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_success_callback_not_sent_without_callback_url(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback") as mock_send,
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

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

        mock_send.assert_not_called()

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_webhook_secret_passed_to_send_callback(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback") as mock_send,
            patch(
                "app.services.doctransform.job.get_webhook_secret",
                return_value="my-secret",
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            execute_job(
                project_id=project.id,
                job_id=str(job.id),
                source_document_id=str(document.id),
                transformer_name="test",
                target_format="markdown",
                task_id=str(uuid4()),
                callback_url="https://example.com/webhook",
                task_instance=None,
            )

        assert mock_send.call_args.kwargs["webhook_secret"] == "my-secret"


# ---------------------------------------------------------------------------
# Callback — failure path
# ---------------------------------------------------------------------------


class TestCallbacksFailure(DocTransformTestBase):
    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_failure_sends_callback_with_error_payload(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback") as mock_send,
            patch(
                "app.services.doctransform.job.get_webhook_secret", return_value=None
            ),
            patch(
                "app.services.doctransform.job.convert_document",
                side_effect=RuntimeError("converter crashed"),
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            with pytest.raises(RuntimeError):
                execute_job.__wrapped__(
                    project_id=project.id,
                    job_id=str(job.id),
                    source_document_id=str(document.id),
                    transformer_name="test",
                    target_format="markdown",
                    task_id=str(uuid4()),
                    callback_url="https://example.com/webhook",
                    task_instance=None,
                )

        mock_send.assert_called_once()
        url_arg, payload_arg = mock_send.call_args.args
        assert payload_arg["success"] is False
        assert "converter crashed" in payload_arg["error"]

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_failure_callback_not_sent_without_callback_url(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback") as mock_send,
            patch(
                "app.services.doctransform.job.convert_document",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            with pytest.raises(RuntimeError):
                execute_job.__wrapped__(
                    project_id=project.id,
                    job_id=str(job.id),
                    source_document_id=str(document.id),
                    transformer_name="test",
                    target_format="markdown",
                    task_id=str(uuid4()),
                    callback_url=None,
                    task_instance=None,
                )

        mock_send.assert_not_called()

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_failure_marks_job_failed_before_callback(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback"),
            patch(
                "app.services.doctransform.job.get_webhook_secret", return_value=None
            ),
            patch(
                "app.services.doctransform.job.convert_document",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            with pytest.raises(RuntimeError):
                execute_job.__wrapped__(
                    project_id=project.id,
                    job_id=str(job.id),
                    source_document_id=str(document.id),
                    transformer_name="test",
                    target_format="markdown",
                    task_id=str(uuid4()),
                    callback_url="https://example.com/webhook",
                    task_instance=None,
                )

        db.refresh(job)
        assert job.status == TransformationStatus.FAILED
        assert "crash" in job.error_message


# ---------------------------------------------------------------------------
# Tmp dir cleanup
# ---------------------------------------------------------------------------


class TestTmpDirCleanup(DocTransformTestBase):
    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_tmp_dir_removed_on_success(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )
        removed: list[str] = []
        real_rmtree = shutil.rmtree

        def capture(path, **kw):
            removed.append(str(path))
            real_rmtree(path, **kw)

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.shutil.rmtree", side_effect=capture),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

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

        assert len(removed) == 1

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_tmp_dir_removed_on_failure(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )
        removed: list[str] = []
        real_rmtree = shutil.rmtree

        def capture(path, **kw):
            removed.append(str(path))
            real_rmtree(path, **kw)

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.shutil.rmtree", side_effect=capture),
            patch(
                "app.services.doctransform.job.convert_document",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            with pytest.raises(RuntimeError):
                execute_job.__wrapped__(
                    project_id=project.id,
                    job_id=str(job.id),
                    source_document_id=str(document.id),
                    transformer_name="test",
                    target_format="markdown",
                    task_id=str(uuid4()),
                    callback_url=None,
                    task_instance=None,
                )

        assert len(removed) == 1


# ---------------------------------------------------------------------------
# Signed URL
# ---------------------------------------------------------------------------


class TestSignedUrl(DocTransformTestBase):
    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_signed_url_included_in_callback_when_available(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )

        mock_storage = MagicMock()
        mock_storage.stream.return_value = BytesIO(b"content")
        mock_storage.put.return_value = "s3://bucket/transformed"
        mock_storage.get_signed_url.return_value = "https://signed.example.com/doc"

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback") as mock_send,
            patch(
                "app.services.doctransform.job.get_webhook_secret", return_value=None
            ),
            patch(
                "app.services.doctransform.job.get_cloud_storage",
                return_value=mock_storage,
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            execute_job(
                project_id=project.id,
                job_id=str(job.id),
                source_document_id=str(document.id),
                transformer_name="test",
                target_format="markdown",
                task_id=str(uuid4()),
                callback_url="https://example.com/webhook",
                task_instance=None,
            )

        payload = mock_send.call_args.args[1]
        assert (
            payload["data"]["transformed_document"]["signed_url"]
            == "https://signed.example.com/doc"
        )

    @mock_aws
    @pytest.mark.usefixtures("aws_credentials")
    def test_signed_url_exception_swallowed_job_still_succeeds(
        self, db: Session, test_document: Tuple[Document, Project]
    ) -> None:
        document, project = test_document
        aws = self.setup_aws_s3()
        self.create_s3_document_content(aws, document)

        job = DocTransformationJobCrud(db, project_id=project.id).create(
            DocTransformJobCreate(source_document_id=document.id)
        )

        mock_storage = MagicMock()
        mock_storage.stream.return_value = BytesIO(b"content")
        mock_storage.put.return_value = "s3://bucket/transformed"
        mock_storage.get_signed_url.side_effect = Exception("token expired")

        with (
            patch("app.services.doctransform.job.Session") as mock_session,
            patch("app.services.doctransform.job.send_callback") as mock_send,
            patch(
                "app.services.doctransform.job.get_webhook_secret", return_value=None
            ),
            patch(
                "app.services.doctransform.job.get_cloud_storage",
                return_value=mock_storage,
            ),
            patch(
                "app.services.doctransform.registry.TRANSFORMERS",
                {"test": MockTestTransformer},
            ),
        ):
            mock_session.return_value.__enter__.return_value = db
            mock_session.return_value.__exit__.return_value = None

            execute_job(
                project_id=project.id,
                job_id=str(job.id),
                source_document_id=str(document.id),
                transformer_name="test",
                target_format="markdown",
                task_id=str(uuid4()),
                callback_url="https://example.com/webhook",
                task_instance=None,
            )

        db.refresh(job)
        assert job.status == TransformationStatus.COMPLETED
        payload = mock_send.call_args.args[1]
        assert payload["data"]["transformed_document"]["signed_url"] is None
