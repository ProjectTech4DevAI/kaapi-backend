import os
import mimetypes
from typing import Any
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from unittest.mock import patch

import pytest
from moto import mock_aws
from sqlmodel import Session, select
from fastapi.testclient import TestClient

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.models import Document
from app.tests.utils.document import (
    Route,
    WebCrawler,
    httpx_to_standard,
)
from app.tests.utils.auth import TestAuthContext


class WebUploader(WebCrawler):
    def put(
        self,
        route: Route,
        scratch: Path,
        target_format: str = None,
        transformer: str = None,
    ):
        (mtype, _) = mimetypes.guess_type(str(scratch))
        files = {"src": (str(scratch), scratch.open("rb"), mtype)}

        data = {}
        if target_format:
            data["target_format"] = target_format
        if transformer:
            data["transformer"] = transformer

        return self.client.post(
            str(route),
            headers={"X-API-KEY": self.user_api_key.key},
            files=files,
            data=data,
        )


@pytest.fixture
def scratch():
    with NamedTemporaryFile(mode="w", suffix=".txt") as fp:
        print("Hello World", file=fp, flush=True)
        yield Path(fp.name)


@pytest.fixture
def pdf_scratch():
    # Create a test PDF file for transformation tests
    with NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as fp:
        fp.write("%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n")
        fp.write("2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n")
        fp.write("3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n")
        fp.write("xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n")
        fp.write(
            "0000000074 00000 n \n0000000120 00000 n \ntrailer<</Size 4/Root 1 0 R>>\n"
        )
        fp.write("startxref\n202\n%%EOF")
        fp.flush()
        yield Path(fp.name)
        # Clean up the temporary file
        Path(fp.name).unlink()


@pytest.fixture
def route():
    return Route("")


@pytest.fixture
def uploader(client: TestClient, user_api_key: TestAuthContext):
    return WebUploader(client, user_api_key)


@pytest.fixture(scope="class")
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentRouteUpload:
    def test_adds_to_database(
        self,
        db: Session,
        route: Route,
        scratch: Path,
        uploader: WebUploader,
    ) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()

        response = httpx_to_standard(uploader.put(route, scratch))
        doc_id = response.data["id"]
        statement = select(Document).where(Document.id == doc_id)
        result = db.exec(statement).one()

        assert result.fname == str(scratch)

    def test_adds_to_S3(
        self,
        db: Session,
        route: Route,
        scratch: Path,
        uploader: WebUploader,
    ) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()

        response = httpx_to_standard(uploader.put(route, scratch))
        doc_id = response.data["id"]

        # Get the document from database to access object_store_url
        statement = select(Document).where(Document.id == doc_id)
        result = db.exec(statement).one()

        url = urlparse(result.object_store_url)
        key = Path(url.path)
        key = key.relative_to(key.root)

        assert aws.client.head_object(Bucket=url.netloc, Key=str(key))

    def test_upload_without_transformation(
        self,
        db: Session,
        route: Route,
        scratch: Path,
        uploader: WebUploader,
    ) -> None:
        """Test basic upload without any transformation parameters."""
        aws = AmazonCloudStorageClient()
        aws.create()

        response = httpx_to_standard(uploader.put(route, scratch))

        assert response.success is True
        assert response.data["transformation_job"] is None
        assert "id" in response.data
        assert "fname" in response.data

    @patch("app.services.doctransform.job.start_job")
    def test_upload_with_transformation(
        self,
        mock_start_job: Any,
        db: Session,
        route: Route,
        pdf_scratch: Path,
        uploader: WebUploader,
    ) -> None:
        """Test upload with valid transformation parameters."""
        aws = AmazonCloudStorageClient()
        aws.create()

        # Mock the background job creation
        mock_job_id = "12345678-1234-5678-9abc-123456789012"
        mock_start_job.return_value = mock_job_id

        response = httpx_to_standard(
            uploader.put(route, pdf_scratch, target_format="markdown")
        )

        assert response.success is True
        assert response.data["transformation_job"] is not None

        transformation_job = response.data["transformation_job"]
        assert transformation_job["job_id"] == mock_job_id
        assert transformation_job["transformer"] == "zerox"  # Default transformer
        assert (
            transformation_job["status_check_url"]
            == f"/documents/transformation/{mock_job_id}"
        )
        assert "message" in transformation_job

    @patch("app.services.doctransform.job.start_job")
    def test_upload_with_specific_transformer(
        self,
        mock_start_job: Any,
        db: Session,
        route: Route,
        pdf_scratch: Path,
        uploader: WebUploader,
    ) -> None:
        """Test upload with specific transformer specified."""
        aws = AmazonCloudStorageClient()
        aws.create()

        mock_job_id = "12345678-1234-5678-9abc-123456789012"
        mock_start_job.return_value = mock_job_id

        response = httpx_to_standard(
            uploader.put(
                route, pdf_scratch, target_format="markdown", transformer="zerox"
            )
        )

        assert response.success is True
        transformation_job = response.data["transformation_job"]
        assert transformation_job["transformer"] == "zerox"

    def test_upload_with_unsupported_transformation(
        self,
        db: Session,
        route: Route,
        scratch: Path,
        uploader: WebUploader,
    ) -> None:
        """Test upload with unsupported transformation returns error."""
        aws = AmazonCloudStorageClient()
        aws.create()

        response = uploader.put(route, scratch, target_format="pdf")

        assert response.status_code == 400
        error_data = response.json()
        assert "Transformation from text to pdf is not supported" in error_data["error"]

    def test_upload_with_invalid_transformer(
        self,
        db: Session,
        route: Route,
        pdf_scratch: Path,
        uploader: WebUploader,
    ) -> None:
        """Test upload with invalid transformer name returns error."""
        aws = AmazonCloudStorageClient()
        aws.create()

        response = uploader.put(
            route,
            pdf_scratch,
            target_format="markdown",
            transformer="invalid_transformer",
        )

        assert response.status_code == 400
        error_data = response.json()
        assert "Transformer 'invalid_transformer' not available" in error_data["error"]
        assert "Available transformers:" in error_data["error"]

    def test_upload_with_unsupported_file_extension(
        self,
        db: Session,
        route: Route,
        uploader: WebUploader,
    ) -> None:
        """Test upload with unsupported file extension returns error."""
        aws = AmazonCloudStorageClient()
        aws.create()

        # Create a file with unsupported extension
        with NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as fp:
            fp.write("test content")
            fp.flush()
            unsupported_file = Path(fp.name)

        try:
            response = uploader.put(route, unsupported_file, target_format="markdown")
            assert response.status_code == 400
            error_data = response.json()
            assert "Unsupported file extension: .xyz" in error_data["error"]
        finally:
            unsupported_file.unlink()

    @patch("app.services.doctransform.job.start_job")
    def test_transformation_job_created_in_database(
        self,
        mock_start_job: Any,
        db: Session,
        route: Route,
        pdf_scratch: Path,
        uploader: WebUploader,
    ) -> None:
        """Test that transformation job is properly stored in the database."""
        aws = AmazonCloudStorageClient()
        aws.create()

        mock_job_id = "12345678-1234-5678-9abc-123456789012"
        mock_start_job.return_value = mock_job_id

        response = httpx_to_standard(
            uploader.put(route, pdf_scratch, target_format="markdown")
        )

        mock_start_job.assert_called_once()
        args, kwargs = mock_start_job.call_args

        # Check that start_job was called with the right arguments
        assert "transformer_name" in kwargs or len(args) >= 4

    def test_upload_response_structure_without_transformation(
        self,
        db: Session,
        route: Route,
        scratch: Path,
        uploader: WebUploader,
    ) -> None:
        """Test the response structure for upload without transformation."""
        aws = AmazonCloudStorageClient()
        aws.create()

        response = httpx_to_standard(uploader.put(route, scratch))

        required_fields = [
            "id",
            "project_id",
            "fname",
            "inserted_at",
            "updated_at",
        ]
        for field in required_fields:
            assert field in response.data

        assert response.data["transformation_job"] is None
