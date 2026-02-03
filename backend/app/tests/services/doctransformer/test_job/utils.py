"""
Base test class for DocTransform service tests.

This module contains DocTransformTestBase with common AWS S3 setup and utilities.
All fixtures are automatically available from conftest.py in the same directory.
"""
from pathlib import Path
from urllib.parse import urlparse

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.services.doctransform.transformer import Transformer
from app.models import Document


class MockTestTransformer(Transformer):
    """
    A mock transformer for testing that returns a hardcoded lorem ipsum string.
    """

    def transform(self, input_path: Path, output_path: Path) -> Path:
        content = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        )
        output_path.write_text(content, encoding="utf-8")
        return output_path


class DocTransformTestBase:
    """Base class for document transformation tests with common setup and utilities."""

    def setup_aws_s3(self) -> AmazonCloudStorageClient:
        """Setup AWS S3 for testing."""
        aws = AmazonCloudStorageClient()
        aws.create()
        return aws

    def create_s3_document_content(
        self,
        aws: AmazonCloudStorageClient,
        document: Document,
        content: bytes = b"Test document content",
    ) -> bytes:
        """Create content in S3 for a document."""
        parsed_url = urlparse(document.object_store_url)
        s3_key = parsed_url.path.lstrip("/")

        aws.client.put_object(Bucket=settings.AWS_S3_BUCKET, Key=s3_key, Body=content)
        return content

    def verify_s3_content(
        self,
        aws: AmazonCloudStorageClient,
        transformed_doc: Document,
        expected_content: str = None,
    ) -> None:
        """Verify the content stored in S3."""
        if expected_content is None:
            expected_content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."

        parsed_url = urlparse(transformed_doc.object_store_url)
        transformed_key = parsed_url.path.lstrip("/")

        response = aws.client.get_object(
            Bucket=settings.AWS_S3_BUCKET, Key=transformed_key
        )
        transformed_content = response["Body"].read().decode("utf-8")
        assert transformed_content == expected_content


def create_failing_convert_document(fail_count: int = 1):
    """Create a side effect function that fails specified times then succeeds."""
    call_count = 0

    def failing_convert_document(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= fail_count:
            raise Exception("Transient error")
        output_path = args[1] if len(args) > 1 else kwargs.get("output_path")
        if output_path:
            output_path.write_text("Success after retries", encoding="utf-8")
            return output_path
        raise ValueError("output_path is required")

    return failing_convert_document


def create_persistent_failing_convert_document(
    error_message: str = "Persistent error",
):
    """Create a side effect function that always fails."""

    def persistent_failing_convert_document(*args, **kwargs):
        raise Exception(error_message)

    return persistent_failing_convert_document
