"""Tests for the bucket provider registry."""

import pytest
from unittest.mock import MagicMock, patch

from sqlmodel import Session

from app.services.buckets.providers.base import BaseBucketProvider
from app.services.buckets.providers.gcs import GCSBucketProvider
from app.services.buckets.providers.registry import (
    BucketProvider,
    get_bucket_provider,
)
from app.tests.utils.utils import get_project


class TestBucketProviderRegistry:
    def test_get_provider_class_returns_gcs(self):
        assert BucketProvider.get_provider_class("gcs") is GCSBucketProvider

    def test_registry_values_are_provider_classes(self):
        for provider_type, provider_class in BucketProvider._registry.items():
            assert issubclass(
                provider_class, BaseBucketProvider
            ), f"Provider '{provider_type}' must inherit from BaseBucketProvider"

    def test_get_provider_class_unknown_raises(self):
        with pytest.raises(ValueError) as exc_info:
            BucketProvider.get_provider_class("s3")
        message = str(exc_info.value)
        assert "s3" in message
        assert "is not supported" in message


class TestGetBucketProvider:
    def test_get_bucket_provider_with_gcs(self, db: Session):
        project = get_project(db)

        credential = {
            "gcs_bucket": "byok-bucket",
            "sa_key": {"project_id": "byok-project"},
        }

        with (
            patch("app.crud.credentials.get_provider_credential") as mock_get_creds,
            patch(
                "app.services.buckets.providers.gcs.service_account."
                "Credentials.from_service_account_info"
            ),
            patch("app.services.buckets.providers.gcs.gcs.Client") as mock_client,
        ):
            mock_get_creds.return_value = credential
            mock_client.return_value = MagicMock()

            provider = get_bucket_provider(
                session=db,
                provider_type="gcs",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GCSBucketProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-gcp",
                project_id=project.id,
                org_id=project.organization_id,
            )

    def test_get_bucket_provider_missing_credential_raises(self, db: Session):
        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.return_value = None

            with pytest.raises(ValueError) as exc_info:
                get_bucket_provider(
                    session=db,
                    provider_type="gcs",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

        assert "google-gcp" in str(exc_info.value)
        assert "not configured" in str(exc_info.value)

    def test_get_bucket_provider_unknown_type_raises(self, db: Session):
        project = get_project(db)

        with pytest.raises(ValueError) as exc_info:
            get_bucket_provider(
                session=db,
                provider_type="s3",
                project_id=project.id,
                organization_id=project.organization_id,
            )

        assert "s3" in str(exc_info.value)
