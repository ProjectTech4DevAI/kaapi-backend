"""
Tests for the LLM provider registry.
"""

import pytest
from unittest.mock import patch

from sqlmodel import Session
from openai import OpenAI

from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.registry import (
    LLMProvider,
    get_llm_provider,
)
from app.tests.utils.utils import get_project


class TestProviderRegistry:
    """Test cases for the PROVIDER_REGISTRY constant."""

    def test_registry_contains_openai(self):
        """Test that registry contains OpenAI provider."""
        assert "openai-native" in LLMProvider._registry
        assert LLMProvider._registry["openai-native"] == OpenAIProvider

    def test_registry_values_are_provider_classes(self):
        """Test that all registry values are BaseProvider subclasses."""
        for provider_type, provider_class in LLMProvider._registry.items():
            assert issubclass(
                provider_class, BaseProvider
            ), f"Provider '{provider_type}' class must inherit from BaseProvider"


class TestGetLLMProvider:
    """Test cases for the get_llm_provider function."""

    def test_get_llm_provider_with_openai(self, db: Session):
        """Test getting OpenAI provider successfully."""
        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.return_value = {"api_key": "test-api-key"}

            provider = get_llm_provider(
                session=db,
                provider_type="openai-native",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, OpenAIProvider)
            assert isinstance(provider.client, OpenAI)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="openai",
                project_id=project.id,
                org_id=project.organization_id,
            )

            mock_get_creds.return_value = {"wrong_key": "value"}
            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="openai-native",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )
            assert "OpenAI credentials not configured for this project." in str(
                exc_info.value
            )

    def test_get_llm_provider_with_invalid_provider(self, db: Session):
        """Test that invalid provider type raises ValueError."""
        project = get_project(db)

        with pytest.raises(ValueError) as exc_info:
            get_llm_provider(
                session=db,
                provider_type="invalid_provider",
                project_id=project.id,
                organization_id=project.organization_id,
            )

        error_message = str(exc_info.value)
        assert "invalid_provider" in error_message
        assert "is not supported" in error_message
        assert "openai-native" in error_message

    def test_get_llm_provider_with_missing_credentials(self, db: Session):
        """Test handling of errors when credentials are not found."""
        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.return_value = None

            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="openai-native",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

            assert "not configured for this project" in str(exc_info.value)

    def test_google_vertex_falls_back_to_platform_settings(self, db: Session):
        """No credential row for google-vertex → create_client synthesizes the
        platform defaults from settings (api_key/project/location/bucket) and
        parses the inline SA JSON from GCP_SA_KEY."""
        import json as _json

        from app.services.llm.providers.gai_vertex import (
            GoogleVertexAIProvider,
            VertexClient,
        )

        project = get_project(db)
        sa_info = {
            "type": "service_account",
            "project_id": "platform-project",
            "client_email": "sa@platform-project.iam.gserviceaccount.com",
            "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----",
        }

        with patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds, patch(
            "app.services.llm.providers.gai_vertex.settings"
        ) as mock_settings:
            mock_get_creds.return_value = None
            mock_settings.GCP_VERTEX_API_KEY = "platform-key"
            mock_settings.GCP_PROJECT_ID = "platform-project"
            mock_settings.GCP_VERTEX_LOCATION = "us-central1"
            mock_settings.GCP_SA_KEY = _json.dumps(sa_info)
            mock_settings.GCS_AUDIO_BUCKET = "platform-bucket"

            provider = get_llm_provider(
                session=db,
                provider_type="google-vertex-native",
                project_id=project.id,
                organization_id=project.organization_id,
            )

        assert isinstance(provider, GoogleVertexAIProvider)
        assert isinstance(provider.client, VertexClient)
        assert provider.client.api_key == "platform-key"
        assert provider.client.project_id == "platform-project"
        assert provider.client.location == "us-central1"
        assert provider.client.sa_info == sa_info
        assert provider.client.gcs_bucket == "platform-bucket"
