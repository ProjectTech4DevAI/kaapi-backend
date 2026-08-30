"""
Tests for the LLM provider registry.
"""

import pytest
from unittest.mock import patch

from sqlmodel import Session
from openai import OpenAI

from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.open_ai import OpenAIProvider
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

    def test_get_llm_provider_google_gcp_falls_back_to_platform_credentials(
        self, db: Session
    ) -> None:
        """No stored ``google-gcp`` credential row shouldn't raise — the
        provider's create_client() falls back to platform-shared settings."""
        from app.services.llm.providers.google_gcp import GoogleGCPProvider

        project = get_project(db)

        with (
            patch("app.crud.credentials.get_provider_credential") as mock_get_creds,
            patch("app.services.llm.providers.google_gcp.settings") as mock_settings,
        ):
            mock_get_creds.return_value = None
            mock_settings.GCP_VERTEX_API_KEY = "platform-key"
            mock_settings.GCP_PROJECT_ID = "platform-project"
            mock_settings.GCP_VERTEX_LOCATION = "us-central1"
            mock_settings.GCS_AUDIO_BUCKET = "platform-bucket"
            mock_settings.GCP_SA_KEY = '{"type": "service_account"}'

            provider = get_llm_provider(
                session=db,
                provider_type="google-gcp",
                project_id=project.id,
                organization_id=project.organization_id,
            )

        assert isinstance(provider, GoogleGCPProvider)
        assert provider.client.api_key == "platform-key"

    def test_get_llm_provider_google_gcp_raises_when_platform_credentials_missing(
        self, db: Session
    ) -> None:
        """Platform fallback still surfaces a ValueError if settings are
        also unconfigured, instead of silently constructing a broken client."""
        project = get_project(db)

        with (
            patch("app.crud.credentials.get_provider_credential") as mock_get_creds,
            patch("app.services.llm.providers.google_gcp.settings") as mock_settings,
        ):
            mock_get_creds.return_value = None
            mock_settings.GCP_VERTEX_API_KEY = None
            mock_settings.GCP_PROJECT_ID = None
            mock_settings.GCP_VERTEX_LOCATION = None
            mock_settings.GCS_AUDIO_BUCKET = None
            mock_settings.GCP_SA_KEY = None

            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="google-gcp",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

        assert "missing required fields" in str(exc_info.value)

    def test_google_native_routes_to_aistudio(self, db: Session):
        """``google`` / ``google-native`` currently route to GoogleAIProvider
        (AI Studio). Vertex routing is temporarily disabled."""
        from app.services.llm.providers.google_aistudio import GoogleAIProvider

        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.return_value = {"api_key": "test-api-key"}

            provider = get_llm_provider(
                session=db,
                provider_type="google-native",
                project_id=project.id,
                organization_id=project.organization_id,
            )

        assert isinstance(provider, GoogleAIProvider)
