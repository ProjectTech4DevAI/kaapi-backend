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

    def test_google_native_routes_to_aistudio(self, db: Session):
        """``google`` / ``google-native`` route to GoogleAIProvider when
        GEMINI_DEFAULT_INFERENCE_ROUTE=aistudio. Credential row is fetched
        under the ``google-aistudio`` key, not ``google``."""
        from app.services.llm.providers.google_aistudio import GoogleAIProvider

        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "aistudio"
            mock_get_creds.return_value = {"api_key": "test-api-key"}

            provider = get_llm_provider(
                session=db,
                provider_type="google-native",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleAIProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-aistudio",
                project_id=project.id,
                org_id=project.organization_id,
            )

    def test_google_routes_to_vertex_when_env_set(self, db: Session):
        """When route=vertex, ``google`` resolves to GoogleVertexAIProvider
        and looks up credentials under the ``google`` key."""
        from app.services.llm.providers.google_ai import GoogleVertexAIProvider

        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "vertex"
            mock_get_creds.return_value = {
                "api_key": "byok-key",
                "project_id": "byok-project",
                "location": "us-central1",
            }

            provider = get_llm_provider(
                session=db,
                provider_type="google",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleVertexAIProvider)
            assert provider.client.api_key == "byok-key"
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google",
                project_id=project.id,
                org_id=project.organization_id,
            )

    def test_vertex_falls_back_to_platform_defaults_when_no_creds(self, db: Session):
        """Vertex must not raise when DB has no row for ``google``; it falls
        back to platform defaults from settings."""
        from app.services.llm.providers.google_ai import GoogleVertexAIProvider

        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as registry_settings, patch(
            "app.services.llm.providers.google_ai.settings"
        ) as google_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            registry_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "vertex"
            google_settings.GCP_VERTEX_API_KEY = "platform-key"
            google_settings.GCP_PROJECT_ID = "platform-project"
            google_settings.GCP_VERTEX_LOCATION = "us-central1"
            google_settings.GCP_SA_KEY = ""
            google_settings.GCS_AUDIO_BUCKET = ""
            mock_get_creds.return_value = None

            provider = get_llm_provider(
                session=db,
                provider_type="google",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleVertexAIProvider)
            assert provider.client.api_key == "platform-key"
            assert provider.client.project_id == "platform-project"

    def test_invalid_inference_route_raises(self, db: Session):
        """Unknown GEMINI_DEFAULT_INFERENCE_ROUTE value raises ValueError."""
        project = get_project(db)

        with patch("app.services.llm.providers.registry.settings") as mock_settings:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "bogus"

            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="google",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

            assert "GEMINI_DEFAULT_INFERENCE_ROUTE" in str(exc_info.value)

    def test_aistudio_caller_overridden_to_vertex_when_env_vertex(self, db: Session):
        """When env=vertex, callers asking for google-aistudio are forced
        to GoogleVertexAIProvider (platform-wide failover)."""
        from app.services.llm.providers.google_ai import GoogleVertexAIProvider

        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "vertex"
            mock_get_creds.return_value = {
                "api_key": "byok-key",
                "project_id": "byok-project",
                "location": "us-central1",
            }

            provider = get_llm_provider(
                session=db,
                provider_type="google-aistudio",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleVertexAIProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google",
                project_id=project.id,
                org_id=project.organization_id,
            )

    def test_google_caller_overridden_to_aistudio_when_env_aistudio(self, db: Session):
        """When env=aistudio, callers asking for google are forced to
        GoogleAIProvider; missing creds must raise (no platform fallback)."""
        from app.services.llm.providers.google_aistudio import GoogleAIProvider

        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "aistudio"
            mock_get_creds.return_value = {"api_key": "test-api-key"}

            provider = get_llm_provider(
                session=db,
                provider_type="google",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleAIProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-aistudio",
                project_id=project.id,
                org_id=project.organization_id,
            )

            mock_get_creds.return_value = None
            with pytest.raises(ValueError):
                get_llm_provider(
                    session=db,
                    provider_type="google",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

    def test_env_unset_respects_caller_choice(self, db: Session):
        """env='' → google routes to vertex, google-aistudio routes to aistudio."""
        from app.services.llm.providers.google_ai import GoogleVertexAIProvider
        from app.services.llm.providers.google_aistudio import GoogleAIProvider

        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = ""
            mock_get_creds.return_value = {
                "api_key": "k",
                "project_id": "p",
                "location": "us-central1",
            }

            vertex = get_llm_provider(
                session=db,
                provider_type="google",
                project_id=project.id,
                organization_id=project.organization_id,
            )
            assert isinstance(vertex, GoogleVertexAIProvider)

            aistudio = get_llm_provider(
                session=db,
                provider_type="google-aistudio",
                project_id=project.id,
                organization_id=project.organization_id,
            )
            assert isinstance(aistudio, GoogleAIProvider)
