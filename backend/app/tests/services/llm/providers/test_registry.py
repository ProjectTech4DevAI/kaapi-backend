"""
Tests for the LLM provider registry.
"""

import pytest
from unittest.mock import patch

from sqlmodel import Session
from openai import OpenAI

from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.open_ai import OpenAIProvider
from app.services.llm.providers.google_ai import GoogleVertexAIProvider
from app.services.llm.providers.google_aistudio import GoogleAIProvider
from app.services.llm.providers.registry import (
    LLMProvider,
    get_llm_provider,
)
from app.tests.utils.utils import get_project


class TestProviderRegistry:
    def test_registry_contains_openai(self):
        assert "openai-native" in LLMProvider._registry
        assert LLMProvider._registry["openai-native"] == OpenAIProvider

    def test_registry_values_are_provider_classes(self):
        for provider_type, provider_class in LLMProvider._registry.items():
            assert issubclass(provider_class, BaseProvider)


class TestGetLLMProvider:
    def test_get_llm_provider_with_openai(self, db: Session):
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

    def test_get_llm_provider_with_invalid_provider(self, db: Session):
        project = get_project(db)

        with pytest.raises(ValueError) as exc_info:
            get_llm_provider(
                session=db,
                provider_type="invalid_provider",
                project_id=project.id,
                organization_id=project.organization_id,
            )

        assert "invalid_provider" in str(exc_info.value)
        assert "is not supported" in str(exc_info.value)

    def test_openai_missing_credentials_raises(self, db: Session):
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

    # ---- Explicit google-vertex / google-aistudio: bypass env, need creds ----

    def test_explicit_google_vertex_bypasses_env(self, db: Session):
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "aistudio"
            mock_get_creds.return_value = {
                "api_key": "byok-key",
                "project_id": "byok-project",
                "location": "us-central1",
            }

            provider = get_llm_provider(
                session=db,
                provider_type="google-vertex",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleVertexAIProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-vertex",
                project_id=project.id,
                org_id=project.organization_id,
            )

    def test_explicit_google_aistudio_bypasses_env(self, db: Session):
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "vertex"
            mock_get_creds.return_value = {"api_key": "byok-key"}

            provider = get_llm_provider(
                session=db,
                provider_type="google-aistudio",
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

    def test_explicit_google_vertex_without_creds_raises(self, db: Session):
        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.return_value = None

            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="google-vertex",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

            assert "google-vertex" in str(exc_info.value)

    def test_explicit_google_aistudio_without_creds_raises(self, db: Session):
        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.return_value = None

            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="google-aistudio",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

            assert "google-aistudio" in str(exc_info.value)

    # ---- Platform-routed `google`: env decides, platform fallback allowed ----

    def test_google_routes_to_vertex_when_env_vertex(self, db: Session):
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
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-vertex",
                project_id=project.id,
                org_id=project.organization_id,
            )

    def test_google_routes_to_aistudio_when_env_aistudio(self, db: Session):
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "aistudio"
            mock_get_creds.return_value = {"api_key": "byok-key"}

            provider = get_llm_provider(
                session=db,
                provider_type="google",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleAIProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google",
                project_id=project.id,
                org_id=project.organization_id,
            )

    def test_google_vertex_route_falls_back_to_platform_defaults(self, db: Session):
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

    def test_google_aistudio_route_without_google_row_raises(self, db: Session):
        """env=aistudio keeps today's behavior: tenant `google` row or error."""
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as registry_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            registry_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "aistudio"
            mock_get_creds.return_value = None

            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="google",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

            assert "'google'" in str(exc_info.value)

    def test_explicit_google_aistudio_falls_back_to_google_row(self, db: Session):
        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.side_effect = lambda **kwargs: (
                {"api_key": "legacy-google-key"}
                if kwargs["provider"] == "google"
                else None
            )

            provider = get_llm_provider(
                session=db,
                provider_type="google-aistudio",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleAIProvider)
            assert [c.kwargs["provider"] for c in mock_get_creds.call_args_list] == [
                "google-aistudio",
                "google",
            ]

    def test_unknown_route_value_defaults_to_aistudio(self, db: Session):
        """Typo/unset env must not silently flip traffic onto vertex."""
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as registry_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            registry_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = ""
            mock_get_creds.return_value = {"api_key": "byok-key"}

            provider = get_llm_provider(
                session=db,
                provider_type="google-native",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleAIProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google",
                project_id=project.id,
                org_id=project.organization_id,
            )
