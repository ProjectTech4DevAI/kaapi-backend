"""
Tests for the LLM provider registry.
"""

import pytest
from unittest.mock import patch

from sqlmodel import Session
from openai import OpenAI

from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.open_ai import OpenAIProvider
from app.services.llm.providers.google_gcp import GoogleGCPProvider
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

    # ---- Explicit google-gcp / google-aistudio: bypass env, need creds ----

    def test_explicit_google_gcp_bypasses_env(self, db: Session):
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "google-aistudio"
            mock_get_creds.return_value = {
                "api_key": "byok-key",
                "project_id": "byok-project",
                "location": "us-central1",
            }

            provider = get_llm_provider(
                session=db,
                provider_type="google-gcp",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleGCPProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-gcp",
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
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "google-gcp"
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

    def test_explicit_google_gcp_without_creds_raises(self, db: Session):
        project = get_project(db)

        with patch("app.crud.credentials.get_provider_credential") as mock_get_creds:
            mock_get_creds.return_value = None

            with pytest.raises(ValueError) as exc_info:
                get_llm_provider(
                    session=db,
                    provider_type="google-gcp",
                    project_id=project.id,
                    organization_id=project.organization_id,
                )

            assert "google-gcp" in str(exc_info.value)

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

    def test_google_routes_to_gcp_when_env_gcp(self, db: Session):
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as mock_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "google-gcp"
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

            assert isinstance(provider, GoogleGCPProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-gcp",
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
            mock_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "google-aistudio"
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

    def test_google_gcp_route_falls_back_to_platform_defaults(self, db: Session):
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as registry_settings, patch(
            "app.services.llm.providers.google_gcp.settings"
        ) as google_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            registry_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "google-gcp"
            google_settings.GOOGLE_GCP_API_KEY = "platform-key"
            google_settings.GOOGLE_GCP_PROJECT_ID = "platform-project"
            google_settings.GOOGLE_GCP_PROJECT_LOCATION = "us-central1"
            google_settings.GOOGLE_GCP_SA_KEY = ""
            google_settings.GOOGLE_GCS_AUDIO_BUCKET = ""
            mock_get_creds.return_value = None

            provider = get_llm_provider(
                session=db,
                provider_type="google",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleGCPProvider)
            assert provider.client.api_key == "platform-key"
            assert provider.client.project_id == "platform-project"

    def test_google_aistudio_route_falls_back_to_platform_key(self, db: Session):
        """env=google-aistudio with no tenant `google` row uses the platform key."""
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as registry_settings, patch(
            "app.services.llm.providers.google_aistudio.settings"
        ) as google_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            registry_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = "google-aistudio"
            google_settings.GOOGLE_AISTUDIO_API_KEY = "platform-aistudio-key"
            mock_get_creds.return_value = None

            provider = get_llm_provider(
                session=db,
                provider_type="google",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleAIProvider)

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

    def test_unknown_route_value_defaults_to_gcp(self, db: Session):
        """GCP is the default flow; only an explicit google-aistudio opts out."""
        project = get_project(db)

        with patch(
            "app.services.llm.providers.registry.settings"
        ) as registry_settings, patch(
            "app.crud.credentials.get_provider_credential"
        ) as mock_get_creds:
            registry_settings.GEMINI_DEFAULT_INFERENCE_ROUTE = ""
            mock_get_creds.return_value = {
                "api_key": "byok-key",
                "project_id": "byok-project",
                "location": "us-central1",
            }

            provider = get_llm_provider(
                session=db,
                provider_type="google-native",
                project_id=project.id,
                organization_id=project.organization_id,
            )

            assert isinstance(provider, GoogleGCPProvider)
            mock_get_creds.assert_called_once_with(
                session=db,
                provider="google-gcp",
                project_id=project.id,
                org_id=project.organization_id,
            )
