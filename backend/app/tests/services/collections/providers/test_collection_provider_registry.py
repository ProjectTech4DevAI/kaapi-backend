"""Tests for the collections provider registry — specifically the google-aistudio
client-construction branch of get_llm_provider."""

import pytest
from sqlmodel import Session

from app.core.security import encrypt_credentials
from app.crud import set_creds_for_org
from app.models import Credential, CredsCreate
from app.services.collections.providers.gemini import GeminiAIStudioProvider
from app.services.collections.providers.registry import get_llm_provider
from app.tests.utils.test_data import create_test_project


class TestGetLLMProviderGoogleAIStudio:
    def test_returns_gemini_provider_with_api_key_credential(self, db: Session) -> None:
        project = create_test_project(db)
        set_creds_for_org(
            session=db,
            creds_add=CredsCreate(
                is_active=True,
                credential={"google-aistudio": {"api_key": "test-gemini-key"}},
            ),
            organization_id=project.organization_id,
            project_id=project.id,
        )

        provider = get_llm_provider(
            session=db,
            provider="google-aistudio",
            project_id=project.id,
            organization_id=project.organization_id,
        )

        assert isinstance(provider, GeminiAIStudioProvider)

    def test_credential_without_api_key_raises(self, db: Session) -> None:
        project = create_test_project(db)
        # Bypass set_creds_for_org validation (which requires api_key) to persist a
        # malformed row and exercise the missing-key branch against the real db.
        db.add(
            Credential(
                organization_id=project.organization_id,
                project_id=project.id,
                is_active=True,
                provider="google-aistudio",
                credential=encrypt_credentials({"model": "gemini-2.5-pro"}),
            )
        )
        db.commit()

        with pytest.raises(
            ValueError, match="Google AI Studio credentials not configured"
        ):
            get_llm_provider(
                session=db,
                provider="google-aistudio",
                project_id=project.id,
                organization_id=project.organization_id,
            )

    def test_no_credential_row_raises(self, db: Session) -> None:
        project = create_test_project(db)

        with pytest.raises(ValueError, match="not configured for this project"):
            get_llm_provider(
                session=db,
                provider="google-aistudio",
                project_id=project.id,
                organization_id=project.organization_id,
            )
