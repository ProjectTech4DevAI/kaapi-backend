import pytest
from sqlmodel import Session

from app.crud import (
    set_creds_for_org,
    get_creds_by_org,
    get_provider_credential,
    update_creds_for_org,
    remove_provider_credential,
    remove_creds_for_org,
)
from app.models import CredsCreate, CredsUpdate
from app.core.providers import Provider
from app.tests.utils.test_data import (
    create_test_project,
    create_test_credential,
    test_credential_data,
)


def test_set_credentials_for_org(db: Session) -> None:
    """Test setting credentials for an organization."""
    project = create_test_project(db)

    credentials_data = {
        "openai": {"api_key": "test-openai-key"},
        "langfuse": {
            "public_key": "test-public-key",
            "secret_key": "test-secret-key",
            "host": "https://cloud.langfuse.com",
        },
    }
    credentials_create = CredsCreate(
        is_active=True,
        credential=credentials_data,
    )

    created_credentials = set_creds_for_org(
        session=db,
        creds_add=credentials_create,
        organization_id=project.organization_id,
        project_id=project.id,
    )

    assert len(created_credentials) == 2
    assert all(
        cred.organization_id == project.organization_id for cred in created_credentials
    )
    assert all(cred.project_id == project.id for cred in created_credentials)
    assert all(cred.is_active for cred in created_credentials)
    assert {cred.provider for cred in created_credentials} == {"openai", "langfuse"}


def test_get_creds_by_org(db: Session) -> None:
    """Test retrieving all credentials for an organization."""
    project = create_test_project(db)

    credentials_data = {
        "openai": {"api_key": "test-openai-key"},
        "langfuse": {
            "public_key": "test-public-key",
            "secret_key": "test-secret-key",
            "host": "https://cloud.langfuse.com",
        },
    }

    credentials_create = CredsCreate(
        is_active=True,
        credential=credentials_data,
    )
    set_creds_for_org(
        session=db,
        creds_add=credentials_create,
        organization_id=project.organization_id,
        project_id=project.id,
    )

    retrieved_creds = get_creds_by_org(
        session=db, org_id=project.organization_id, project_id=project.id
    )

    assert len(retrieved_creds) == 2
    assert all(
        cred.organization_id == project.organization_id for cred in retrieved_creds
    )
    assert {cred.provider for cred in retrieved_creds} == {"openai", "langfuse"}


def test_get_provider_credential(db: Session) -> None:
    """Test retrieving credentials for a specific provider."""
    credentials_create = test_credential_data(db)
    original_api_key = credentials_create.credential[Provider.OPENAI.value]["api_key"]

    project = create_test_project(db)
    set_creds_for_org(
        session=db,
        creds_add=credentials_create,
        organization_id=project.organization_id,
        project_id=project.id,
    )
    retrieved_cred = get_provider_credential(
        session=db,
        org_id=project.organization_id,
        provider="openai",
        project_id=project.id,
    )

    assert retrieved_cred is not None
    assert "api_key" in retrieved_cred
    assert retrieved_cred["api_key"] == original_api_key


def test_update_creds_for_org(db: Session) -> None:
    """Test updating credentials for a provider."""
    _, project = create_test_credential(db)

    credential = get_provider_credential(
        session=db,
        org_id=project.organization_id,
        provider="openai",
        project_id=project.id,
        full=True,
    )
    updated_creds = {"api_key": "updated-key"}
    creds_update = CredsUpdate(provider="openai", credential=updated_creds)

    updated = update_creds_for_org(
        session=db,
        org_id=credential.organization_id,
        creds_in=creds_update,
        project_id=project.id,
    )

    assert len(updated) == 1
    assert updated[0].provider == "openai"
    retrieved_cred = get_provider_credential(
        session=db,
        org_id=credential.organization_id,
        provider="openai",
        project_id=project.id,
    )
    assert retrieved_cred["api_key"] == "updated-key"


def test_remove_provider_credential(db: Session) -> None:
    """Test removing credentials for a specific provider."""
    _, project = create_test_credential(db)

    credential = get_provider_credential(
        session=db,
        org_id=project.organization_id,
        provider="openai",
        project_id=project.id,
        full=True,
    )

    remove_provider_credential(
        session=db,
        org_id=credential.organization_id,
        provider="openai",
        project_id=project.id,
    )

    creds = get_provider_credential(
        session=db,
        org_id=credential.organization_id,
        provider="openai",
        project_id=project.id,
    )
    assert creds is None


def test_remove_creds_for_org(db: Session) -> None:
    """Test removing all credentials for an organization."""
    project = create_test_project(db)

    credentials_data = {
        "openai": {"api_key": "test-openai-key"},
        "langfuse": {
            "public_key": "test-public-key",
            "secret_key": "test-secret-key",
            "host": "https://cloud.langfuse.com",
        },
    }

    creds_create = CredsCreate(
        is_active=True,
        credential=credentials_data,
    )
    set_creds_for_org(
        session=db,
        creds_add=creds_create,
        organization_id=project.organization_id,
        project_id=project.id,
    )

    remove_creds_for_org(
        session=db, org_id=project.organization_id, project_id=project.id
    )

    creds = get_creds_by_org(
        session=db, org_id=project.organization_id, project_id=project.id
    )
    assert creds == []


def test_invalid_provider(db: Session) -> None:
    """Test handling of invalid provider names."""
    project = create_test_project(db)

    credentials_data = {"gemini": {"api_key": "test-key"}}
    credentials_create = CredsCreate(
        is_active=True,
        credential=credentials_data,
    )

    with pytest.raises(ValueError, match="Unsupported provider"):
        set_creds_for_org(
            session=db,
            creds_add=credentials_create,
            organization_id=project.organization_id,
            project_id=project.id,
        )


def test_duplicate_provider_credentials(db: Session) -> None:
    """Test handling of duplicate provider credentials."""
    project = create_test_project(db)

    credentials_data = {"openai": {"api_key": "test-key"}}

    credentials_create = CredsCreate(
        is_active=True,
        credential=credentials_data,
    )
    set_creds_for_org(
        session=db,
        creds_add=credentials_create,
        organization_id=project.organization_id,
        project_id=project.id,
    )

    existing_creds = get_provider_credential(
        session=db,
        org_id=project.organization_id,
        provider="openai",
        project_id=project.id,
    )
    assert existing_creds is not None
    assert "api_key" in existing_creds
    assert existing_creds["api_key"] == "test-key"


def test_langfuse_credential_validation(db: Session) -> None:
    """Test validation of Langfuse credentials structure."""
    project = create_test_project(db)

    # Test with missing required fields
    invalid_credentials = {
        "langfuse": {
            "public_key": "test-public-key",
            "secret_key": "test-secret-key"
            # Missing host
        }
    }
    credentials_create = CredsCreate(
        is_active=True,
        credential=invalid_credentials,
    )

    with pytest.raises(ValueError):
        set_creds_for_org(
            session=db,
            creds_add=credentials_create,
            organization_id=project.organization_id,
            project_id=project.id,
        )

    valid_credentials = {
        "langfuse": {
            "public_key": "test-public-key",
            "secret_key": "test-secret-key",
            "host": "https://cloud.langfuse.com",
        }
    }

    credentials_create = CredsCreate(
        is_active=True,
        credential=valid_credentials,
    )

    created_credentials = set_creds_for_org(
        session=db,
        creds_add=credentials_create,
        organization_id=project.organization_id,
        project_id=project.id,
    )
    assert len(created_credentials) == 1
    assert created_credentials[0].provider == "langfuse"
