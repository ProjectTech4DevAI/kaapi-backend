from unittest.mock import patch

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


def test_set_credentials_for_google_vertex_with_sa_key(db: Session) -> None:
    """sa_key on google-vertex must be uploaded to SM and stripped before storage;
    the persisted credential dict carries only the secret reference."""
    project = create_test_project(db)

    sa_key = {
        "type": "service_account",
        "project_id": "starlit-lotus-492004-k0",
        "client_email": "test@starlit-lotus-492004-k0.iam.gserviceaccount.com",
        "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----",
    }
    payload = CredsCreate(
        is_active=True,
        credential={
            "google-vertex": {
                "api_key": "vkey",
                "project_id": "starlit-lotus-492004-k0",
                "location": "us-central1",
                "sa_key": sa_key,
                "gcs_bucket": "my-bucket",
            }
        },
    )

    with patch("app.crud.credentials.upsert_byok_secret_for_provider") as mock_hook:
        # Simulate the real hook's rewrite without touching AWS.
        secret_name = (
            f"kaapi/test/orgs/{project.organization_id}"
            f"/projects/{project.id}/google-vertex/sa"
        )
        mock_hook.return_value = {
            "api_key": "vkey",
            "project_id": "starlit-lotus-492004-k0",
            "location": "us-central1",
            "gcs_bucket": "my-bucket",
            "gcp_sa_secret_name": secret_name,
            "gcp_sa_secret_region": "ap-south-1",
        }

        created = set_creds_for_org(
            session=db,
            creds_add=payload,
            organization_id=project.organization_id,
            project_id=project.id,
        )

    mock_hook.assert_called_once()
    args, kwargs = mock_hook.call_args
    assert args[0] == "google-vertex"
    assert args[1]["sa_key"] == sa_key
    assert kwargs == {"org_id": project.organization_id, "project_id": project.id}

    assert len(created) == 1
    stored = get_provider_credential(
        session=db,
        org_id=project.organization_id,
        provider="google-vertex",
        project_id=project.id,
    )
    assert stored is not None
    assert "sa_key" not in stored
    assert stored["gcp_sa_secret_name"] == secret_name
    assert stored["gcp_sa_secret_region"] == "ap-south-1"
    assert stored["api_key"] == "vkey"


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
    from app.core.exception_handlers import HTTPException

    project = create_test_project(db)

    credentials_data = {"invalid_provider": {"api_key": "test-key"}}
    credentials_create = CredsCreate(
        is_active=True,
        credential=credentials_data,
    )

    with pytest.raises(HTTPException) as exc_info:
        set_creds_for_org(
            session=db,
            creds_add=credentials_create,
            organization_id=project.organization_id,
            project_id=project.id,
        )

    assert exc_info.value.status_code == 400
    assert "Unsupported provider" in exc_info.value.detail


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
    from app.core.exception_handlers import HTTPException

    project = create_test_project(db)

    # Test with missing required fields
    invalid_credentials = {
        "langfuse": {
            "public_key": "test-public-key",
            "secret_key": "test-secret-key",
            # Missing host
        }
    }
    credentials_create = CredsCreate(
        is_active=True,
        credential=invalid_credentials,
    )

    with pytest.raises(HTTPException) as exc_info:
        set_creds_for_org(
            session=db,
            creds_add=credentials_create,
            organization_id=project.organization_id,
            project_id=project.id,
        )

    assert exc_info.value.status_code == 400
    assert "Missing required fields for langfuse" in exc_info.value.detail

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
