from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.core.providers import Provider
from app.models.credentials import Credential
from app.core.security import decrypt_credentials
from app.tests.utils.utils import generate_random_string
from app.tests.utils.auth import TestAuthContext


def test_read_credentials(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/credentials/",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data, list)
    assert len(data) >= 2

    providers = [cred["provider"] for cred in data]
    assert Provider.OPENAI.value in providers
    assert Provider.LANGFUSE.value in providers


def test_read_provider_credential_langfuse(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test reading Langfuse credentials from data."""
    response = client.get(
        f"{settings.API_V1_STR}/credentials/provider/{Provider.LANGFUSE.value}",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    data = response_data.get("data", response_data)
    assert "secret_key" in data
    assert "public_key" in data
    assert "host" in data


def test_read_provider_credential_not_found(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test reading credentials for non-existent provider returns 404."""
    client.delete(
        f"{settings.API_V1_STR}/credentials/provider/{Provider.OPENAI.value}",
        headers={"X-API-KEY": user_api_key.key},
    )

    response = client.get(
        f"{settings.API_V1_STR}/credentials/provider/{Provider.OPENAI.value}",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 404
    assert "Provider credentials not found" in response.json()["error"]


def test_update_credentials(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test updating existing OpenAI credentials."""
    new_api_key = "sk-" + generate_random_string()
    update_data = {
        "provider": Provider.OPENAI.value,
        "credential": {
            "api_key": new_api_key,
            "model": "gpt-4-turbo",
            "temperature": 0.8,
        },
    }

    response = client.patch(
        f"{settings.API_V1_STR}/credentials/",
        json=update_data,
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["provider"] == Provider.OPENAI.value
    assert data[0]["credential"]["model"] == "gpt-4-turbo"
    assert data[0]["updated_at"] is not None

    verify_response = client.get(
        f"{settings.API_V1_STR}/credentials/provider/{Provider.OPENAI.value}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert verify_response.status_code == 200
    verify_data = verify_response.json().get("data", verify_response.json())
    assert verify_data["api_key"] == new_api_key


def test_create_credential(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test deleting and recreating OpenAI credentials."""
    delete_response = client.delete(
        f"{settings.API_V1_STR}/credentials/provider/{Provider.OPENAI.value}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert delete_response.status_code == 200

    api_key = "sk-" + generate_random_string(10)
    credential_data = {
        "organization_id": user_api_key.organization_id,
        "project_id": user_api_key.project_id,
        "is_active": True,
        "credential": {
            Provider.OPENAI.value: {
                "api_key": api_key,
                "model": "gpt-4",
                "temperature": 0.7,
            }
        },
    }

    create_response = client.post(
        f"{settings.API_V1_STR}/credentials/",
        json=credential_data,
        headers={"X-API-KEY": user_api_key.key},
    )

    assert create_response.status_code == 200
    data = create_response.json()["data"]
    assert len(data) == 1
    assert data[0]["provider"] == Provider.OPENAI.value
    assert data[0]["credential"]["api_key"] == api_key


def test_credential_encryption(
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Verify credentials are encrypted in database."""
    db_credential = db.exec(
        select(Credential).where(
            Credential.organization_id == user_api_key.organization_id,
            Credential.project_id == user_api_key.project_id,
            Credential.is_active,
            Credential.provider == Provider.OPENAI.value,
        )
    ).first()

    assert db_credential is not None
    assert isinstance(db_credential.credential, str)

    decrypted_creds = decrypt_credentials(db_credential.credential)
    assert "api_key" in decrypted_creds
    assert decrypted_creds["api_key"].startswith("sk-")


def test_update_nonexistent_provider_returns_404(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test updating credentials for non-existent provider."""
    # Delete OpenAI first
    client.delete(
        f"{settings.API_V1_STR}/credentials/provider/{Provider.OPENAI.value}",
        headers={"X-API-KEY": user_api_key.key},
    )

    update_data = {
        "provider": Provider.OPENAI.value,
        "credential": {
            "api_key": "sk-" + generate_random_string(),
            "model": "gpt-4-turbo",
        },
    }

    response = client.patch(
        f"{settings.API_V1_STR}/credentials/",
        json=update_data,
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 404
    assert "Credentials not found" in response.json()["error"]


def test_create_ignores_mismatched_ids(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that route uses API key context, ignoring body IDs."""
    client.delete(
        f"{settings.API_V1_STR}/credentials/provider/{Provider.OPENAI.value}",
        headers={"X-API-KEY": user_api_key.key},
    )

    credential_data = {
        "organization_id": 999999,
        "project_id": 999999,
        "is_active": True,
        "credential": {
            Provider.OPENAI.value: {"api_key": "sk-test123", "model": "gpt-4"}
        },
    }

    response = client.post(
        f"{settings.API_V1_STR}/credentials/",
        json=credential_data,
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    data = response.json()["data"][0]
    assert data["organization_id"] == user_api_key.organization_id
    assert data["project_id"] == user_api_key.project_id


def test_duplicate_credential_creation_fails(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that creating duplicate credentials fails with 400."""
    api_key = "sk-" + generate_random_string(10)
    duplicate_credential = {
        "organization_id": user_api_key.organization_id,
        "project_id": user_api_key.project_id,
        "is_active": True,
        "credential": {
            Provider.OPENAI.value: {
                "api_key": api_key,
                "model": "gpt-4",
            }
        },
    }

    response = client.post(
        f"{settings.API_V1_STR}/credentials/",
        json=duplicate_credential,
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 400
    assert "already exist" in response.json()["error"]


def test_delete_all_credentials(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test deleting all credentials for a project."""
    response = client.delete(
        f"{settings.API_V1_STR}/credentials/",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["data"]["message"] == "All credentials deleted successfully"

    get_response = client.get(
        f"{settings.API_V1_STR}/credentials/",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert get_response.status_code == 404


def test_delete_all_when_none_exist_returns_404(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test deleting when no credentials exist."""
    client.delete(
        f"{settings.API_V1_STR}/credentials/",
        headers={"X-API-KEY": user_api_key.key},
    )

    response = client.delete(
        f"{settings.API_V1_STR}/credentials/",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 404
