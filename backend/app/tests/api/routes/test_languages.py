import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud.language import get_language_by_id, get_languages
from app.main import app


def test_list_languages(
    client: TestClient, db: Session, superuser_token_headers: dict[str, str]
) -> None:
    """Test retrieving list of all active languages."""
    response = client.get(
        f"{settings.API_V1_STR}/languages/",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert "data" in response_data
    assert "data" in response_data["data"]
    assert "count" in response_data["data"]
    assert isinstance(response_data["data"]["data"], list)
    assert response_data["data"]["count"] > 0


def test_list_languages_with_pagination(
    client: TestClient, db: Session, superuser_token_headers: dict[str, str]
) -> None:
    """Test retrieving languages with pagination parameters."""
    response = client.get(
        f"{settings.API_V1_STR}/languages/?skip=0&limit=5",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert len(response_data["data"]["data"]) <= 5


def test_get_language_by_id(
    client: TestClient, db: Session, superuser_token_headers: dict[str, str]
) -> None:
    """Test retrieving a specific language by ID."""
    # First get a language from the database
    languages = get_languages(session=db)
    assert len(languages) > 0
    language = languages[0]

    response = client.get(
        f"{settings.API_V1_STR}/languages/{language.id}",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert "data" in response_data
    assert response_data["data"]["id"] == language.id
    assert response_data["data"]["label"] == language.label
    assert response_data["data"]["locale"] == language.locale


def test_get_language_not_found(
    client: TestClient, db: Session, superuser_token_headers: dict[str, str]
) -> None:
    """Test retrieving a non-existent language returns 404."""
    response = client.get(
        f"{settings.API_V1_STR}/languages/99999",
        headers=superuser_token_headers,
    )

    assert response.status_code == 404
    response_data = response.json()
    assert response_data["success"] is False
    assert response_data["error"] == "Language not found"


def test_list_languages_with_api_key(
    client: TestClient, db: Session, superuser_api_key_header: dict[str, str]
) -> None:
    """Test retrieving languages using API key authentication."""
    response = client.get(
        f"{settings.API_V1_STR}/languages/",
        headers=superuser_api_key_header,
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert "data" in response_data


def test_get_language_with_api_key(
    client: TestClient, db: Session, superuser_api_key_header: dict[str, str]
) -> None:
    """Test retrieving a specific language using API key authentication."""
    languages = get_languages(session=db)
    assert len(languages) > 0
    language = languages[0]

    response = client.get(
        f"{settings.API_V1_STR}/languages/{language.id}",
        headers=superuser_api_key_header,
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["id"] == language.id


def test_list_languages_unauthorized(client: TestClient) -> None:
    """Test that listing languages without authentication returns 401."""
    response = client.get(f"{settings.API_V1_STR}/languages/")

    assert response.status_code == 401


def test_get_language_unauthorized(client: TestClient) -> None:
    """Test that getting a language without authentication returns 401."""
    response = client.get(f"{settings.API_V1_STR}/languages/1")

    assert response.status_code == 401
