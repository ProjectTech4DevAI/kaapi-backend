from unittest.mock import patch
from typing import Any
from uuid import uuid4

import pytest
from sqlmodel import Session
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.tests.utils.llm_provider import mock_openai_assistant
from app.tests.utils.utils import get_assistant
from app.tests.utils.auth import TestAuthContext


@pytest.fixture
def assistant_create_payload() -> dict[str, Any]:
    return {
        "name": "Test Assistant",
        "instructions": "This is a test instruction.",
        "model": "gpt-4o",
        "vector_store_ids": ["vs_test_1", "vs_test_2"],
        "temperature": 0.5,
        "max_num_results": 10,
    }


@pytest.fixture
def assistant_id() -> str:
    return str(uuid4())


@patch("app.api.routes.assistants.fetch_assistant_from_openai")
def test_ingest_assistant_success(
    mock_fetch_assistant: Any,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful assistant ingestion from OpenAI."""
    mock_assistant = mock_openai_assistant()

    mock_fetch_assistant.return_value = mock_assistant

    response = client.post(
        f"/api/v1/assistant/{mock_assistant.id}/ingest",
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 201
    response_json = response.json()
    assert response_json["success"] is True
    assert response_json["data"]["assistant_id"] == mock_assistant.id


@patch("app.crud.assistants.verify_vector_store_ids_exist")
def test_create_assistant_success(
    mock_verify_vector_ids: Any,
    client: TestClient,
    assistant_create_payload: dict[str, Any],
    user_api_key: TestAuthContext,
) -> None:
    """Test successful assistant creation with OpenAI vector store ID verification."""

    mock_verify_vector_ids.return_value = None

    response = client.post(
        "/api/v1/assistant",
        json=assistant_create_payload,
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["name"] == assistant_create_payload["name"]
    assert (
        response_data["data"]["instructions"]
        == assistant_create_payload["instructions"]
    )
    assert response_data["data"]["model"] == assistant_create_payload["model"]
    assert (
        response_data["data"]["vector_store_ids"]
        == assistant_create_payload["vector_store_ids"]
    )
    assert (
        response_data["data"]["temperature"] == assistant_create_payload["temperature"]
    )
    assert (
        response_data["data"]["max_num_results"]
        == assistant_create_payload["max_num_results"]
    )


@patch("app.crud.assistants.verify_vector_store_ids_exist")
def test_create_assistant_invalid_vector_store(
    mock_verify_vector_ids: Any,
    client: TestClient,
    assistant_create_payload: dict[str, Any],
    user_api_key: TestAuthContext,
) -> None:
    """Test failure when one or more vector store IDs are invalid."""

    mock_verify_vector_ids.side_effect = HTTPException(
        status_code=400, detail="Vector store ID vs_test_999 not found in OpenAI."
    )

    payload = assistant_create_payload.copy()
    payload["vector_store_ids"] = ["vs_test_999"]

    response = client.post(
        "/api/v1/assistant",
        json=payload,
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 400
    response_data = response.json()
    assert response_data["error"] == "Vector store ID vs_test_999 not found in OpenAI."


def test_update_assistant_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful assistant update."""
    update_payload = {
        "name": "Updated Assistant",
        "instructions": "Updated instructions.",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_num_results": 5,
    }

    assistant = get_assistant(db, project_id=user_api_key.project_id)

    response = client.patch(
        f"/api/v1/assistant/{assistant.assistant_id}",
        json=update_payload,
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["name"] == update_payload["name"]
    assert response_data["data"]["instructions"] == update_payload["instructions"]
    assert response_data["data"]["model"] == update_payload["model"]
    assert response_data["data"]["temperature"] == update_payload["temperature"]
    assert response_data["data"]["max_num_results"] == update_payload["max_num_results"]


@patch("app.crud.assistants.verify_vector_store_ids_exist")
def test_update_assistant_invalid_vector_store(
    mock_verify_vector_ids: Any,
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test failure when updating assistant with invalid vector store IDs."""
    mock_verify_vector_ids.side_effect = HTTPException(
        status_code=400, detail="Vector store ID vs_invalid not found in OpenAI."
    )

    update_payload = {"vector_store_ids_add": ["vs_invalid"]}

    assistant = get_assistant(db, project_id=user_api_key.project_id)

    response = client.patch(
        f"/api/v1/assistant/{assistant.assistant_id}",
        json=update_payload,
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 400
    response_data = response.json()
    assert response_data["error"] == "Vector store ID vs_invalid not found in OpenAI."


def test_update_assistant_not_found(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test failure when updating a non-existent assistant."""
    update_payload = {"name": "Updated Assistant"}

    non_existent_id = str(uuid4())

    response = client.patch(
        f"/api/v1/assistant/{non_existent_id}",
        json=update_payload,
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 404
    response_data = response.json()
    assert "not found" in response_data["error"].lower()


def test_get_assistant_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful retrieval of a single assistant."""
    assistant = get_assistant(db, project_id=user_api_key.project_id)

    response = client.get(
        f"/api/v1/assistant/{assistant.assistant_id}",
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["assistant_id"] == assistant.assistant_id
    assert response_data["data"]["name"] == assistant.name
    assert response_data["data"]["instructions"] == assistant.instructions
    assert response_data["data"]["model"] == assistant.model


def test_get_assistant_not_found(
    client: TestClient,
    user_api_key_header: dict[str, Any],
) -> None:
    """Test failure when fetching a non-existent assistant."""
    non_existent_id = str(uuid4())

    response = client.get(
        f"/api/v1/assistant/{non_existent_id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 404
    response_data = response.json()
    assert f"Assistant with ID {non_existent_id} not found." in response_data["error"]


def test_list_assistants_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful retrieval of assistants list."""
    assistant = get_assistant(db, project_id=user_api_key.project_id)

    response = client.get(
        "/api/v1/assistant/",
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert isinstance(response_data["data"], list)
    assert len(response_data["data"]) >= 1

    # Check that our test assistant is in the list
    assistant_ids = [a["assistant_id"] for a in response_data["data"]]
    assert assistant.assistant_id in assistant_ids


def test_list_assistants_invalid_pagination(
    client: TestClient,
    user_api_key_header: dict[str, Any],
) -> None:
    """Test assistants list with invalid pagination parameters."""
    # Test negative skip
    response = client.get(
        "/api/v1/assistant/?skip=-1&limit=10",
        headers=user_api_key_header,
    )
    assert response.status_code == 422

    # Test limit too high
    response = client.get(
        "/api/v1/assistant/?skip=0&limit=101",
        headers=user_api_key_header,
    )
    assert response.status_code == 422

    # Test limit too low
    response = client.get(
        "/api/v1/assistant/?skip=0&limit=0",
        headers=user_api_key_header,
    )
    assert response.status_code == 422


def test_delete_assistant_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful soft deletion of an assistant."""
    assistant = get_assistant(db, project_id=user_api_key.project_id)

    response = client.delete(
        f"/api/v1/assistant/{assistant.assistant_id}",
        headers={"X-API-KEY": f"{user_api_key.key}"},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["message"] == "Assistant deleted successfully."


def test_delete_assistant_not_found(
    client: TestClient,
    user_api_key_header: dict[str, Any],
) -> None:
    """Test failure when deleting a non-existent assistant."""
    non_existent_id = str(uuid4())

    response = client.delete(
        f"/api/v1/assistant/{non_existent_id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 404
    response_data = response.json()
    assert "not found" in response_data["error"].lower()
