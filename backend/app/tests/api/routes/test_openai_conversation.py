from sqlmodel import Session
from fastapi.testclient import TestClient

from app.crud.openai_conversation import create_conversation
from app.models import OpenAIConversationCreate
from app.tests.utils.openai import generate_openai_id
from app.tests.utils.auth import TestAuthContext


def test_get_conversation_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful conversation retrieval."""

    response_id = generate_openai_id("resp_", 40)
    conversation_data = OpenAIConversationCreate(
        response_id=response_id,
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of France?",
        response="The capital of France is Paris.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )
    response = client.get(
        f"/api/v1/openai-conversation/{conversation.id}",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["id"] == conversation.id
    assert response_data["data"]["response_id"] == conversation.response_id


def test_get_conversation_not_found(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation retrieval with non-existent ID."""
    response = client.get(
        "/api/v1/openai-conversation/99999",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 404
    response_data = response.json()
    assert "not found" in response_data["error"]


def test_get_conversation_by_response_id_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful conversation retrieval by response ID."""
    response_id = generate_openai_id("resp_", 40)
    conversation_data = OpenAIConversationCreate(
        response_id=response_id,
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of France?",
        response="The capital of France is Paris.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )

    response = client.get(
        f"/api/v1/openai-conversation/response/{response_id}",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["response_id"] == response_id
    assert response_data["data"]["id"] == conversation.id


def test_get_conversation_by_response_id_not_found(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation retrieval with non-existent response ID."""
    response = client.get(
        "/api/v1/openai-conversation/response/nonexistent_response_id",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 404
    response_data = response.json()
    assert "not found" in response_data["error"]


def test_get_conversation_by_ancestor_id_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful conversation retrieval by ancestor ID."""
    ancestor_response_id = generate_openai_id("resp_", 40)
    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=ancestor_response_id,
        previous_response_id=None,
        user_question="What is the capital of France?",
        response="The capital of France is Paris.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )

    response = client.get(
        f"/api/v1/openai-conversation/ancestor/{ancestor_response_id}",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["ancestor_response_id"] == ancestor_response_id
    assert response_data["data"]["id"] == conversation.id


def test_get_conversation_by_ancestor_id_not_found(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation retrieval with non-existent ancestor ID."""
    response = client.get(
        "/api/v1/openai-conversation/ancestor/nonexistent_ancestor_id",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 404
    response_data = response.json()
    assert "not found" in response_data["error"]


def test_list_conversations_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful conversation listing."""
    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of France?",
        response="The capital of France is Paris.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    # Actually create the conversation in the database
    create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )

    response = client.get(
        "/api/v1/openai-conversation",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert isinstance(response_data["data"], list)
    assert len(response_data["data"]) > 0


def test_list_conversations_with_pagination(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation listing with pagination."""
    # Create multiple conversations
    conversation_data_1 = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of Japan?",
        response="The capital of Japan is Tokyo.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation_data_2 = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of Brazil?",
        response="The capital of Brazil is Brasília.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    # Actually create the conversations in the database
    create_conversation(
        session=db,
        conversation=conversation_data_1,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )

    create_conversation(
        session=db,
        conversation=conversation_data_2,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )

    response = client.get(
        "/api/v1/openai-conversation?skip=1&limit=2",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert isinstance(response_data["data"], list)
    assert len(response_data["data"]) <= 2

    # Check pagination metadata
    assert "metadata" in response_data
    metadata = response_data["metadata"]
    assert metadata["skip"] == 1
    assert metadata["limit"] == 2
    assert "total" in metadata
    assert isinstance(metadata["total"], int)
    assert metadata["total"] >= 2


def test_list_conversations_pagination_metadata(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation listing pagination metadata."""
    # Create 5 conversations
    for i in range(5):
        conversation_data = OpenAIConversationCreate(
            response_id=generate_openai_id("resp_", 40),
            ancestor_response_id=generate_openai_id("resp_", 40),
            previous_response_id=None,
            user_question=f"Test question {i}",
            response=f"Test response {i}",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )

        create_conversation(
            session=db,
            conversation=conversation_data,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
        )

    # Test first page
    response = client.get(
        "/api/v1/openai-conversation?skip=0&limit=3",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True

    metadata = response_data["metadata"]
    assert metadata["skip"] == 0
    assert metadata["limit"] == 3
    assert (
        metadata["total"] >= 5
    )  # Should include the 5 we created plus any existing ones

    # Test second page
    response = client.get(
        "/api/v1/openai-conversation?skip=3&limit=3",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True

    metadata = response_data["metadata"]
    assert metadata["skip"] == 3
    assert metadata["limit"] == 3
    assert metadata["total"] >= 5


def test_list_conversations_default_pagination(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation listing with default pagination parameters."""
    # Create a conversation
    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="Test question",
        response="Test response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )

    # Test without pagination parameters (should use defaults)
    response = client.get(
        "/api/v1/openai-conversation",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True

    metadata = response_data["metadata"]
    assert metadata["skip"] == 0  # Default skip
    assert metadata["limit"] == 100  # Default limit
    assert "total" in metadata
    assert isinstance(metadata["total"], int)


def test_list_conversations_edge_cases(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation listing edge cases for pagination."""
    # Test with skip larger than total
    response = client.get(
        "/api/v1/openai-conversation?skip=1000&limit=10",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert len(response_data["data"]) == 0  # Should return empty list

    metadata = response_data["metadata"]
    assert metadata["skip"] == 1000
    assert metadata["limit"] == 10
    assert "total" in metadata

    # Test with maximum limit
    response = client.get(
        "/api/v1/openai-conversation?skip=0&limit=100",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True

    metadata = response_data["metadata"]
    assert metadata["skip"] == 0
    assert metadata["limit"] == 100
    assert "total" in metadata


def test_list_conversations_invalid_pagination(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation listing with invalid pagination parameters."""
    response = client.get(
        "/api/v1/openai-conversation?skip=-1&limit=0",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 422


def test_delete_conversation_success(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
) -> None:
    """Test successful conversation deletion."""
    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of Japan?",
        response="The capital of Japan is Tokyo.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    # Create the conversation in the database and get the created object with ID
    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )
    conversation_id = conversation.id

    response = client.delete(
        f"/api/v1/openai-conversation/{conversation_id}",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert "deleted successfully" in response_data["data"]["message"]

    # Verify the conversation is marked as deleted
    response = client.get(
        f"/api/v1/openai-conversation/{conversation_id}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_delete_conversation_not_found(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test conversation deletion with non-existent ID."""
    response = client.delete(
        "/api/v1/openai-conversation/99999",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 404
    response_data = response.json()
    assert "not found" in response_data["error"]
