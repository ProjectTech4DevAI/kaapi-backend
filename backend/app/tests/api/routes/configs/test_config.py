from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_config, create_test_project


def test_create_config_success(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test creating a config successfully with API key authentication."""
    config_data = {
        "name": "test-llm-config",
        "description": "A test LLM configuration",
        "config_blob": {
            "completion": {
                "provider": "openai-native",
                "params": {
                    "model": "gpt-4",
                    "temperature": 0.8,
                    "max_tokens": 2000,
                },
            }
        },
        "commit_message": "Initial configuration",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
        json=config_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["name"] == config_data["name"]
    assert data["data"]["description"] == config_data["description"]
    assert data["data"]["project_id"] == user_api_key.project_id
    assert "id" in data["data"]
    assert "version" in data["data"]
    assert data["data"]["version"]["version"] == 1
    assert data["data"]["version"]["config_blob"] == config_data["config_blob"]


def test_create_config_empty_blob_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that creating a config with empty config_blob fails validation."""
    config_data = {
        "name": "test-config",
        "description": "Test",
        "config_blob": {},
        "commit_message": "Initial",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
        json=config_data,
    )
    assert response.status_code == 422


def test_create_config_duplicate_name_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that creating a config with duplicate name in same project fails."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="duplicate-config",
    )

    # Try to create another with same name
    config_data = {
        "name": "duplicate-config",
        "description": "Should fail",
        "config_blob": {
            "completion": {
                "provider": "openai",
                "params": {"model": "gpt-4"},
            }
        },
        "commit_message": "Initial",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
        json=config_data,
    )
    assert response.status_code == 409
    response_data = response.json()
    error = response_data.get("error", response_data.get("detail", ""))
    assert "already exists" in error.lower()


def test_list_configs(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test listing configs for a project."""
    created_configs = []
    for i in range(3):
        config = create_test_config(
            db=db,
            project_id=user_api_key.project_id,
            name=f"list-test-config-{i}",
        )
        created_configs.append(config)

    response = client.get(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)
    assert len(data["data"]) >= 3

    config_names = [c["name"] for c in data["data"]]
    for config in created_configs:
        assert config.name in config_names


def test_list_configs_with_pagination(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test listing configs with pagination parameters."""
    for i in range(5):
        create_test_config(
            db=db,
            project_id=user_api_key.project_id,
            name=f"pagination-test-{i}",
        )

    # Test with limit
    response = client.get(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
        params={"skip": 0, "limit": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 2

    # Test with skip
    response = client.get(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
        params={"skip": 2, "limit": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) >= 2


def test_get_config_by_id(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test retrieving a specific config by ID."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="get-by-id-test",
        description="Test config for retrieval",
    )

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["id"] == str(config.id)
    assert data["data"]["name"] == config.name
    assert data["data"]["description"] == config.description
    assert data["data"]["project_id"] == user_api_key.project_id


def test_get_config_nonexistent(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test retrieving a non-existent config returns 404."""
    fake_uuid = uuid4()
    response = client.get(
        f"{settings.API_V1_STR}/configs/{fake_uuid}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_get_config_from_different_project_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that users cannot access configs from other projects."""
    other_project = create_test_project(db)
    config = create_test_config(
        db=db,
        project_id=other_project.id,
        name="other-project-config",
    )

    # Try to access it with user_api_key from different project
    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_update_config_name(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test updating a config's name."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="original-name",
    )

    update_data = {
        "name": "updated-name",
    }

    response = client.patch(
        f"{settings.API_V1_STR}/configs/{config.id}",
        headers={"X-API-KEY": user_api_key.key},
        json=update_data,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["name"] == "updated-name"
    assert data["data"]["id"] == str(config.id)


def test_update_config_description(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test updating a config's description."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
        description="Original description",
    )

    update_data = {
        "description": "Updated description",
    }

    response = client.patch(
        f"{settings.API_V1_STR}/configs/{config.id}",
        headers={"X-API-KEY": user_api_key.key},
        json=update_data,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["description"] == "Updated description"


def test_update_config_to_duplicate_name_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that updating a config to a duplicate name fails."""
    config1 = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="config-one",
    )
    config2 = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="config-two",
    )

    # Try to update config2 to have the same name as config1
    update_data = {
        "name": "config-one",
    }

    response = client.patch(
        f"{settings.API_V1_STR}/configs/{config2.id}",
        headers={"X-API-KEY": user_api_key.key},
        json=update_data,
    )
    assert response.status_code == 409
    response_data = response.json()
    error = response_data.get("error", response_data.get("detail", ""))
    assert "already exists" in error.lower()


def test_update_config_nonexistent(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test updating a non-existent config returns 404."""
    fake_uuid = uuid4()
    update_data = {
        "name": "new-name",
    }

    response = client.patch(
        f"{settings.API_V1_STR}/configs/{fake_uuid}",
        headers={"X-API-KEY": user_api_key.key},
        json=update_data,
    )
    assert response.status_code == 404


def test_delete_config(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test deleting a config (soft delete)."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="config-to-delete",
    )

    response = client.delete(
        f"{settings.API_V1_STR}/configs/{config.id}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "deleted successfully" in data["data"]["message"].lower()

    # Verify the config is no longer accessible
    get_response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert get_response.status_code == 404


def test_delete_config_nonexistent(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test deleting a non-existent config returns 404."""
    fake_uuid = uuid4()
    response = client.delete(
        f"{settings.API_V1_STR}/configs/{fake_uuid}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_delete_config_from_different_project_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that users cannot delete configs from other projects."""
    # Create config in different project
    other_project = create_test_project(db)
    config = create_test_config(
        db=db,
        project_id=other_project.id,
        name="other-project-config",
    )

    # Try to delete it with user_api_key from different project
    response = client.delete(
        f"{settings.API_V1_STR}/configs/{config.id}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_configs_isolated_by_project(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that configs are properly isolated between projects."""
    # Create configs in user's project
    user_configs = []
    for i in range(2):
        config = create_test_config(
            db=db,
            project_id=user_api_key.project_id,
            name=f"user-config-{i}",
        )
        user_configs.append(config)

    # Create configs in different project
    other_project = create_test_project(db)
    for i in range(3):
        create_test_config(
            db=db,
            project_id=other_project.id,
            name=f"other-config-{i}",
        )

    # User should only see their project's configs
    response = client.get(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify we only get configs from user's project
    for config_data in data["data"]:
        assert config_data["project_id"] == user_api_key.project_id
