from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_project,
    create_test_version,
)
from app.models import ConfigBlob
from app.models.llm.request import NativeCompletionConfig


def test_create_version_success(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test creating a new version for a config successfully."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai-native",
                "params": {
                    "model": "gpt-4-turbo",
                    "temperature": 0.9,
                    "max_tokens": 3000,
                },
            }
        },
        "commit_message": "Updated model to gpt-4-turbo",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert (
        data["data"]["version"] == 2
    )  # First version created with config, this is second
    assert data["data"]["config_blob"] == version_data["config_blob"]
    assert data["data"]["commit_message"] == version_data["commit_message"]
    assert data["data"]["config_id"] == str(config.id)


def test_create_version_empty_blob_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that creating a version with empty config_blob fails validation."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    version_data = {
        "config_blob": {},
        "commit_message": "Empty blob",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 422


def test_create_version_nonexistent_config(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test creating a version for a non-existent config returns 404."""
    fake_uuid = uuid4()
    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "params": {"model": "gpt-4"},
            }
        },
        "commit_message": "Test",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{fake_uuid}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 404


def test_create_version_different_project_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that creating a version for a config in a different project fails."""
    other_project = create_test_project(db)
    config = create_test_config(
        db=db,
        project_id=other_project.id,
        name="other-project-config",
    )

    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "params": {"model": "gpt-4"},
            }
        },
        "commit_message": "Should fail",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 404


def test_create_version_auto_increments(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that version numbers are automatically incremented."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    # Create multiple versions and verify they increment
    for i in range(2, 5):
        version_data = {
            "config_blob": {
                "completion": {
                    "provider": "openai",
                    "params": {"model": f"gpt-4-version-{i}"},
                }
            },
            "commit_message": f"Version {i}",
        }

        response = client.post(
            f"{settings.API_V1_STR}/configs/{config.id}/versions",
            headers={"X-API-KEY": user_api_key.key},
            json=version_data,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["version"] == i


def test_list_versions(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test listing all versions for a config."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    # Create additional versions
    for i in range(3):
        create_test_version(
            db=db,
            config_id=config.id,
            project_id=user_api_key.project_id,
            commit_message=f"Version {i + 2}",
        )

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 4  # 1 initial + 3 created

    # Verify versions are ordered by version number descending
    versions = data["data"]
    for i in range(len(versions) - 1):
        assert versions[i]["version"] > versions[i + 1]["version"]


def test_list_versions_with_pagination(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test listing versions with pagination parameters."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    for i in range(5):
        create_test_version(
            db=db,
            config_id=config.id,
            project_id=user_api_key.project_id,
        )

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        params={"skip": 0, "limit": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 3

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        params={"skip": 3, "limit": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 3


def test_list_versions_nonexistent_config(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test listing versions for a non-existent config returns 404."""
    fake_uuid = uuid4()

    response = client.get(
        f"{settings.API_V1_STR}/configs/{fake_uuid}/versions",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_list_versions_different_project_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that listing versions for a config in a different project fails."""
    other_project = create_test_project(db)
    config = create_test_config(
        db=db,
        project_id=other_project.id,
        name="other-project-config",
    )

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_get_version_by_number(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test retrieving a specific version by version number."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    version = create_test_version(
        db=db,
        config_id=config.id,
        project_id=user_api_key.project_id,
        config_blob=ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-4-turbo", "temperature": 0.5},
            )
        ),
        commit_message="Updated config",
    )

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions/{version.version}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["version"] == version.version
    assert data["data"]["config_blob"] == version.config_blob
    assert data["data"]["commit_message"] == version.commit_message
    assert data["data"]["config_id"] == str(config.id)


def test_get_version_nonexistent(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test retrieving a non-existent version returns 404."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions/999",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_get_version_from_different_project_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that users cannot access versions from configs in other projects."""
    other_project = create_test_project(db)
    config = create_test_config(
        db=db,
        project_id=other_project.id,
        name="other-project-config",
    )

    response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions/1",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_delete_version(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test deleting a version (soft delete)."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    version = create_test_version(
        db=db,
        config_id=config.id,
        project_id=user_api_key.project_id,
    )

    response = client.delete(
        f"{settings.API_V1_STR}/configs/{config.id}/versions/{version.version}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "deleted successfully" in data["data"]["message"].lower()

    get_response = client.get(
        f"{settings.API_V1_STR}/configs/{config.id}/versions/{version.version}",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert get_response.status_code == 404


def test_delete_version_nonexistent(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test deleting a non-existent version returns 404."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    response = client.delete(
        f"{settings.API_V1_STR}/configs/{config.id}/versions/999",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_delete_version_from_different_project_fails(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that users cannot delete versions from configs in other projects."""
    other_project = create_test_project(db)
    config = create_test_config(
        db=db,
        project_id=other_project.id,
        name="other-project-config",
    )

    # Try to delete the initial version
    response = client.delete(
        f"{settings.API_V1_STR}/configs/{config.id}/versions/1",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404


def test_versions_isolated_by_project(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that versions are properly isolated between projects."""
    # Create config in user's project with additional versions
    user_config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="user-config",
    )
    for i in range(2):
        create_test_version(
            db=db,
            config_id=user_config.id,
            project_id=user_api_key.project_id,
        )

    # Create config in different project with versions
    other_project = create_test_project(db)
    other_config = create_test_config(
        db=db,
        project_id=other_project.id,
        name="other-config",
    )
    for i in range(3):
        create_test_version(
            db=db,
            config_id=other_config.id,
            project_id=other_project.id,
        )

    # User should only see versions from their project's config
    response = client.get(
        f"{settings.API_V1_STR}/configs/{user_config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 3  # 1 initial + 2 created

    # User should NOT be able to access other project's versions
    response = client.get(
        f"{settings.API_V1_STR}/configs/{other_config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
    )
    assert response.status_code == 404
