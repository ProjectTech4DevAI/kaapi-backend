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
    """Test creating a new version with partial config update."""
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="test-config",
    )

    # Only send the fields we want to update (partial update)
    version_data = {
        "config_blob": {
            "completion": {
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
    assert data["data"]["commit_message"] == version_data["commit_message"]
    assert data["data"]["config_id"] == str(config.id)

    # Verify params were updated
    config_blob = data["data"]["config_blob"]
    assert config_blob["completion"]["params"]["model"] == "gpt-4-turbo"
    assert config_blob["completion"]["params"]["temperature"] == 0.9

    # Verify type was inherited from existing config
    assert config_blob["completion"]["type"] == "text"


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
                type="text",
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


def test_create_version_cannot_change_type_from_text_to_stt(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that config type cannot be changed from 'text' to 'stt' in a new version."""
    from app.models.llm.request import KaapiCompletionConfig, TextLLMParams

    # Create initial config with type='text'
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={"model": "gpt-4", "temperature": 0.7},
        )
    )
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="text-config",
        config_blob=config_blob,
    )

    # Try to create a new version with type='stt'
    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "type": "stt",
                "params": {
                    "model": "whisper-1",
                    "instructions": "Transcribe audio",
                    "temperature": 0.2,
                },
            }
        },
        "commit_message": "Attempting to change type to stt",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 400
    error_detail = response.json().get("error", "")
    assert "cannot change config type" in error_detail.lower()
    assert "text" in error_detail
    assert "stt" in error_detail


def test_create_version_cannot_change_type_from_stt_to_tts(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that config type cannot be changed from 'stt' to 'tts' in a new version."""
    from app.models.llm.request import KaapiCompletionConfig

    # Create initial config with type='stt'
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="stt",
            params={
                "model": "whisper-1",
                "instructions": "Transcribe audio",
                "temperature": 0.2,
            },
        )
    )
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="stt-config",
        config_blob=config_blob,
    )

    # Try to create a new version with type='tts'
    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "type": "tts",
                "params": {
                    "model": "tts-1",
                    "voice": "alloy",
                    "language": "en",
                },
            }
        },
        "commit_message": "Attempting to change type to tts",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 400


def test_create_version_cannot_change_type_from_tts_to_text(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that config type cannot be changed from 'tts' to 'text' in a new version."""
    from app.models.llm.request import KaapiCompletionConfig

    # Create initial config with type='tts'
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="tts",
            params={
                "model": "tts-1",
                "voice": "alloy",
                "language": "en",
            },
        )
    )
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="tts-config",
        config_blob=config_blob,
    )

    # Try to create a new version with type='text'
    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "type": "text",
                "params": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                },
            }
        },
        "commit_message": "Attempting to change type to text",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 400


def test_create_version_same_type_succeeds(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that creating a new version with the same type succeeds."""
    from app.models.llm.request import KaapiCompletionConfig

    # Create initial config with type='text'
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4",
                "temperature": 0.7,
            },
        )
    )
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="text-config",
        config_blob=config_blob,
    )

    # Create a new version with the same type='text'
    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "type": "text",
                "params": {
                    "model": "gpt-4-turbo",
                    "temperature": 0.9,
                },
            }
        },
        "commit_message": "Updated to gpt-4-turbo with same type",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["data"]["version"] == 2
    assert data["data"]["config_blob"]["completion"]["type"] == "text"


def test_create_version_partial_update_params_only(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test partial update - only updating params, inheriting provider and type."""
    from app.models.llm.request import KaapiCompletionConfig

    # Create initial config
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4",
                "temperature": 0.7,
            },
        )
    )
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="partial-update-test",
        config_blob=config_blob,
    )

    # Only send params update - provider and type will be inherited
    version_data = {
        "config_blob": {
            "completion": {
                "params": {
                    "model": "gpt-4-turbo",
                    "temperature": 0.9,
                },
            }
        },
        "commit_message": "Only updating model and temperature",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["data"]["version"] == 2

    config_blob_result = data["data"]["config_blob"]
    # Provider and type should be inherited
    assert config_blob_result["completion"]["provider"] == "openai"
    assert config_blob_result["completion"]["type"] == "text"
    # Params should be updated
    assert config_blob_result["completion"]["params"]["model"] == "gpt-4-turbo"
    assert config_blob_result["completion"]["params"]["temperature"] == 0.9


def test_create_config_with_kaapi_provider_success(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test creating a config with Kaapi provider (openai) works correctly."""
    config_data = {
        "name": "kaapi-text-config",
        "description": "A Kaapi configuration for text completion",
        "config_blob": {
            "completion": {
                "provider": "openai",
                "type": "text",
                "params": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                },
            }
        },
        "commit_message": "Initial Kaapi configuration",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/",
        headers={"X-API-KEY": user_api_key.key},
        json=config_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["data"]["name"] == config_data["name"]
    assert data["data"]["version"]["config_blob"]["completion"]["provider"] == "openai"
    assert data["data"]["version"]["config_blob"]["completion"]["type"] == "text"


def test_create_version_with_kaapi_stt_provider_success(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test creating STT config and version with Kaapi provider works correctly."""
    from app.models.llm.request import KaapiCompletionConfig

    # Create initial STT config with Kaapi provider
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="stt",
            params={
                "model": "whisper-1",
                "instructions": "Transcribe audio accurately",
                "temperature": 0.2,
            },
        )
    )
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="kaapi-stt-config",
        config_blob=config_blob,
    )

    # Create a new version with the same type='stt'
    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "type": "stt",
                "params": {
                    "model": "whisper-1",
                    "instructions": "Transcribe with high accuracy",
                    "temperature": 0.1,
                },
            }
        },
        "commit_message": "Updated STT instructions",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["data"]["version"] == 2
    assert data["data"]["config_blob"]["completion"]["provider"] == "openai"
    assert data["data"]["config_blob"]["completion"]["type"] == "stt"


def test_create_version_with_kaapi_tts_provider_success(
    db: Session,
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test creating TTS config and version with Kaapi provider works correctly."""
    from app.models.llm.request import KaapiCompletionConfig

    # Create initial TTS config with Kaapi provider
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="tts",
            params={
                "model": "tts-1",
                "voice": "alloy",
                "language": "en",
            },
        )
    )
    config = create_test_config(
        db=db,
        project_id=user_api_key.project_id,
        name="kaapi-tts-config",
        config_blob=config_blob,
    )

    # Create a new version with the same type='tts'
    version_data = {
        "config_blob": {
            "completion": {
                "provider": "openai",
                "type": "tts",
                "params": {
                    "model": "tts-1-hd",
                    "voice": "nova",
                    "language": "en",
                },
            }
        },
        "commit_message": "Updated TTS to HD model with nova voice",
    }

    response = client.post(
        f"{settings.API_V1_STR}/configs/{config.id}/versions",
        headers={"X-API-KEY": user_api_key.key},
        json=version_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["data"]["version"] == 2
    assert data["data"]["config_blob"]["completion"]["provider"] == "openai"
    assert data["data"]["config_blob"]["completion"]["type"] == "tts"
