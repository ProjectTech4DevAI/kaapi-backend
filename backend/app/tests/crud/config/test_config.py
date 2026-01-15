from uuid import uuid4

import pytest
from sqlmodel import Session
from fastapi import HTTPException

from app.models import (
    ConfigBlob,
    CompletionConfig,
    ConfigCreate,
    ConfigUpdate,
)
from app.models.llm.request import NativeCompletionConfig
from app.crud.config import ConfigCrud
from app.tests.utils.test_data import create_test_project, create_test_config
from app.tests.utils.utils import random_lower_string


@pytest.fixture
def example_config_blob():
    return ConfigBlob(
        completion=NativeCompletionConfig(
            provider="openai-native",
            params={
                "model": "gpt-4",
                "temperature": 0.8,
                "max_tokens": 1500,
            },
        )
    )


def test_create_config(db: Session, example_config_blob: ConfigBlob) -> None:
    """Test creating a new configuration with initial version."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    config_name = f"test-config-{random_lower_string()}"
    config_create = ConfigCreate(
        name=config_name,
        description="Test configuration",
        config_blob=example_config_blob,
        commit_message="Initial version",
    )

    config, version = config_crud.create_or_raise(config_create)

    assert config.id is not None
    assert config.name == config_name
    assert config.description == "Test configuration"
    assert config.project_id == project.id
    assert config.deleted_at is None

    # Verify initial version was created
    assert version.id is not None
    assert version.config_id == config.id
    assert version.version == 1
    assert version.config_blob == example_config_blob.model_dump()
    assert version.commit_message == "Initial version"


def test_create_config_duplicate_name(
    db: Session, example_config_blob: ConfigBlob
) -> None:
    """Test creating a configuration with a duplicate name raises HTTPException."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    config_name = f"test-config-{random_lower_string()}"
    config_create = ConfigCreate(
        name=config_name,
        description="Test configuration",
        config_blob=example_config_blob,
        commit_message="Initial version",
    )

    # Create first config
    config_crud.create_or_raise(config_create)

    # Attempt to create second config with same name
    with pytest.raises(
        HTTPException, match=f"Config with name '{config_name}' already exists"
    ):
        config_crud.create_or_raise(config_create)


def test_create_config_different_projects_same_name(
    db: Session, example_config_blob: ConfigBlob
) -> None:
    """Test creating configs with same name in different projects succeeds."""
    project1 = create_test_project(db)
    project2 = create_test_project(db)

    config_name = f"test-config-{random_lower_string()}"
    config_blob = example_config_blob

    # Create config in project1
    config_crud1 = ConfigCrud(session=db, project_id=project1.id)
    config_create = ConfigCreate(
        name=config_name,
        description="Test configuration",
        config_blob=config_blob,
        commit_message="Initial version",
    )
    config1, _ = config_crud1.create_or_raise(config_create)

    # Create config with same name in project2
    config_crud2 = ConfigCrud(session=db, project_id=project2.id)
    config2, _ = config_crud2.create_or_raise(config_create)

    assert config1.id != config2.id
    assert config1.name == config2.name == config_name
    assert config1.project_id == project1.id
    assert config2.project_id == project2.id


def test_read_one_config(db: Session) -> None:
    """Test reading a single configuration by ID."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    fetched_config = config_crud.read_one(config.id)

    assert fetched_config is not None
    assert fetched_config.id == config.id
    assert fetched_config.name == config.name
    assert fetched_config.project_id == config.project_id


def test_read_one_config_not_found(db: Session) -> None:
    """Test reading a non-existent configuration returns None."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    non_existent_id = uuid4()
    fetched_config = config_crud.read_one(non_existent_id)

    assert fetched_config is None


def test_read_one_config_different_project(db: Session) -> None:
    """Test reading a config from a different project returns None."""
    # Create config in project1
    config = create_test_config(db)

    # Try to read from project2
    project2 = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project2.id)

    fetched_config = config_crud.read_one(config.id)

    assert fetched_config is None


def test_read_one_deleted_config(db: Session) -> None:
    """Test reading a deleted configuration returns None."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    # Delete the config
    config_crud.delete_or_raise(config.id)

    # Try to read deleted config
    fetched_config = config_crud.read_one(config.id)

    assert fetched_config is None


def test_read_all_configs(db: Session) -> None:
    """Test reading all configurations for a project."""
    project = create_test_project(db)

    # Create multiple configs
    config1 = create_test_config(db, project_id=project.id, name="config-1")
    config2 = create_test_config(db, project_id=project.id, name="config-2")
    config3 = create_test_config(db, project_id=project.id, name="config-3")

    config_crud = ConfigCrud(session=db, project_id=project.id)
    configs = config_crud.read_all()

    config_ids = [c.id for c in configs]
    assert config1.id in config_ids
    assert config2.id in config_ids
    assert config3.id in config_ids


def test_read_all_configs_pagination(db: Session) -> None:
    """Test reading configurations with pagination."""
    project = create_test_project(db)

    # Create 5 configs
    for i in range(5):
        create_test_config(db, project_id=project.id, name=f"config-{i}")

    config_crud = ConfigCrud(session=db, project_id=project.id)

    # Test skip and limit
    configs_page1 = config_crud.read_all(skip=0, limit=2)
    configs_page2 = config_crud.read_all(skip=2, limit=2)

    assert len(configs_page1) == 2
    assert len(configs_page2) == 2
    assert configs_page1[0].id != configs_page2[0].id


def test_read_all_configs_ordered_by_updated_at(db: Session) -> None:
    """Test that configurations are ordered by updated_at in descending order."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    # Create configs (they will have different updated_at timestamps)
    config1 = create_test_config(db, project_id=project.id, name="config-1")
    config2 = create_test_config(db, project_id=project.id, name="config-2")
    config3 = create_test_config(db, project_id=project.id, name="config-3")

    # Update config1 to make it the most recently updated
    config_crud.update_or_raise(
        config1.id, ConfigUpdate(description="Updated description")
    )

    configs = config_crud.read_all()

    # config1 should be first because it was most recently updated
    assert configs[0].id == config1.id


def test_read_all_configs_excludes_deleted(db: Session) -> None:
    """Test that read_all excludes deleted configurations."""
    project = create_test_project(db)

    config1 = create_test_config(db, project_id=project.id, name="config-1")
    config2 = create_test_config(db, project_id=project.id, name="config-2")

    config_crud = ConfigCrud(session=db, project_id=project.id)

    # Delete config1
    config_crud.delete_or_raise(config1.id)

    configs = config_crud.read_all()

    config_ids = [c.id for c in configs]
    assert config1.id not in config_ids
    assert config2.id in config_ids


def test_read_all_configs_different_projects(db: Session) -> None:
    """Test that read_all only returns configs for the specific project."""
    project1 = create_test_project(db)
    project2 = create_test_project(db)

    config1 = create_test_config(db, project_id=project1.id, name="config-1")
    config2 = create_test_config(db, project_id=project2.id, name="config-2")

    config_crud = ConfigCrud(session=db, project_id=project1.id)
    configs = config_crud.read_all()

    config_ids = [c.id for c in configs]
    assert config1.id in config_ids
    assert config2.id not in config_ids


def test_update_config_name(db: Session) -> None:
    """Test updating a configuration's name."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    new_name = f"updated-config-{random_lower_string()}"
    config_update = ConfigUpdate(name=new_name)

    updated_config = config_crud.update_or_raise(config.id, config_update)

    assert updated_config.name == new_name
    assert updated_config.id == config.id


def test_update_config_description(db: Session) -> None:
    """Test updating a configuration's description."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    new_description = "Updated description"
    config_update = ConfigUpdate(description=new_description)

    updated_config = config_crud.update_or_raise(config.id, config_update)

    assert updated_config.description == new_description
    assert updated_config.id == config.id


def test_update_config_multiple_fields(db: Session) -> None:
    """Test updating multiple fields of a configuration."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    new_name = f"updated-config-{random_lower_string()}"
    new_description = "Updated description"
    config_update = ConfigUpdate(name=new_name, description=new_description)

    updated_config = config_crud.update_or_raise(config.id, config_update)

    assert updated_config.name == new_name
    assert updated_config.description == new_description


def test_update_config_duplicate_name(db: Session) -> None:
    """Test updating a config to a duplicate name raises HTTPException."""
    project = create_test_project(db)

    config1 = create_test_config(db, project_id=project.id, name="config-1")
    config2 = create_test_config(db, project_id=project.id, name="config-2")

    config_crud = ConfigCrud(session=db, project_id=project.id)

    # Try to update config2 to have config1's name
    config_update = ConfigUpdate(name=config1.name)

    with pytest.raises(
        HTTPException, match=f"Config with name '{config1.name}' already exists"
    ):
        config_crud.update_or_raise(config2.id, config_update)


def test_update_config_same_name(db: Session) -> None:
    """Test updating a config to its own name succeeds."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    # Update to same name should succeed
    config_update = ConfigUpdate(name=config.name, description="Updated")

    updated_config = config_crud.update_or_raise(config.id, config_update)

    assert updated_config.name == config.name
    assert updated_config.description == "Updated"


def test_update_config_not_found(db: Session) -> None:
    """Test updating a non-existent configuration raises HTTPException."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    non_existent_id = uuid4()
    config_update = ConfigUpdate(name="new-name")

    with pytest.raises(
        HTTPException, match=f"config with id '{non_existent_id}' not found"
    ):
        config_crud.update_or_raise(non_existent_id, config_update)


def test_update_config_updates_timestamp(db: Session) -> None:
    """Test that updating a config updates the updated_at timestamp."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    original_updated_at = config.updated_at

    config_update = ConfigUpdate(description="Updated description")
    updated_config = config_crud.update_or_raise(config.id, config_update)

    assert updated_config.updated_at > original_updated_at


def test_delete_config(db: Session) -> None:
    """Test soft deleting a configuration."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    config_crud.delete_or_raise(config.id)

    # Verify soft delete (deleted_at is set)
    db.refresh(config)
    assert config.deleted_at is not None


def test_delete_config_not_found(db: Session) -> None:
    """Test deleting a non-existent configuration raises HTTPException."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    non_existent_id = uuid4()

    with pytest.raises(
        HTTPException, match=f"config with id '{non_existent_id}' not found"
    ):
        config_crud.delete_or_raise(non_existent_id)


def test_delete_config_different_project(db: Session) -> None:
    """Test deleting a config from a different project raises HTTPException."""
    # Create config in project1
    config = create_test_config(db)

    # Try to delete from project2
    project2 = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project2.id)

    with pytest.raises(HTTPException, match=f"config with id '{config.id}' not found"):
        config_crud.delete_or_raise(config.id)


def test_exists_config(db: Session) -> None:
    """Test that exists returns the config when it exists."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    existing_config = config_crud.exists_or_raise(config.id)

    assert existing_config.id == config.id
    assert existing_config.name == config.name


def test_exists_config_not_found(db: Session) -> None:
    """Test that exists raises HTTPException when config doesn't exist."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    non_existent_id = uuid4()

    with pytest.raises(
        HTTPException, match=f"config with id '{non_existent_id}' not found"
    ):
        config_crud.exists_or_raise(non_existent_id)


def test_exists_deleted_config(db: Session) -> None:
    """Test that exists raises HTTPException for deleted configs."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    # Delete the config
    config_crud.delete_or_raise(config.id)

    # exists should raise HTTPException
    with pytest.raises(HTTPException, match=f"config with id '{config.id}' not found"):
        config_crud.exists_or_raise(config.id)


def test_check_unique_name_with_existing_name(db: Session) -> None:
    """Test that _check_unique_name raises HTTPException for duplicate names."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    with pytest.raises(
        HTTPException, match=f"Config with name '{config.name}' already exists"
    ):
        config_crud._check_unique_name_or_raise(config.name)


def test_check_unique_name_with_new_name(db: Session) -> None:
    """Test that _check_unique_name passes for unique names."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    # Should not raise exception
    unique_name = f"unique-name-{random_lower_string()}"
    config_crud._check_unique_name_or_raise(unique_name)


def test_read_by_name(db: Session) -> None:
    """Test reading a configuration by name."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    fetched_config = config_crud._read_by_name(config.name)

    assert fetched_config is not None
    assert fetched_config.id == config.id
    assert fetched_config.name == config.name


def test_read_by_name_not_found(db: Session) -> None:
    """Test that _read_by_name returns None for non-existent names."""
    project = create_test_project(db)
    config_crud = ConfigCrud(session=db, project_id=project.id)

    non_existent_name = f"non-existent-{random_lower_string()}"
    fetched_config = config_crud._read_by_name(non_existent_name)

    assert fetched_config is None


def test_read_by_name_deleted_config(db: Session) -> None:
    """Test that _read_by_name returns None for deleted configs."""
    config = create_test_config(db)
    config_crud = ConfigCrud(session=db, project_id=config.project_id)

    # Delete the config
    config_crud.delete_or_raise(config.id)

    # Should return None
    fetched_config = config_crud._read_by_name(config.name)

    assert fetched_config is None
