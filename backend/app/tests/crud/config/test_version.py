from uuid import uuid4

import pytest
from sqlmodel import Session
from fastapi import HTTPException

from app.models import ConfigVersionCreate, ConfigBlob
from app.models.llm.request import NativeCompletionConfig
from app.crud.config import ConfigVersionCrud
from app.tests.utils.test_data import (
    create_test_project,
    create_test_config,
    create_test_version,
)


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


def test_create_version(db: Session, example_config_blob: ConfigBlob) -> None:
    """Test creating a new version for an existing configuration."""
    config = create_test_config(db)
    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    config_blob = example_config_blob.model_dump()
    version_create = ConfigVersionCreate(
        config_blob=config_blob,
        commit_message="Updated model and parameters",
    )

    version = version_crud.create_or_raise(version_create)

    assert version.id is not None
    assert version.config_id == config.id
    assert version.version == 2  # Should be 2 since config creation creates version 1
    assert version.config_blob == config_blob
    assert version.commit_message == "Updated model and parameters"
    assert version.deleted_at is None


def test_create_version_auto_increment(
    db: Session, example_config_blob: ConfigBlob
) -> None:
    """Test that version numbers auto-increment correctly."""
    config = create_test_config(db)
    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    # Create multiple versions
    version2 = version_crud.create_or_raise(
        ConfigVersionCreate(config_blob=example_config_blob, commit_message="Version 2")
    )
    version3 = version_crud.create_or_raise(
        ConfigVersionCreate(config_blob=example_config_blob, commit_message="Version 3")
    )
    version4 = version_crud.create_or_raise(
        ConfigVersionCreate(config_blob=example_config_blob, commit_message="Version 4")
    )

    assert version2.version == 2
    assert version3.version == 3
    assert version4.version == 4


def test_create_version_config_not_found(
    db: Session, example_config_blob: ConfigBlob
) -> None:
    """Test creating a version for a non-existent config raises HTTPException."""
    project = create_test_project(db)
    non_existent_config_id = uuid4()

    version_crud = ConfigVersionCrud(
        session=db, project_id=project.id, config_id=non_existent_config_id
    )

    version_create = ConfigVersionCreate(
        config_blob=example_config_blob, commit_message="Test"
    )

    with pytest.raises(
        HTTPException, match=f"config with id '{non_existent_config_id}' not found"
    ):
        version_crud.create_or_raise(version_create)


def test_read_one_version(db: Session, example_config_blob: ConfigBlob) -> None:
    """Test reading a specific version by its version number."""
    config = create_test_config(db)
    version = create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        config_blob=example_config_blob,
        commit_message="Test version",
    )

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    fetched_version = version_crud.read_one(version.version)

    assert fetched_version is not None
    assert fetched_version.id == version.id
    assert fetched_version.version == version.version
    assert fetched_version.config_id == config.id
    assert fetched_version.config_blob == example_config_blob.model_dump()


def test_read_one_version_not_found(db: Session) -> None:
    """Test reading a non-existent version returns None."""
    config = create_test_config(db)
    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    non_existent_version = 999
    fetched_version = version_crud.read_one(non_existent_version)

    assert fetched_version is None


def test_read_one_version_deleted(db: Session) -> None:
    """Test reading a deleted version returns None."""
    config = create_test_config(db)
    version = create_test_version(db, config_id=config.id, project_id=config.project_id)

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    # Delete the version
    version_crud.delete_or_raise(version.version)

    # Try to read deleted version
    fetched_version = version_crud.read_one(version.version)

    assert fetched_version is None


def test_read_all_versions(db: Session) -> None:
    """Test reading all versions for a configuration."""
    config = create_test_config(db)

    # Create additional versions (config already has version 1)
    version2 = create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        commit_message="Version 2",
    )
    version3 = create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        commit_message="Version 3",
    )

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )
    versions = version_crud.read_all()

    assert len(versions) == 3
    version_numbers = [v.version for v in versions]
    assert 1 in version_numbers
    assert version2.version in version_numbers
    assert version3.version in version_numbers


def test_read_all_versions_pagination(db: Session) -> None:
    """Test reading versions with pagination."""
    config = create_test_config(db)

    # Create 4 additional versions
    for i in range(4):
        create_test_version(
            db,
            config_id=config.id,
            project_id=config.project_id,
            commit_message=f"Version {i + 2}",
        )

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    # Test skip and limit
    versions_page1 = version_crud.read_all(skip=0, limit=2)
    versions_page2 = version_crud.read_all(skip=2, limit=2)

    assert len(versions_page1) == 2
    assert len(versions_page2) == 2
    assert versions_page1[0].id != versions_page2[0].id


def test_read_all_versions_ordered_by_version_desc(db: Session) -> None:
    """Test that versions are ordered by version number in descending order."""
    config = create_test_config(db)

    # Create additional versions
    version2 = create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        commit_message="Version 2",
    )
    version3 = create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        commit_message="Version 3",
    )

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )
    versions = version_crud.read_all()

    # Versions should be in descending order (3, 2, 1)
    assert versions[0].version == version3.version
    assert versions[1].version == version2.version
    assert versions[2].version == 1


def test_read_all_versions_excludes_blob(db: Session) -> None:
    """Test that read_all returns ConfigVersionItems without config_blob."""
    config = create_test_config(db)
    create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
    )

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )
    versions = version_crud.read_all()

    # Verify versions are ConfigVersionItems (should not have config_blob field)
    for version in versions:
        assert hasattr(version, "id")
        assert hasattr(version, "version")
        assert hasattr(version, "commit_message")
        # ConfigVersionItems should not include config_blob
        assert not hasattr(version, "config_blob")


def test_read_all_versions_excludes_deleted(db: Session) -> None:
    """Test that read_all excludes deleted versions."""
    config = create_test_config(db)

    version2 = create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        commit_message="Version 2",
    )
    version3 = create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        commit_message="Version 3",
    )

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    # Delete version 2
    version_crud.delete_or_raise(version2.version)

    versions = version_crud.read_all()

    version_numbers = [v.version for v in versions]
    assert version2.version not in version_numbers
    assert 1 in version_numbers
    assert version3.version in version_numbers


def test_delete_version(db: Session) -> None:
    """Test soft deleting a version."""
    config = create_test_config(db)
    version = create_test_version(db, config_id=config.id, project_id=config.project_id)

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    version_crud.delete_or_raise(version.version)

    # Verify soft delete (deleted_at is set)
    db.refresh(version)
    assert version.deleted_at is not None


def test_delete_version_not_found(db: Session) -> None:
    """Test deleting a non-existent version raises HTTPException."""
    config = create_test_config(db)
    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    non_existent_version = 999

    with pytest.raises(
        HTTPException,
        match=f"Version with number '{non_existent_version}' not found for config '{config.id}'",
    ):
        version_crud.delete_or_raise(non_existent_version)


def test_exists_version(db: Session) -> None:
    """Test that exists returns the version when it exists."""
    config = create_test_config(db)
    version = create_test_version(db, config_id=config.id, project_id=config.project_id)

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    existing_version = version_crud.exists_or_raise(version.version)

    assert existing_version.id == version.id
    assert existing_version.version == version.version


def test_exists_version_not_found(db: Session) -> None:
    """Test that exists raises HTTPException when version doesn't exist."""
    config = create_test_config(db)
    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    non_existent_version = 999

    with pytest.raises(
        HTTPException,
        match=f"Version with number '{non_existent_version}' not found for config '{config.id}'",
    ):
        version_crud.exists_or_raise(non_existent_version)


def test_exists_version_deleted(db: Session) -> None:
    """Test that exists raises HTTPException for deleted versions."""
    config = create_test_config(db)
    version = create_test_version(db, config_id=config.id, project_id=config.project_id)

    version_crud = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    )

    # Delete the version
    version_crud.delete_or_raise(version.version)

    # exists should raise HTTPException
    with pytest.raises(
        HTTPException,
        match=f"Version with number '{version.version}' not found for config '{config.id}'",
    ):
        version_crud.exists_or_raise(version.version)


def test_create_version_different_configs(
    db: Session, example_config_blob: ConfigBlob
) -> None:
    """Test that version numbers are independent across different configs."""
    project = create_test_project(db)

    # Create two configs
    config1 = create_test_config(db, project_id=project.id, name="config-1")
    config2 = create_test_config(db, project_id=project.id, name="config-2")

    # Create versions for config1
    version_crud1 = ConfigVersionCrud(
        session=db, project_id=project.id, config_id=config1.id
    )
    version2_config1 = version_crud1.create_or_raise(
        ConfigVersionCreate(config_blob=example_config_blob, commit_message="V2")
    )

    # Create versions for config2
    version_crud2 = ConfigVersionCrud(
        session=db, project_id=project.id, config_id=config2.id
    )
    version2_config2 = version_crud2.create_or_raise(
        ConfigVersionCreate(config_blob=example_config_blob, commit_message="V2")
    )

    # Both should have version 2 (independent numbering)
    assert version2_config1.version == 2
    assert version2_config2.version == 2
    assert version2_config1.config_id == config1.id
    assert version2_config2.config_id == config2.id


def test_read_all_versions_config_not_found(db: Session) -> None:
    """Test reading versions for a non-existent config raises HTTPException."""
    project = create_test_project(db)
    non_existent_config_id = uuid4()

    version_crud = ConfigVersionCrud(
        session=db, project_id=project.id, config_id=non_existent_config_id
    )

    with pytest.raises(
        HTTPException, match=f"config with id '{non_existent_config_id}' not found"
    ):
        version_crud.read_all()
