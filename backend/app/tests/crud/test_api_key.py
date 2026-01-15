from uuid import uuid4

import pytest
from sqlmodel import Session
from fastapi import HTTPException

from app.crud import APIKeyCrud
from app.models import APIKey, Project, User
from app.tests.utils.test_data import create_test_project, create_test_api_key
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import get_non_existent_id


def test_create_api_key(db: Session) -> None:
    """Test creating a new API key"""
    project = create_test_project(db)
    user = create_random_user(db)

    api_key_crud = APIKeyCrud(session=db, project_id=project.id)
    raw_key, api_key = api_key_crud.create(user_id=user.id, project_id=project.id)

    assert api_key.id is not None
    assert api_key.user_id == user.id
    assert api_key.project_id == project.id
    assert api_key.organization_id == project.organization_id
    assert api_key.key_prefix is not None
    assert api_key.key_hash is not None
    assert api_key.is_deleted is False
    assert api_key.deleted_at is None
    assert raw_key is not None
    assert len(raw_key) > 0


def test_create_api_key_with_nonexistent_project(db: Session) -> None:
    """Test creating API key with a project that doesn't exist"""
    user = create_random_user(db)
    fake_project_id = get_non_existent_id(session=db, model=Project)

    api_key_crud = APIKeyCrud(session=db, project_id=fake_project_id)

    with pytest.raises(HTTPException) as exc_info:
        api_key_crud.create(user_id=user.id, project_id=fake_project_id)

    assert exc_info.value.status_code == 404
    assert "Project not found" in str(exc_info.value.detail)


def test_create_api_key_with_nonexistent_user(db: Session) -> None:
    """Test creating API key with a user that doesn't exist"""
    project = create_test_project(db)
    fake_user_id = get_non_existent_id(session=db, model=User)

    api_key_crud = APIKeyCrud(session=db, project_id=project.id)

    with pytest.raises(HTTPException) as exc_info:
        api_key_crud.create(user_id=fake_user_id, project_id=project.id)

    assert exc_info.value.status_code == 404
    assert "User not found" in str(exc_info.value.detail)


def test_read_one_api_key(db: Session) -> None:
    """Test reading a single API key by ID"""
    api_key = create_test_api_key(db=db)
    api_key_crud = APIKeyCrud(session=db, project_id=api_key.project_id)
    retrieved_key = api_key_crud.read_one(key_id=api_key.id)

    assert retrieved_key is not None
    assert retrieved_key.id == api_key.id
    assert retrieved_key.key_prefix == api_key.key_prefix
    assert retrieved_key.project_id == api_key.project_id


def test_read_one_api_key_nonexistent(db: Session) -> None:
    """Test reading an API key that doesn't exist"""
    project = create_test_project(db)

    api_key_crud = APIKeyCrud(session=db, project_id=project.id)
    fake_key_id = uuid4()

    retrieved_key = api_key_crud.read_one(key_id=fake_key_id)

    assert retrieved_key is None


def test_read_one_api_key_wrong_project(db: Session) -> None:
    """Test that reading an API key from a different project returns None"""

    api_key = create_test_api_key(db=db)
    project2 = create_test_project(db)

    # Try to read it from project2 scope
    api_key_crud2 = APIKeyCrud(session=db, project_id=project2.id)
    retrieved_key = api_key_crud2.read_one(key_id=api_key.id)

    assert retrieved_key is None


def test_read_one_deleted_api_key(db: Session) -> None:
    """Test that reading a deleted API key returns None"""
    api_key = create_test_api_key(db=db)

    api_key_crud = APIKeyCrud(session=db, project_id=api_key.project_id)

    api_key_crud.delete(key_id=api_key.id)
    retrieved_key = api_key_crud.read_one(key_id=api_key.id)
    assert retrieved_key is None


def test_read_all_api_keys(db: Session) -> None:
    """Test reading all API keys for a project"""
    project = create_test_project(db)
    user = create_random_user(db)

    api_key_crud = APIKeyCrud(session=db, project_id=project.id)

    # Create multiple API keys
    key1 = create_test_api_key(db=db, project_id=project.id, user_id=user.id)
    key2 = create_test_api_key(db=db, project_id=project.id, user_id=user.id)
    key3 = create_test_api_key(db=db, project_id=project.id, user_id=user.id)

    # Read all keys
    all_keys = api_key_crud.read_all()

    assert len(all_keys) == 3
    key_ids = {key.id for key in all_keys}
    assert key1.id in key_ids
    assert key2.id in key_ids
    assert key3.id in key_ids


def test_read_all_api_keys_with_pagination(db: Session) -> None:
    """Test reading API keys with skip and limit parameters"""
    project = create_test_project(db)
    user = create_random_user(db)

    api_key_crud = APIKeyCrud(session=db, project_id=project.id)

    # Create 5 API keys
    for _ in range(5):
        create_test_api_key(db=db, project_id=project.id, user_id=user.id)

    # Test pagination
    page1 = api_key_crud.read_all(skip=0, limit=2)
    assert len(page1) == 2

    page2 = api_key_crud.read_all(skip=2, limit=2)
    assert len(page2) == 2

    page3 = api_key_crud.read_all(skip=4, limit=2)
    assert len(page3) == 1

    all_ids = (
        {key.id for key in page1}
        | {key.id for key in page2}
        | {key.id for key in page3}
    )
    assert len(all_ids) == 5


def test_read_all_excludes_deleted_keys(db: Session) -> None:
    """Test that read_all excludes deleted API keys"""
    project = create_test_project(db)
    user = create_random_user(db)

    api_key_crud = APIKeyCrud(session=db, project_id=project.id)

    # Create 3 API keys
    key1 = create_test_api_key(db=db, project_id=project.id, user_id=user.id)
    key2 = create_test_api_key(db=db, project_id=project.id, user_id=user.id)
    key3 = create_test_api_key(db=db, project_id=project.id, user_id=user.id)

    # Delete one
    api_key_crud.delete(key_id=key2.id)

    # Read all should only return 2
    all_keys = api_key_crud.read_all()
    assert len(all_keys) == 2
    key_ids = {key.id for key in all_keys}
    assert key1.id in key_ids
    assert key2.id not in key_ids
    assert key3.id in key_ids


def test_read_all_scoped_to_project(db: Session) -> None:
    """Test that read_all only returns keys for the specified project"""
    project1 = create_test_project(db)
    project2 = create_test_project(db)
    user = create_random_user(db)

    api_key_crud1 = APIKeyCrud(session=db, project_id=project1.id)
    api_key_crud2 = APIKeyCrud(session=db, project_id=project2.id)

    # Create keys for both projects
    api_key_crud1.create(user_id=user.id, project_id=project1.id)
    api_key_crud1.create(user_id=user.id, project_id=project1.id)
    api_key_crud2.create(user_id=user.id, project_id=project2.id)

    # Each project should only see its own keys
    project1_keys = api_key_crud1.read_all()
    project2_keys = api_key_crud2.read_all()

    assert len(project1_keys) == 2
    assert len(project2_keys) == 1
    assert all(key.project_id == project1.id for key in project1_keys)
    assert all(key.project_id == project2.id for key in project2_keys)


def test_delete_api_key(db: Session) -> None:
    """Test soft deleting an API key"""
    api_key = create_test_api_key(db=db)

    api_key_crud = APIKeyCrud(session=db, project_id=api_key.project_id)

    api_key_crud.delete(key_id=api_key.id)

    db_key = db.get(APIKey, api_key.id)
    assert db_key is not None
    assert db_key.is_deleted is True
    assert db_key.deleted_at is not None

    retrieved_key = api_key_crud.read_one(key_id=api_key.id)
    assert retrieved_key is None


def test_delete_nonexistent_api_key(db: Session) -> None:
    """Test deleting an API key that doesn't exist"""
    project = create_test_project(db)

    api_key_crud = APIKeyCrud(session=db, project_id=project.id)
    fake_key_id = uuid4()

    with pytest.raises(HTTPException) as exc_info:
        api_key_crud.delete(key_id=fake_key_id)

    assert exc_info.value.status_code == 404
    assert "API Key not found" in str(exc_info.value.detail)


def test_delete_api_key_from_wrong_project(db: Session) -> None:
    """Test that deleting an API key from a different project fails"""
    api_key = create_test_api_key(db=db)
    project2 = create_test_project(db)

    api_key_crud2 = APIKeyCrud(session=db, project_id=project2.id)
    with pytest.raises(HTTPException) as exc_info:
        api_key_crud2.delete(key_id=api_key.id)

    assert exc_info.value.status_code == 404
    assert "API Key not found" in str(exc_info.value.detail)

    db_key = db.get(APIKey, api_key.id)
    assert db_key is not None
    assert db_key.is_deleted is False


def test_delete_already_deleted_api_key(db: Session) -> None:
    """Test deleting an API key that's already deleted"""
    api_key = create_test_api_key(db=db)
    api_key_crud = APIKeyCrud(session=db, project_id=api_key.project_id)

    api_key_crud.delete(key_id=api_key.id)

    with pytest.raises(HTTPException) as exc_info:
        api_key_crud.delete(key_id=api_key.id)

    assert exc_info.value.status_code == 404
    assert "API Key not found" in str(exc_info.value.detail)
