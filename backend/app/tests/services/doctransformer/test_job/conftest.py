"""
Pytest fixtures for document transformation service tests.
"""
import os
from typing import Any, Callable, Generator, Tuple
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks
from sqlmodel import Session
from tenacity import retry, stop_after_attempt, wait_fixed

from app.crud import get_project_by_id
from app.services.doctransform import job
from app.core.config import settings
from app.models import Document, Project, AuthContext
from app.tests.utils.document import DocumentStore
from app.tests.utils.auth import TestAuthContext


@pytest.fixture(scope="class")
def aws_credentials() -> None:
    """Set up AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION


@pytest.fixture
def fast_execute_job() -> (
    Generator[
        Callable[[int, str, str, str, str, str, str | None, Any], Any], None, None
    ]
):
    """Create a version of execute_job without retry delays for faster testing."""

    original_execute_job = job.execute_job

    @retry(
        stop=stop_after_attempt(2), wait=wait_fixed(0.01)
    )  # Very fast retry for tests
    def fast_execute_job_func(
        project_id: int,
        job_id: str,
        source_document_id: str,
        transformer_name: str,
        target_format: str,
        task_id: str,
        callback_url: str | None,
        task_instance,
    ) -> Any:
        # Call the original function's implementation without the decorator
        return original_execute_job.__wrapped__(
            project_id,
            job_id,
            source_document_id,
            transformer_name,
            target_format,
            task_id,
            callback_url,
            task_instance,
        )

    with patch.object(job, "execute_job", fast_execute_job_func):
        yield fast_execute_job_func


@pytest.fixture
def current_user(db: Session, user_api_key: TestAuthContext) -> AuthContext:
    """Create a test user for testing."""
    return AuthContext(
        user=user_api_key.user,
        organization=user_api_key.organization,
        project=user_api_key.project,
    )


@pytest.fixture
def background_tasks() -> BackgroundTasks:
    """Create BackgroundTasks instance."""
    return BackgroundTasks()


@pytest.fixture
def test_document(db: Session, current_user: AuthContext) -> Tuple[Document, Project]:
    """Create a test document for the current user's project."""
    store = DocumentStore(db, current_user.project.id)
    project = get_project_by_id(session=db, project_id=current_user.project.id)
    return store.put(), project
