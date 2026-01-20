from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest
from openai import OpenAI
from sqlmodel import Session

from app.services.response.response import process_response
from app.models import (
    ResponsesAPIRequest,
    Assistant,
    Job,
    JobStatus,
    AssistantCreate,
    Project,
    JobType,
)
from app.utils import APIResponse
from app.tests.utils.test_data import create_test_credential
from app.tests.utils.llm_provider import mock_openai_response, generate_openai_id
from app.crud import JobCrud, create_assistant


@pytest.fixture
def setup_db(db: Session) -> tuple[Assistant, Job, Project]:
    """
    Fixture to set up a job and assistant in the database.

    Note: OpenAI and Langfuse credentials are already available via seed data,
    so this fixture only creates the assistant and job.
    """
    _, project = create_test_credential(db)

    assistant_create = AssistantCreate(
        name="Test Assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4",
    )
    client = OpenAI(api_key="test_api_key")
    assistant = create_assistant(
        session=db,
        assistant=assistant_create,
        openai_client=client,
        project_id=project.id,
        organization_id=project.organization_id,
    )

    job = JobCrud(session=db).create(
        job_type=JobType.RESPONSE,
        trace_id=str(uuid4()),
    )

    return assistant, job, project


def make_request(assistant_id: str, previous_response_id: str | None = None):
    return ResponsesAPIRequest(
        assistant_id=assistant_id,
        question="What is the capital of France?",
        callback_url="http://example.com/callback",
        response_id=previous_response_id,
    )


def test_process_response_success(
    db: Session, setup_db: tuple[Assistant, Job, Project]
) -> None:
    assistant, job, project = setup_db
    prev_id = generate_openai_id("resp_")
    request = make_request(assistant.assistant_id, prev_id)
    job_id = job.id
    task_id = "task_123"

    response, error = mock_openai_response("Mock response text.", prev_id), None

    # Mock LangfuseTracer to avoid actual Langfuse API calls
    mock_tracer = MagicMock()

    with (
        patch(
            "app.services.response.response.generate_response",
            return_value=(response, error),
        ),
        patch(
            "app.services.response.response.LangfuseTracer",
            return_value=mock_tracer,
        ),
        patch("app.services.response.response.Session", return_value=db),
    ):
        api_response: APIResponse = process_response(
            request=request,
            project_id=project.id,
            organization_id=project.organization_id,
            job_id=job_id,
            task_id=task_id,
            task_instance=None,
        )

        job = db.get(Job, job_id)
        assert api_response.success is True
        assert job.status == JobStatus.SUCCESS


def test_process_response_assistant_not_found(
    db: Session, setup_db: tuple[Assistant, Job, Project]
) -> None:
    _, job, project = setup_db
    request: ResponsesAPIRequest = make_request("non_existent_asst")

    with patch("app.services.response.response.Session", return_value=db):
        api_response: APIResponse = process_response(
            request=request,
            project_id=project.id,
            organization_id=project.organization_id,
            job_id=job.id,
            task_id="task_456",
            task_instance=None,
        )

    job = db.get(Job, job.id)
    assert api_response.success is False
    assert "Assistant not found" in api_response.error
    assert job.status == JobStatus.FAILED


def test_process_response_generate_response_failure(
    db: Session, setup_db: tuple[Assistant, Job, Project]
) -> None:
    assistant, job, project = setup_db
    request: ResponsesAPIRequest = make_request(assistant.assistant_id)

    # Mock LangfuseTracer to avoid actual Langfuse API calls
    mock_tracer = MagicMock()

    with (
        patch(
            "app.services.response.response.generate_response",
            return_value=(None, "Some error"),
        ),
        patch(
            "app.services.response.response.LangfuseTracer",
            return_value=mock_tracer,
        ),
        patch("app.services.response.response.Session", return_value=db),
    ):
        api_response: APIResponse = process_response(
            request=request,
            project_id=project.id,
            organization_id=project.organization_id,
            job_id=job.id,
            task_id="task_789",
            task_instance=None,
        )

    job = db.get(Job, job.id)
    assert api_response.success is False
    assert "Some error" in api_response.error
    assert job.status == JobStatus.FAILED


def test_process_response_unexpected_exception(
    db: Session, setup_db: tuple[Assistant, Job, Project]
) -> None:
    assistant, job, project = setup_db
    request: ResponsesAPIRequest = make_request(assistant.assistant_id)

    with (
        patch(
            "app.services.response.response.generate_response",
            side_effect=Exception("Boom"),
        ),
        patch("app.services.response.response.Session", return_value=db),
    ):
        api_response: APIResponse = process_response(
            request=request,
            project_id=project.id,
            organization_id=project.organization_id,
            job_id=job.id,
            task_id="task_999",
            task_instance=None,
        )

    job = db.get(Job, job.id)
    assert api_response.success is False
    assert "Unexpected error" in api_response.error
    assert job.status == JobStatus.FAILED
