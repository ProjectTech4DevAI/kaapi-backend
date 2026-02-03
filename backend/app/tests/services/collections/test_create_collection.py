from typing import Any
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.parse import urlparse
import uuid
from uuid import UUID, uuid4

import pytest
from moto import mock_aws
from sqlmodel import Session

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.crud import CollectionCrud, CollectionJobCrud, DocumentCollectionCrud
from app.models import CollectionJobStatus, CollectionJob, CollectionActionType, Project
from app.models.collection import CreationRequest
from app.services.collections.create_collection import start_job, execute_job
from app.tests.utils.llm_provider import get_mock_provider
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_collection_job, get_assistant_collection
from app.tests.utils.document import DocumentStore


@pytest.fixture(scope="function")
def aws_credentials() -> Any:
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION


def create_collection_job_for_create(
    db: Session,
    project: Project,
    job_id: UUID,
) -> CollectionJob:
    """Pre-create a CREATE job with the given id so start_job can update it."""
    return CollectionJobCrud(db, project.id).create(
        CollectionJob(
            id=job_id,
            action_type=CollectionActionType.CREATE,
            project_id=project.id,
            collection_id=None,
            status=CollectionJobStatus.PENDING,
        )
    )


def test_start_job_creates_collection_job_and_schedules_task(db: Session) -> None:
    """
    start_job should:
      - update an existing CollectionJob (status=PENDING, action=CREATE)
      - call start_low_priority_job with the correct kwargs
      - return the job UUID (same one that was passed in)
    """
    project = get_project(db)
    request = CreationRequest(
        documents=[UUID("f3e86a17-1e6f-41ec-b020-5b08eebef928")],
        batch_size=1,
        callback_url=None,
        provider="openai",
    )
    job_id = uuid4()

    _ = get_collection_job(
        db,
        project,
        job_id=job_id,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        collection_id=None,
    )

    with patch(
        "app.services.collections.create_collection.start_low_priority_job"
    ) as mock_schedule:
        mock_schedule.return_value = "fake-task-id"

        returned_job_id = start_job(
            db=db,
            request=request,
            project_id=project.id,
            collection_job_id=job_id,
            with_assistant=True,
            organization_id=project.organization_id,
        )

        assert returned_job_id == job_id

        job = CollectionJobCrud(db, project.id).read_one(job_id)
        assert job.id == job_id
        assert job.project_id == project.id
        assert job.status == CollectionJobStatus.PENDING
        assert job.action_type in (
            CollectionActionType.CREATE,
            CollectionActionType.CREATE.value,
        )
        assert job.collection_id is None

        mock_schedule.assert_called_once()
        kwargs = mock_schedule.call_args.kwargs
        assert (
            kwargs["function_path"]
            == "app.services.collections.create_collection.execute_job"
        )
        assert kwargs["project_id"] == project.id
        assert kwargs["organization_id"] == project.organization_id
        assert kwargs["job_id"] == str(job_id)
        assert kwargs["request"] == request.model_dump(mode="json")


@pytest.mark.usefixtures("aws_credentials")
@mock_aws
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_job_success_flow_updates_job_and_creates_collection(
    mock_get_llm_provider: MagicMock, db: Session
) -> None:
    """
    execute_job should:
      - set task_id on the CollectionJob
      - ingest documents into a vector store
      - create an OpenAI assistant
      - create a Collection with llm fields filled
      - link the CollectionJob -> collection_id, set status=successful
      - create DocumentCollection links
    """
    project = get_project(db)

    aws = AmazonCloudStorageClient()
    aws.create()

    store = DocumentStore(db=db, project_id=project.id)
    document = store.put()
    s3_key = Path(urlparse(document.object_store_url).path).relative_to("/")
    aws.client.put_object(Bucket=settings.AWS_S3_BUCKET, Key=str(s3_key), Body=b"test")

    sample_request = CreationRequest(
        documents=[document.id], batch_size=1, callback_url=None, provider="openai"
    )

    mock_get_llm_provider.return_value = get_mock_provider(
        llm_service_id="mock_vector_store_id", llm_service_name="openai vector store"
    )

    job_id = uuid4()
    _ = get_collection_job(
        db,
        project,
        job_id=job_id,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        collection_id=None,
    )

    task_id = uuid4()

    with patch("app.services.collections.create_collection.Session") as SessionCtor:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        execute_job(
            request=sample_request.model_dump(),
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(task_id),
            with_assistant=True,
            job_id=str(job_id),
            task_instance=None,
        )

    updated_job = CollectionJobCrud(db, project.id).read_one(job_id)
    assert updated_job.task_id == str(task_id)
    assert updated_job.status == CollectionJobStatus.SUCCESSFUL
    assert updated_job.collection_id is not None

    created_collection = CollectionCrud(db, project.id).read_one(
        updated_job.collection_id
    )
    assert created_collection.llm_service_id == "mock_vector_store_id"
    assert created_collection.llm_service_name == "openai vector store"
    assert created_collection.updated_at is not None

    docs = DocumentCollectionCrud(db).read(created_collection, skip=0, limit=10)
    assert len(docs) == 1
    assert docs[0].fname == document.fname


@pytest.mark.usefixtures("aws_credentials")
@mock_aws
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_job_assistant_create_failure_marks_failed_and_deletes_collection(
    mock_get_llm_provider: MagicMock, db
) -> None:
    project = get_project(db)

    job = get_collection_job(
        db,
        project,
        job_id=uuid4(),
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        collection_id=None,
    )

    req = CreationRequest(
        documents=[], batch_size=1, callback_url=None, provider="openai"
    )

    mock_provider = get_mock_provider(
        llm_service_id="vs_123", llm_service_name="openai vector store"
    )
    mock_get_llm_provider.return_value = mock_provider

    with patch(
        "app.services.collections.create_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.create_collection.CollectionCrud"
    ) as MockCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        MockCrud.return_value.create.side_effect = Exception("DB constraint violation")

        task_id = str(uuid4())
        execute_job(
            request=req.model_dump(),
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=task_id,
            with_assistant=True,
            job_id=str(job.id),
            task_instance=None,
        )

    mock_provider.delete.assert_called_once()


@pytest.mark.usefixtures("aws_credentials")
@mock_aws
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.send_callback")
def test_execute_job_success_flow_callback_job_and_creates_collection(
    mock_send_callback: MagicMock,
    mock_get_llm_provider: MagicMock,
    db,
) -> None:
    """
    execute_job should:
      - set task_id on the CollectionJob
      - ingest documents into a vector store
      - create an OpenAI assistant
      - create a Collection with llm fields filled
      - link the CollectionJob -> collection_id, set status=successful
      - create DocumentCollection links
    """
    project = get_project(db)

    aws = AmazonCloudStorageClient()
    aws.create()

    store = DocumentStore(db=db, project_id=project.id)
    document = store.put()
    s3_key = Path(urlparse(document.object_store_url).path).relative_to("/")
    aws.client.put_object(Bucket=settings.AWS_S3_BUCKET, Key=str(s3_key), Body=b"test")

    callback_url = "https://example.com/collections/create-success"

    sample_request = CreationRequest(
        documents=[document.id],
        batch_size=1,
        callback_url=callback_url,
        provider="openai",
    )

    mock_get_llm_provider.return_value = get_mock_provider(
        llm_service_id="mock_vector_store_id", llm_service_name="openai vector store"
    )

    job_id = uuid.uuid4()
    _ = get_collection_job(
        db,
        project,
        job_id=job_id,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        collection_id=None,
    )

    task_id = uuid.uuid4()

    with patch("app.services.collections.create_collection.Session") as SessionCtor:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        mock_send_callback.return_value = MagicMock(status_code=403)

        execute_job(
            request=sample_request.model_dump(),
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(task_id),
            with_assistant=True,
            job_id=str(job_id),
            task_instance=None,
        )

    updated_job = CollectionJobCrud(db, project.id).read_one(job_id)
    collection = CollectionCrud(db, project.id).read_one(updated_job.collection_id)

    mock_send_callback.assert_called_once()
    cb_url_arg, payload_arg = mock_send_callback.call_args.args
    assert str(cb_url_arg) == callback_url
    assert payload_arg["success"] is True
    assert payload_arg["data"]["status"] == CollectionJobStatus.SUCCESSFUL
    assert payload_arg["data"]["collection"]["id"] == str(collection.id)
    assert uuid.UUID(payload_arg["data"]["job_id"]) == job_id


@pytest.mark.usefixtures("aws_credentials")
@mock_aws
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.send_callback")
def test_execute_job_success_creates_collection_with_callback(
    mock_send_callback: MagicMock,
    mock_get_llm_provider: MagicMock,
    db,
) -> None:
    """
    execute_job should:
      - set task_id on the CollectionJob
      - ingest documents into a vector store
      - create an OpenAI assistant
      - create a Collection with llm fields filled
      - link the CollectionJob -> collection_id, set status=successful
      - create DocumentCollection links
    """
    project = get_project(db)

    aws = AmazonCloudStorageClient()
    aws.create()

    store = DocumentStore(db=db, project_id=project.id)
    document = store.put()
    s3_key = Path(urlparse(document.object_store_url).path).relative_to("/")
    aws.client.put_object(Bucket=settings.AWS_S3_BUCKET, Key=str(s3_key), Body=b"test")

    callback_url = "https://example.com/collections/create-success"

    sample_request = CreationRequest(
        documents=[document.id],
        batch_size=1,
        callback_url=callback_url,
        provider="openai",
    )

    mock_get_llm_provider.return_value = get_mock_provider(
        llm_service_id="mock_vector_store_id", llm_service_name="gpt-4o"
    )

    job_id = uuid.uuid4()
    _ = get_collection_job(
        db,
        project,
        job_id=job_id,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        collection_id=None,
    )

    task_id = uuid.uuid4()

    with patch("app.services.collections.create_collection.Session") as SessionCtor:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        mock_send_callback.return_value = MagicMock(status_code=403)

        execute_job(
            request=sample_request.model_dump(),
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(task_id),
            with_assistant=True,
            job_id=str(job_id),
            task_instance=None,
        )

    updated_job = CollectionJobCrud(db, project.id).read_one(job_id)
    collection = CollectionCrud(db, project.id).read_one(updated_job.collection_id)

    mock_send_callback.assert_called_once()
    cb_url_arg, payload_arg = mock_send_callback.call_args.args
    assert str(cb_url_arg) == callback_url
    assert payload_arg["success"] is True
    assert payload_arg["data"]["status"] == CollectionJobStatus.SUCCESSFUL
    assert payload_arg["data"]["collection"]["id"] == str(collection.id)
    assert uuid.UUID(payload_arg["data"]["job_id"]) == job_id


@pytest.mark.usefixtures("aws_credentials")
@mock_aws
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.send_callback")
@patch("app.services.collections.create_collection.CollectionCrud")
def test_execute_job_failure_flow_callback_job_and_marks_failed(
    MockCollectionCrud,
    mock_send_callback: MagicMock,
    mock_get_llm_provider: MagicMock,
    db: Session,
) -> None:
    """
    When creation fails, the job should be marked as FAILED, an error should be logged,
    and a failure callback with the error message should be triggered.
    """
    project = get_project(db)

    collection = get_assistant_collection(db, project, assistant_id="asst_123")
    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        collection_id=None,
    )

    mock_get_llm_provider.return_value = MagicMock()

    callback_url = "https://example.com/collections/create-failure"

    collection_crud_instance = MockCollectionCrud.return_value
    collection_crud_instance.read_one.return_value = collection

    sample_request = CreationRequest(
        documents=[uuid.uuid4()],
        batch_size=1,
        callback_url=callback_url,
        provider="openai",
    )

    task_id = uuid.uuid4()

    with patch("app.services.collections.create_collection.Session") as SessionCtor:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        execute_job(
            request=sample_request.model_dump(),
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(task_id),
            with_assistant=True,
            job_id=str(job.id),
            task_instance=None,
        )

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)

    assert updated_job.status == CollectionJobStatus.FAILED
    assert "Requested atleast 1 document retrieved 0" in (
        updated_job.error_message or ""
    )

    mock_send_callback.assert_called_once()
    cb_url_arg, payload_arg = mock_send_callback.call_args.args
    assert str(cb_url_arg) == callback_url
    assert payload_arg["success"] is False
    assert "Requested atleast 1 document retrieved 0" in (payload_arg["error"] or "")
    assert payload_arg["data"]["status"] == CollectionJobStatus.FAILED
    assert payload_arg["data"]["collection"] is None
    assert uuid.UUID(payload_arg["data"]["job_id"]) == job.id
