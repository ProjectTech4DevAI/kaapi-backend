from typing import Any
import os
from unittest.mock import patch, MagicMock
import uuid
from uuid import UUID, uuid4

from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.crud import CollectionCrud, CollectionJobCrud, DocumentCollectionCrud
from app.models import CollectionJobStatus, CollectionJob, CollectionActionType, Project
from app.models.collection import CreationRequest
from app.services.collections.create_collection import (
    start_job,
    execute_setup_job,
    execute_batch_job,
)
from app.tests.utils.llm_provider import get_mock_provider
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_collection_job
from app.tests.utils.document import DocumentStore


@pytest.fixture(scope="function")
def aws_credentials() -> Any:
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION


def _mock_provider_with_size(llm_service_id: str, llm_service_name: str):
    """Returns a mock provider whose upload_files sets file_size_kb=10.0 on each doc."""
    mock_provider = get_mock_provider(llm_service_id, llm_service_name)

    def _set_file_size(storage, docs, project_id):
        for doc in docs:
            doc.file_size_kb = 10.0

    mock_provider.upload_files.side_effect = _set_file_size
    return mock_provider


def _patch_session(db: Session):
    """Context manager that routes all Session(engine) calls to the test db."""
    patcher = patch("app.services.collections.create_collection.Session")
    mock_ctor = patcher.start()
    mock_ctor.return_value.__enter__.return_value = db
    mock_ctor.return_value.__exit__.return_value = False
    return patcher


# ---------------------------------------------------------------------------
# start_job
# ---------------------------------------------------------------------------


def test_start_job_creates_collection_job_and_schedules_task(db: Session) -> None:
    project = get_project(db)
    request = CreationRequest(
        documents=[UUID("f3e86a17-1e6f-41ec-b020-5b08eebef928")],
        callback_url=None,
        provider="openai",
    )
    job_id = uuid4()

    get_collection_job(
        db,
        project,
        job_id=job_id,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        collection_id=None,
    )

    with patch(
        "app.services.collections.create_collection.start_create_collection_job"
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
    mock_schedule.assert_called_once()
    kwargs = mock_schedule.call_args.kwargs
    assert kwargs["project_id"] == project.id
    assert kwargs["organization_id"] == project.organization_id
    assert kwargs["job_id"] == str(job_id)
    assert kwargs["request"] == request.model_dump(mode="json")


# ---------------------------------------------------------------------------
# execute_setup_job
# ---------------------------------------------------------------------------


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_setup_job_marks_processing_and_queues_first_batch(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_get_provider.return_value = _mock_provider_with_size(
        "vs_123", "openai vector store"
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)
    task_id = str(uuid4())

    patcher = _patch_session(db)
    try:
        execute_setup_job(
            request=request.model_dump(mode="json"),
            with_assistant=False,
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=task_id,
            job_id=str(job.id),
            task_instance=None,
        )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.PROCESSING
    assert updated_job.task_id == task_id

    mock_queue_batch.assert_called_once()
    kw = mock_queue_batch.call_args.kwargs
    assert kw["batch_number"] == 1
    assert kw["vector_store_id"] is None
    assert str(doc.id) in kw["batch_doc_ids"]
    assert kw["remaining_batches"] == []


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_setup_job_failure_marks_job_failed_and_raises(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = _mock_provider_with_size("vs_123", "openai vector store")
    mock_provider.upload_files.side_effect = RuntimeError("S3 upload failed")
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(RuntimeError, match="S3 upload failed"):
            execute_setup_job(
                request=request.model_dump(mode="json"),
                with_assistant=False,
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
            )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "S3 upload failed" in (updated_job.error_message or "")
    mock_queue_batch.assert_not_called()


@patch("app.services.collections.create_collection.send_callback")
@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_setup_job_failure_sends_callback(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    mock_send_callback: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = _mock_provider_with_size("vs_123", "openai vector store")
    mock_provider.upload_files.side_effect = RuntimeError("upload error")
    mock_get_provider.return_value = mock_provider

    callback_url = "https://example.com/callback"
    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
    )
    request = CreationRequest(
        documents=[doc.id], provider="openai", callback_url=callback_url
    )

    patcher = _patch_session(db)
    try:
        with pytest.raises(RuntimeError):
            execute_setup_job(
                request=request.model_dump(mode="json"),
                with_assistant=False,
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
            )
    finally:
        patcher.stop()

    mock_send_callback.assert_called_once()
    cb_url, payload = mock_send_callback.call_args.args
    assert str(cb_url) == callback_url
    assert payload["success"] is False
    assert payload["data"]["status"] == CollectionJobStatus.FAILED


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_setup_job_timeout_marks_failed_and_reraises(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = _mock_provider_with_size("vs_123", "openai vector store")
    mock_provider.upload_files.side_effect = Timeout(300)
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(Timeout):
            execute_setup_job(
                request=request.model_dump(mode="json"),
                with_assistant=False,
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
            )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "soft time limit" in (updated_job.error_message or "")


# ---------------------------------------------------------------------------
# execute_batch_job
# ---------------------------------------------------------------------------


@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_batch_job_non_final_queues_next_batch(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc1 = store.put()
    doc2 = store.put()

    mock_get_provider.return_value = get_mock_provider("vs_123", "openai vector store")

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(
        documents=[doc1.id, doc2.id], provider="openai", callback_url=None
    )
    task_id = str(uuid4())

    patcher = _patch_session(db)
    try:
        execute_batch_job(
            request=request.model_dump(mode="json"),
            with_assistant=False,
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=task_id,
            job_id=str(job.id),
            task_instance=None,
            vector_store_id="vs_123",
            batch_number=1,
            batch_doc_ids=[str(doc1.id)],
            remaining_batches=[[str(doc2.id)]],
        )
    finally:
        patcher.stop()

    mock_queue_batch.assert_called_once()
    kw = mock_queue_batch.call_args.kwargs
    assert kw["batch_number"] == 2
    assert kw["batch_doc_ids"] == [str(doc2.id)]
    assert kw["remaining_batches"] == []
    assert kw["vector_store_id"] == "vs_123"

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.current_batch_number == 1
    assert str(doc1.id) in (updated_job.documents_uploaded or [])


@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_batch_job_final_batch_creates_collection_and_marks_successful(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_get_provider.return_value = get_mock_provider(
        "vs_final", "openai vector store"
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        execute_batch_job(
            request=request.model_dump(mode="json"),
            with_assistant=False,
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(uuid4()),
            job_id=str(job.id),
            task_instance=None,
            vector_store_id=None,
            batch_number=1,
            batch_doc_ids=[str(doc.id)],
            remaining_batches=[],
        )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.SUCCESSFUL
    assert updated_job.collection_id is not None

    collection = CollectionCrud(db, project.id).read_one(updated_job.collection_id)
    assert collection.llm_service_id == "vs_final"

    linked_docs = DocumentCollectionCrud(db).read(collection, skip=0, limit=10)
    assert len(linked_docs) == 1
    assert linked_docs[0].id == doc.id

    mock_queue_batch.assert_not_called()


@patch("app.services.collections.create_collection.send_callback")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_batch_job_final_batch_sends_success_callback(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_send_callback: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_get_provider.return_value = get_mock_provider(
        "vs_final", "openai vector store"
    )

    callback_url = "https://example.com/success"
    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(
        documents=[doc.id], provider="openai", callback_url=callback_url
    )

    patcher = _patch_session(db)
    try:
        execute_batch_job(
            request=request.model_dump(mode="json"),
            with_assistant=False,
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(uuid4()),
            job_id=str(job.id),
            task_instance=None,
            vector_store_id=None,
            batch_number=1,
            batch_doc_ids=[str(doc.id)],
            remaining_batches=[],
        )
    finally:
        patcher.stop()

    mock_send_callback.assert_called_once()
    cb_url, payload = mock_send_callback.call_args.args
    assert str(cb_url) == callback_url
    assert payload["success"] is True
    assert payload["data"]["status"] == CollectionJobStatus.SUCCESSFUL
    assert payload["data"]["collection"] is not None


@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_provider_failure_marks_failed_and_raises(
    mock_get_provider: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = RuntimeError("vector store error")
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(RuntimeError, match="vector store error"):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                with_assistant=False,
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id=None,
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "vector store error" in (updated_job.error_message or "")


@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.CollectionCrud")
def test_execute_batch_job_cleanup_called_when_provider_create_succeeds_but_db_fails(
    MockCollectionCrud: MagicMock,
    mock_get_provider: MagicMock,
    db: Session,
) -> None:
    """provider.delete should be called if create() succeeded but finalization fails."""
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_get_provider.return_value = mock_provider

    MockCollectionCrud.return_value.create.side_effect = Exception("DB write failed")

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(Exception, match="DB write failed"):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                with_assistant=False,
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id=None,
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    mock_provider.delete.assert_called_once()


@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_timeout_marks_failed_and_reraises(
    mock_get_provider: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = Timeout(300)
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(Timeout):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                with_assistant=False,
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id=None,
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "soft time limit" in (updated_job.error_message or "")


@patch("app.services.collections.create_collection.send_callback")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_failure_sends_callback(
    mock_get_provider: MagicMock,
    mock_send_callback: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = RuntimeError("batch failed")
    mock_get_provider.return_value = mock_provider

    callback_url = "https://example.com/failure"
    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(
        documents=[doc.id], provider="openai", callback_url=callback_url
    )

    patcher = _patch_session(db)
    try:
        with pytest.raises(RuntimeError):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                with_assistant=False,
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id=None,
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    mock_send_callback.assert_called_once()
    cb_url, payload = mock_send_callback.call_args.args
    assert str(cb_url) == callback_url
    assert payload["success"] is False
    assert "batch failed" in (payload["error"] or "")
    assert payload["data"]["status"] == CollectionJobStatus.FAILED
