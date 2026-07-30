from typing import Any
import os
from unittest.mock import patch, MagicMock
import uuid
from uuid import UUID, uuid4

from celery.exceptions import Reject, Retry, SoftTimeLimitExceeded
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


def _mock_storage() -> MagicMock:
    """Storage mock returning a fixed file size for documents missing one."""
    storage = MagicMock()
    storage.get_file_size_kb.return_value = 10.0
    return storage


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
        "app.services.collections.create_collection.start_collection_setup_job"
    ) as mock_schedule:
        mock_schedule.return_value = "fake-task-id"

        returned_job_id = start_job(
            db=db,
            request=request,
            project_id=project.id,
            collection_job_id=job_id,
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

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_get_provider.return_value = mock_provider
    mock_get_storage.return_value = _mock_storage()

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
    assert updated_job.knowledge_base_id == "vs_123"

    mock_provider.upload_files.assert_not_called()
    mock_queue_batch.assert_called_once()
    kw = mock_queue_batch.call_args.kwargs
    assert kw["batch_number"] == 1
    assert kw["vector_store_id"] == "vs_123"
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

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create_vector_store.side_effect = RuntimeError(
        "vector store create failed"
    )
    mock_get_provider.return_value = mock_provider
    mock_get_storage.return_value = _mock_storage()

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(RuntimeError, match="vector store create failed"):
            execute_setup_job(
                request=request.model_dump(mode="json"),
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
    assert "vector store create failed" in (updated_job.error_message or "")
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

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create_vector_store.side_effect = RuntimeError("create error")
    mock_get_provider.return_value = mock_provider
    mock_get_storage.return_value = _mock_storage()

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

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create_vector_store.side_effect = Timeout(300)
    mock_get_provider.return_value = mock_provider
    mock_get_storage.return_value = _mock_storage()

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


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_setup_job_soft_time_limit_marks_failed_and_reraises(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create_vector_store.side_effect = SoftTimeLimitExceeded()
    mock_get_provider.return_value = mock_provider
    mock_get_storage.return_value = _mock_storage()

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(SoftTimeLimitExceeded):
            execute_setup_job(
                request=request.model_dump(mode="json"),
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


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_batch_job_non_final_queues_next_batch(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc1 = store.put()
    doc2 = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_get_provider.return_value = mock_provider

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

    mock_provider.upload_files.assert_called_once()
    mock_queue_batch.assert_called_once()
    kw = mock_queue_batch.call_args.kwargs
    assert kw["batch_number"] == 2
    assert kw["batch_doc_ids"] == [str(doc2.id)]
    assert kw["remaining_batches"] == []
    assert kw["vector_store_id"] == "vs_123"

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.current_batch_number == 1
    assert str(doc1.id) in (updated_job.documents_uploaded or [])


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_batch_job_final_batch_creates_collection_and_marks_successful(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
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
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(uuid4()),
            job_id=str(job.id),
            task_instance=None,
            vector_store_id="vs_final",
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
    assert collection.knowledge_base_id == "vs_final"

    linked_docs = DocumentCollectionCrud(db).read(collection, skip=0, limit=10)
    assert len(linked_docs) == 1
    assert linked_docs[0].id == doc.id

    mock_queue_batch.assert_not_called()


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.send_callback")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.start_collection_batch_job")
def test_execute_batch_job_final_batch_sends_success_callback(
    mock_queue_batch: MagicMock,
    mock_get_provider: MagicMock,
    mock_send_callback: MagicMock,
    mock_get_storage: MagicMock,
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
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(uuid4()),
            job_id=str(job.id),
            task_instance=None,
            vector_store_id="vs_final",
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


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_provider_failure_marks_failed_and_raises(
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
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
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "vector store error" in (updated_job.error_message or "")


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
@patch("app.services.collections.create_collection.CollectionCrud")
def test_execute_batch_job_cleanup_called_when_provider_create_succeeds_but_db_fails(
    MockCollectionCrud: MagicMock,
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
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
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    mock_provider.delete.assert_called_once()


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_timeout_marks_failed_and_reraises(
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
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
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "soft time limit" in (updated_job.error_message or "")


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_soft_time_limit_marks_failed_and_reraises(
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = SoftTimeLimitExceeded()
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
        with pytest.raises(SoftTimeLimitExceeded):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "soft time limit" in (updated_job.error_message or "")


def _celery_task_instance(retries: int, max_retries: int | None = 3) -> MagicMock:
    """Bound-task stand-in; production always passes `task_instance=self`."""
    task = MagicMock()
    task.request.retries = retries
    task.max_retries = max_retries
    task.retry.side_effect = Retry("requeued")
    return task


@pytest.mark.parametrize("timeout_exc", [Timeout(300), SoftTimeLimitExceeded()])
@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_timeout_retries_and_keeps_vector_store(
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    timeout_exc: BaseException,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = timeout_exc
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)
    task = _celery_task_instance(retries=0)

    patcher = _patch_session(db)
    try:
        with pytest.raises(Retry):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=task,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    task.retry.assert_called_once()
    mock_provider.delete.assert_not_called()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.PROCESSING


@pytest.mark.parametrize(
    "retry_effect",
    [
        pytest.param(Reject(OSError("broker down"), False), id="broker_publish_fails"),
        pytest.param(TimeoutError("soft limit"), id="called_outside_worker"),
    ],
)
@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_unschedulable_retry_fails_job(
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    retry_effect: BaseException,
    db: Session,
) -> None:
    """Reject (publish failed) and a bare exc (called outside a worker) both mean
    nothing was queued, so the job must fail rather than strand in PROCESSING."""
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = SoftTimeLimitExceeded()
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)
    task = _celery_task_instance(retries=0)
    task.retry.side_effect = retry_effect

    patcher = _patch_session(db)
    try:
        with pytest.raises(SoftTimeLimitExceeded):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=task,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    mock_provider.delete.assert_called_once()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_infinite_max_retries_does_not_crash(
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    """`max_retries=None` is retry-forever; comparing against it raises TypeError,
    which would skip failure handling entirely."""
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = SoftTimeLimitExceeded()
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)
    task = _celery_task_instance(retries=0, max_retries=None)

    patcher = _patch_session(db)
    try:
        with pytest.raises(SoftTimeLimitExceeded):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=task,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    task.retry.assert_not_called()
    assert (
        CollectionJobCrud(db, project.id).read_one(job.id).status
        == CollectionJobStatus.FAILED
    )


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_timeout_fails_once_retries_exhausted(
    mock_get_provider: MagicMock,
    mock_get_storage: MagicMock,
    db: Session,
) -> None:
    project = get_project(db)
    store = DocumentStore(db=db, project_id=project.id)
    doc = store.put()

    mock_provider = get_mock_provider("vs_123", "openai vector store")
    mock_provider.create.side_effect = SoftTimeLimitExceeded()
    mock_get_provider.return_value = mock_provider

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PROCESSING,
    )
    request = CreationRequest(documents=[doc.id], provider="openai", callback_url=None)
    task = _celery_task_instance(retries=3)

    patcher = _patch_session(db)
    try:
        with pytest.raises(SoftTimeLimitExceeded):
            execute_batch_job(
                request=request.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=task,
                vector_store_id="vs_123",
                batch_number=1,
                batch_doc_ids=[str(doc.id)],
                remaining_batches=[],
            )
    finally:
        patcher.stop()

    task.retry.assert_not_called()
    mock_provider.delete.assert_called_once()

    updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert updated_job.status == CollectionJobStatus.FAILED
    assert "soft time limit" in (updated_job.error_message or "")


@patch("app.services.collections.create_collection.get_cloud_storage")
@patch("app.services.collections.create_collection.send_callback")
@patch("app.services.collections.create_collection.get_llm_provider")
def test_execute_batch_job_failure_sends_callback(
    mock_get_provider: MagicMock,
    mock_send_callback: MagicMock,
    mock_get_storage: MagicMock,
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
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                task_instance=None,
                vector_store_id="vs_123",
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
