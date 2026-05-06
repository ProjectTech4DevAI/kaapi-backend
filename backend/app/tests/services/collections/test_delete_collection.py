from unittest.mock import patch, MagicMock
from uuid import uuid4, UUID

from gevent import Timeout

import pytest
from sqlmodel import Session

from app.models.collection import DeletionRequest
from app.tests.utils.utils import get_project
from app.crud import CollectionJobCrud
from app.models import CollectionJobStatus, CollectionActionType
from app.tests.utils.collection import get_collection_job, get_vector_store_collection
from app.services.collections.delete_collection import start_job, execute_job


def test_start_job_creates_collection_job_and_schedules_task(db: Session) -> None:
    """
    - start_job should update an existing CollectionJob (status=PENDING, action=DELETE)
    - schedule the task with the provided job_id and collection_id
    - return the same job_id (UUID)
    """
    project = get_project(db)
    created_collection = get_vector_store_collection(db, project)

    req = DeletionRequest(collection_id=created_collection.id)

    with patch(
        "app.services.collections.delete_collection.start_delete_collection_job"
    ) as mock_schedule:
        mock_schedule.return_value = "fake-task-id"

        collection_job_id = uuid4()
        _ = get_collection_job(
            db,
            project,
            job_id=collection_job_id,
            action_type=CollectionActionType.DELETE,
            status=CollectionJobStatus.PENDING,
            collection_id=created_collection.id,
        )

        returned = start_job(
            db=db,
            request=req,
            project_id=project.id,
            collection_job_id=collection_job_id,
            organization_id=project.organization_id,
        )

        assert returned == collection_job_id

        jobs = CollectionJobCrud(db, project.id).read_all()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.id == collection_job_id
        assert job.project_id == project.id
        assert job.collection_id == created_collection.id
        assert job.status == CollectionJobStatus.PENDING
        assert job.action_type == CollectionActionType.DELETE

        mock_schedule.assert_called_once()
        kwargs = mock_schedule.call_args.kwargs
        assert kwargs["project_id"] == project.id
        assert kwargs["organization_id"] == project.organization_id
        assert kwargs["job_id"] == str(job.id)
        assert kwargs["collection_id"] == str(created_collection.id)
        assert kwargs["request"] == req.model_dump(mode="json")
        assert "trace_id" in kwargs


@patch("app.services.collections.delete_collection.get_llm_provider")
def test_execute_job_delete_success_updates_job_and_calls_delete(
    mock_get_llm_provider: MagicMock, db
) -> None:
    """
    - execute_job should set task_id on the CollectionJob
    - call provider.delete() to delete remote resources
    - delete local record via CollectionCrud.delete_by_id(...)
    - mark job successful and clear error_message
    """
    project = get_project(db)

    collection = get_vector_store_collection(db, project, vector_store_id="asst_123")

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_provider = MagicMock()
    mock_provider.delete = MagicMock()
    mock_get_llm_provider.return_value = mock_provider

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        collection_crud_instance = MockCollectionCrud.return_value
        collection_crud_instance.read_one.return_value = collection

        task_id = uuid4()
        req = DeletionRequest(collection_id=collection.id)

        execute_job(
            request=req.model_dump(mode="json"),
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(task_id),
            job_id=str(job.id),
            collection_id=str(collection.id),
            task_instance=None,
        )

        updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
        assert updated_job.task_id == str(task_id)
        assert updated_job.status == CollectionJobStatus.SUCCESSFUL
        assert updated_job.error_message in (None, "")

        mock_provider.delete.assert_called_once_with(collection)

        MockCollectionCrud.assert_called_with(db, project.id)
        collection_crud_instance.read_one.assert_called_once_with(collection.id)
        collection_crud_instance.delete_by_id.assert_called_once_with(collection.id)

        mock_get_llm_provider.assert_called_once()


@patch("app.services.collections.delete_collection.get_llm_provider")
def test_execute_job_delete_failure_marks_job_failed(
    mock_get_llm_provider: MagicMock, db
) -> None:
    """
    When provider.delete() raises an exception:
    - Job should be marked FAILED
    - error_message should be set
    - Local collection should NOT be deleted
    """
    project = get_project(db)

    collection = get_vector_store_collection(
        db,
        project,
        vector_store_id="vector_123",
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_provider = MagicMock()
    mock_provider.delete.side_effect = Exception("Remote deletion failed")
    mock_get_llm_provider.return_value = mock_provider

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        collection_crud_instance = MockCollectionCrud.return_value
        collection_crud_instance.read_one.return_value = collection

        task_id = uuid4()
        req = DeletionRequest(collection_id=collection.id)

        with pytest.raises(Exception, match="Remote deletion failed"):
            execute_job(
                request=req.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(task_id),
                job_id=str(job.id),
                collection_id=str(collection.id),
                task_instance=None,
            )

        failed_job = CollectionJobCrud(db, project.id).read_one(job.id)
        assert failed_job.task_id == str(task_id)
        assert failed_job.status == CollectionJobStatus.FAILED
        assert (
            failed_job.error_message
            and "Remote deletion failed" in failed_job.error_message
        )

        mock_provider.delete.assert_called_once_with(collection)

        MockCollectionCrud.assert_called_with(db, project.id)
        collection_crud_instance.read_one.assert_called_once_with(collection.id)
        collection_crud_instance.delete_by_id.assert_not_called()

        mock_get_llm_provider.assert_called_once()


@patch("app.services.collections.delete_collection.get_llm_provider")
@patch("app.services.collections.delete_collection.send_callback")
def test_execute_job_delete_success_with_callback_sends_success_payload(
    mock_send_callback: MagicMock,
    mock_get_llm_provider: MagicMock,
    db,
) -> None:
    """
    When deletion succeeds and a callback_url is provided:
    - job is marked SUCCESSFUL
    - send_callback is called once
    - success payload has success=True, status=SUCCESSFUL, and correct collection id
    """
    project = get_project(db)

    collection = get_vector_store_collection(
        db,
        project,
        vector_store_id="vector 123",
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_provider = MagicMock()
    mock_provider.delete = MagicMock()
    mock_get_llm_provider.return_value = mock_provider

    callback_url = "https://example.com/collections/delete-success"

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        collection_crud_instance = MockCollectionCrud.return_value
        collection_crud_instance.read_one.return_value = collection

        task_id = uuid4()
        req = DeletionRequest(collection_id=collection.id, callback_url=callback_url)

        execute_job(
            request=req.model_dump(mode="json"),
            project_id=project.id,
            organization_id=project.organization_id,
            task_id=str(task_id),
            job_id=str(job.id),
            collection_id=str(collection.id),
            task_instance=None,
        )

        updated_job = CollectionJobCrud(db, project.id).read_one(job.id)
        assert updated_job.task_id == str(task_id)
        assert updated_job.status == CollectionJobStatus.SUCCESSFUL
        assert updated_job.error_message in (None, "")

        mock_provider.delete.assert_called_once_with(collection)

        MockCollectionCrud.assert_called_with(db, project.id)
        collection_crud_instance.read_one.assert_called_once_with(collection.id)
        collection_crud_instance.delete_by_id.assert_called_once_with(collection.id)
        mock_get_llm_provider.assert_called_once()

        mock_send_callback.assert_called_once()
        cb_url_arg, payload_arg = mock_send_callback.call_args.args

        assert str(cb_url_arg) == callback_url
        assert payload_arg["success"] is True
        assert payload_arg["data"]["status"] == CollectionJobStatus.SUCCESSFUL
        assert payload_arg["data"]["collection"]["id"] == str(collection.id)
        assert UUID(payload_arg["data"]["job_id"]) == job.id


@patch("app.services.collections.delete_collection.get_llm_provider")
@patch("app.services.collections.delete_collection.send_callback")
def test_execute_job_delete_remote_failure_with_callback_sends_failure_payload(
    mock_send_callback: MagicMock,
    mock_get_llm_provider: MagicMock,
    db,
) -> None:
    """
    When provider.delete() raises AND a callback_url is provided:
    - job is marked FAILED with error_message set
    - send_callback is called once
    - failure payload has success=False, status=FAILED, correct collection id, and error message
    """
    project = get_project(db)

    collection = get_vector_store_collection(
        db,
        project,
        vector_store_id="vector_123",
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_provider = MagicMock()
    mock_provider.delete.side_effect = Exception("Remote deletion failed")
    mock_get_llm_provider.return_value = mock_provider

    callback_url = "https://example.com/collections/delete-failed"

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        collection_crud_instance = MockCollectionCrud.return_value
        collection_crud_instance.read_one.return_value = collection

        task_id = uuid4()
        req = DeletionRequest(collection_id=collection.id, callback_url=callback_url)

        with pytest.raises(Exception, match="Remote deletion failed"):
            execute_job(
                request=req.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(task_id),
                job_id=str(job.id),
                collection_id=str(collection.id),
                task_instance=None,
            )

        failed_job = CollectionJobCrud(db, project.id).read_one(job.id)
        assert failed_job.task_id == str(task_id)
        assert failed_job.status == CollectionJobStatus.FAILED
        assert (
            failed_job.error_message
            and "Remote deletion failed" in failed_job.error_message
        )

        mock_provider.delete.assert_called_once_with(collection)

        MockCollectionCrud.assert_called_with(db, project.id)
        collection_crud_instance.read_one.assert_called_once_with(collection.id)
        collection_crud_instance.delete_by_id.assert_not_called()
        mock_get_llm_provider.assert_called_once()

        mock_send_callback.assert_called_once()
        cb_url_arg, payload_arg = mock_send_callback.call_args.args

        assert str(cb_url_arg) == callback_url
        assert payload_arg["success"] is False
        assert "Remote deletion failed" in (payload_arg["error"] or "")
        assert payload_arg["data"]["status"] == CollectionJobStatus.FAILED
        assert payload_arg["data"]["collection"]["id"] == str(collection.id)
        assert UUID(payload_arg["data"]["job_id"]) == job.id


@patch("app.services.collections.delete_collection.get_llm_provider")
def test_execute_job_local_delete_failure_after_remote_success_marks_failed(
    mock_get_llm_provider: MagicMock, db
) -> None:
    """
    When provider.delete() succeeds but the local CollectionCrud.delete_by_id fails:
    - job should be marked FAILED with error_message set
    - exception is re-raised
    - provider.delete was already called (remote resource is gone)
    """
    project = get_project(db)

    collection = get_vector_store_collection(
        db, project, vector_store_id="vs_local_fail"
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_provider = MagicMock()
    mock_provider.delete = MagicMock()
    mock_get_llm_provider.return_value = mock_provider

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        collection_crud_instance = MockCollectionCrud.return_value
        collection_crud_instance.read_one.return_value = collection
        collection_crud_instance.delete_by_id.side_effect = Exception(
            "Local DB delete failed"
        )

        task_id = uuid4()
        req = DeletionRequest(collection_id=collection.id)

        with pytest.raises(Exception, match="Local DB delete failed"):
            execute_job(
                request=req.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(task_id),
                job_id=str(job.id),
                collection_id=str(collection.id),
                task_instance=None,
            )

        failed_job = CollectionJobCrud(db, project.id).read_one(job.id)
        assert failed_job.task_id == str(task_id)
        assert failed_job.status == CollectionJobStatus.FAILED
        assert (
            failed_job.error_message
            and "Local DB delete failed" in failed_job.error_message
        )

        mock_provider.delete.assert_called_once_with(collection)
        collection_crud_instance.delete_by_id.assert_called_once_with(collection.id)


@patch("app.services.collections.delete_collection.get_llm_provider")
def test_execute_job_provider_factory_failure_marks_job_failed(
    mock_get_llm_provider: MagicMock, db
) -> None:
    """
    When get_llm_provider itself raises (e.g. missing credentials):
    - job should be marked FAILED with error_message set
    - provider.delete is never called
    - local collection is not deleted
    """
    project = get_project(db)

    collection = get_vector_store_collection(
        db, project, vector_store_id="vs_provider_fail"
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_get_llm_provider.side_effect = Exception("Provider credentials missing")

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        collection_crud_instance = MockCollectionCrud.return_value
        collection_crud_instance.read_one.return_value = collection

        task_id = uuid4()
        req = DeletionRequest(collection_id=collection.id)

        with pytest.raises(Exception, match="Provider credentials missing"):
            execute_job(
                request=req.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(task_id),
                job_id=str(job.id),
                collection_id=str(collection.id),
                task_instance=None,
            )

        failed_job = CollectionJobCrud(db, project.id).read_one(job.id)
        assert failed_job.task_id == str(task_id)
        assert failed_job.status == CollectionJobStatus.FAILED
        assert (
            failed_job.error_message
            and "Provider credentials missing" in failed_job.error_message
        )

        collection_crud_instance.delete_by_id.assert_not_called()
        mock_get_llm_provider.assert_called_once()


@patch("app.services.collections.delete_collection.get_llm_provider")
def test_execute_job_timeout_marks_job_failed(
    mock_get_llm_provider: MagicMock, db
) -> None:
    project = get_project(db)

    collection = get_vector_store_collection(db, project, vector_store_id="vs_timeout")

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_provider = MagicMock()
    mock_provider.delete.side_effect = Timeout(300)
    mock_get_llm_provider.return_value = mock_provider

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        MockCollectionCrud.return_value.read_one.return_value = collection

        req = DeletionRequest(collection_id=collection.id)

        with pytest.raises(Timeout):
            execute_job(
                request=req.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                collection_id=str(collection.id),
                task_instance=None,
            )

    failed_job = CollectionJobCrud(db, project.id).read_one(job.id)
    assert failed_job.status == CollectionJobStatus.FAILED
    assert "soft time limit" in (failed_job.error_message or "")

    MockCollectionCrud.return_value.delete_by_id.assert_not_called()


@patch("app.services.collections.delete_collection.get_llm_provider")
@patch("app.services.collections.delete_collection.send_callback")
def test_execute_job_timeout_sends_failure_callback(
    mock_send_callback: MagicMock,
    mock_get_llm_provider: MagicMock,
    db,
) -> None:
    project = get_project(db)
    callback_url = "https://example.com/collections/delete-timeout"

    collection = get_vector_store_collection(
        db, project, vector_store_id="vs_timeout_cb"
    )

    job = get_collection_job(
        db,
        project,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.PENDING,
        collection_id=collection.id,
    )

    mock_provider = MagicMock()
    mock_provider.delete.side_effect = Timeout(300)
    mock_get_llm_provider.return_value = mock_provider

    with patch(
        "app.services.collections.delete_collection.Session"
    ) as SessionCtor, patch(
        "app.services.collections.delete_collection.CollectionCrud"
    ) as MockCollectionCrud:
        SessionCtor.return_value.__enter__.return_value = db
        SessionCtor.return_value.__exit__.return_value = False

        MockCollectionCrud.return_value.read_one.return_value = collection

        req = DeletionRequest(collection_id=collection.id, callback_url=callback_url)

        with pytest.raises(Timeout):
            execute_job(
                request=req.model_dump(mode="json"),
                project_id=project.id,
                organization_id=project.organization_id,
                task_id=str(uuid4()),
                job_id=str(job.id),
                collection_id=str(collection.id),
                task_instance=None,
            )

    mock_send_callback.assert_called_once()
    cb_url_arg, payload_arg = mock_send_callback.call_args.args
    assert str(cb_url_arg) == callback_url
    assert payload_arg["success"] is False
    assert "soft time limit" in (payload_arg["error"] or "")
    assert payload_arg["data"]["status"] == CollectionJobStatus.FAILED
    assert UUID(payload_arg["data"]["job_id"]) == job.id
