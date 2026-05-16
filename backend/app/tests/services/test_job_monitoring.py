from datetime import timedelta
from unittest.mock import patch

from sqlmodel import Session, delete

from app.core.util import now
from app.models import (
    CollectionActionType,
    CollectionJob,
    CollectionJobStatus,
    DocTransformationJob,
    Document,
    Job,
    JobStatus,
    JobType,
    TransformationStatus,
)
from app.services.job_monitoring import monitor_pending_jobs
from app.tests.utils.test_data import create_test_project


def _table(result: dict, table_name: str) -> dict:
    return next(table for table in result["tables"] if table["table"] == table_name)


def _clear_job_monitor_tables(db: Session) -> None:
    db.exec(delete(DocTransformationJob))
    db.exec(delete(CollectionJob))
    db.exec(delete(Job))
    db.commit()


def test_monitor_pending_jobs_reports_stale_pending_rows(db: Session) -> None:
    _clear_job_monitor_tables(db)
    project = create_test_project(db)
    # Stale window is [now-30min, now-4min]. 20min ago = stale, 1min ago = too fresh.
    stale_at = now() - timedelta(minutes=20)
    fresh_at = now() - timedelta(minutes=1)

    document = Document(
        project_id=project.id,
        fname="source.pdf",
        object_store_url="s3://test/source.pdf",
        inserted_at=stale_at,
        updated_at=stale_at,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    db.add_all(
        [
            Job(
                project_id=project.id,
                job_type=JobType.LLM_API,
                status=JobStatus.PENDING,
                created_at=stale_at,
                updated_at=stale_at,
            ),
            Job(
                project_id=project.id,
                job_type=JobType.RESPONSE,
                status=JobStatus.PENDING,
                created_at=stale_at,
                updated_at=stale_at,
            ),
            Job(
                project_id=project.id,
                job_type=JobType.LLM_API,
                status=JobStatus.PENDING,
                created_at=fresh_at,
                updated_at=fresh_at,
            ),
            CollectionJob(
                project_id=project.id,
                action_type=CollectionActionType.CREATE,
                status=CollectionJobStatus.PENDING,
                inserted_at=stale_at,
                updated_at=stale_at,
            ),
            CollectionJob(
                project_id=project.id,
                action_type=CollectionActionType.DELETE,
                status=CollectionJobStatus.PENDING,
                inserted_at=fresh_at,
                updated_at=fresh_at,
            ),
            DocTransformationJob(
                source_document_id=document.id,
                status=TransformationStatus.PENDING,
                inserted_at=stale_at,
                updated_at=stale_at,
            ),
            DocTransformationJob(
                source_document_id=document.id,
                status=TransformationStatus.PENDING,
                inserted_at=fresh_at,
                updated_at=fresh_at,
            ),
        ]
    )
    db.commit()

    with (
        patch("app.services.job_monitoring.record_stale_pending_jobs") as record_metric,
        patch("app.services.job_monitoring.sentry_sdk.capture_event") as capture_event,
    ):
        result = monitor_pending_jobs(db)

    assert result["status"] == "ok"
    assert result["total_stale_pending"] == 4

    job_summary = _table(result, "job")
    assert job_summary["stale_count"] == 2
    assert {group["job_type"] for group in job_summary["groups"]} == {
        JobType.LLM_API.value,
        JobType.RESPONSE.value,
    }

    collection_summary = _table(result, "collection_jobs")
    assert collection_summary["stale_count"] == 1
    assert collection_summary["groups"][0]["action_type"] == (
        CollectionActionType.CREATE.value
    )

    doc_summary = _table(result, "doc_transformation_job")
    assert doc_summary["stale_count"] == 1

    assert record_metric.called
    assert capture_event.call_count == 3
    events = [call.args[0] for call in capture_event.call_args_list]
    by_table = {event["contexts"]["pending_jobs"]["table"]: event for event in events}
    assert set(by_table) == {"job", "collection_jobs", "doc_transformation_job"}
    assert by_table["job"]["fingerprint"] == ["pending-jobs", "job", "PENDING"]
    assert by_table["job"]["tags"]["job.table"] == "job"
    assert by_table["job"]["contexts"]["pending_jobs"]["stale_count"] == 2
    assert by_table["collection_jobs"]["contexts"]["pending_jobs"]["stale_count"] == 1
    assert (
        by_table["doc_transformation_job"]["contexts"]["pending_jobs"]["stale_count"]
        == 1
    )
    for event in events:
        assert event["message"] == (
            "Some jobs have been pending for longer than expected"
        )


def test_monitor_pending_jobs_ignores_fresh_and_non_pending_rows(
    db: Session,
) -> None:
    _clear_job_monitor_tables(db)
    project = create_test_project(db)
    # Stale window is [now-30min, now-4min]. 20min ago = stale, 1min ago = too fresh.
    stale_at = now() - timedelta(minutes=20)
    fresh_at = now() - timedelta(minutes=1)

    document = Document(
        project_id=project.id,
        fname="source.pdf",
        object_store_url="s3://test/source.pdf",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    db.add_all(
        [
            Job(
                project_id=project.id,
                job_type=JobType.LLM_API,
                status=JobStatus.PENDING,
                created_at=fresh_at,
                updated_at=fresh_at,
            ),
            Job(
                project_id=project.id,
                job_type=JobType.LLM_API,
                status=JobStatus.PROCESSING,
                created_at=stale_at,
                updated_at=stale_at,
            ),
            Job(
                project_id=project.id,
                job_type=JobType.LLM_API,
                status=JobStatus.SUCCESS,
                created_at=stale_at,
                updated_at=stale_at,
            ),
            CollectionJob(
                project_id=project.id,
                action_type=CollectionActionType.CREATE,
                status=CollectionJobStatus.PENDING,
                inserted_at=fresh_at,
                updated_at=fresh_at,
            ),
            CollectionJob(
                project_id=project.id,
                action_type=CollectionActionType.CREATE,
                status=CollectionJobStatus.SUCCESSFUL,
                inserted_at=stale_at,
                updated_at=stale_at,
            ),
            DocTransformationJob(
                source_document_id=document.id,
                status=TransformationStatus.PENDING,
                inserted_at=fresh_at,
                updated_at=fresh_at,
            ),
            DocTransformationJob(
                source_document_id=document.id,
                status=TransformationStatus.COMPLETED,
                inserted_at=stale_at,
                updated_at=stale_at,
            ),
        ]
    )
    db.commit()

    with (
        patch("app.services.job_monitoring.record_stale_pending_jobs"),
        patch("app.services.job_monitoring.sentry_sdk.capture_event") as capture_event,
    ):
        result = monitor_pending_jobs(db)

    assert result["total_stale_pending"] == 0
    assert _table(result, "job")["stale_count"] == 0
    assert _table(result, "collection_jobs")["stale_count"] == 0
    assert _table(result, "doc_transformation_job")["stale_count"] == 0
    capture_event.assert_not_called()


def test_monitor_pending_jobs_ignores_rows_older_than_upper_threshold(
    db: Session,
) -> None:
    _clear_job_monitor_tables(db)
    project = create_test_project(db)
    ancient_at = now() - timedelta(minutes=45)

    document = Document(
        project_id=project.id,
        fname="source.pdf",
        object_store_url="s3://test/source.pdf",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    db.add_all(
        [
            Job(
                project_id=project.id,
                job_type=JobType.LLM_API,
                status=JobStatus.PENDING,
                created_at=ancient_at,
                updated_at=ancient_at,
            ),
            CollectionJob(
                project_id=project.id,
                action_type=CollectionActionType.CREATE,
                status=CollectionJobStatus.PENDING,
                inserted_at=ancient_at,
                updated_at=ancient_at,
            ),
            DocTransformationJob(
                source_document_id=document.id,
                status=TransformationStatus.PENDING,
                inserted_at=ancient_at,
                updated_at=ancient_at,
            ),
        ]
    )
    db.commit()

    with (
        patch("app.services.job_monitoring.record_stale_pending_jobs"),
        patch("app.services.job_monitoring.sentry_sdk.capture_event") as capture_event,
    ):
        result = monitor_pending_jobs(db)

    assert result["total_stale_pending"] == 0
    capture_event.assert_not_called()
