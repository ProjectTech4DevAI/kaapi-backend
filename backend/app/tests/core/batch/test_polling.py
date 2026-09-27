"""Tests for poll_batch_status (app/core/batch/polling.py).

Real batch_job rows on the transactional session; only the provider client is stubbed.
"""

from unittest.mock import MagicMock

from sqlmodel import Session

from app.core.batch.polling import poll_batch_status
from app.models.batch_job import BatchJob, BatchJobType
from app.tests.utils.auth import get_user_test_auth_context


def _batch_job(db: Session, **kwargs) -> BatchJob:
    auth = get_user_test_auth_context(db)
    job = BatchJob(
        provider="openai",
        job_type=BatchJobType.ASSESSMENT.value,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        provider_batch_id="batch_abc",
        total_items=1,
        **kwargs,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _provider(status_result: dict) -> MagicMock:
    provider = MagicMock()
    provider.get_batch_status.return_value = status_result
    return provider


class TestPollBatchStatus:
    def test_error_file_id_persists_when_the_status_did_not_change(
        self, db: Session
    ) -> None:
        job = _batch_job(db, provider_status="in_progress")

        poll_batch_status(
            session=db,
            provider=_provider(
                {"provider_status": "in_progress", "error_file_id": "file-err-1"}
            ),
            batch_job=job,
        )

        db.refresh(job)
        assert job.provider_error_file_id == "file-err-1"
        assert job.provider_status == "in_progress"

    def test_unchanged_fields_are_not_written(self, db: Session) -> None:
        job = _batch_job(
            db,
            provider_status="completed",
            provider_output_file_id="file-out",
            provider_error_file_id="file-err",
        )
        before = job.updated_at

        poll_batch_status(
            session=db,
            provider=_provider(
                {
                    "provider_status": "completed",
                    "provider_output_file_id": "file-out",
                    "error_file_id": "file-err",
                }
            ),
            batch_job=job,
        )

        db.refresh(job)
        assert job.updated_at == before

    def test_omitted_field_does_not_null_the_stored_value(self, db: Session) -> None:
        job = _batch_job(
            db, provider_status="in_progress", provider_output_file_id="file-out"
        )

        poll_batch_status(
            session=db,
            provider=_provider({"provider_status": "completed"}),
            batch_job=job,
        )

        db.refresh(job)
        assert job.provider_status == "completed"
        assert job.provider_output_file_id == "file-out"

    def test_status_flip_persists_the_terminal_fields(self, db: Session) -> None:
        job = _batch_job(db, provider_status="in_progress")

        poll_batch_status(
            session=db,
            provider=_provider(
                {
                    "provider_status": "failed",
                    "provider_output_file_id": "file-out",
                    "error_message": "provider rejected the batch",
                }
            ),
            batch_job=job,
        )

        db.refresh(job)
        assert job.provider_status == "failed"
        assert job.provider_output_file_id == "file-out"
        assert job.error_message == "provider rejected the batch"

    def test_status_result_is_returned_verbatim(self, db: Session) -> None:
        # _poll_outcome reads error_file_id off the return value, not off the row.
        job = _batch_job(db, provider_status="in_progress")
        status_result = {
            "provider_status": "completed",
            "error_file_id": "file-err-2",
            "request_counts": {"completed": 3, "failed": 1},
        }

        returned = poll_batch_status(
            session=db, provider=_provider(status_result), batch_job=job
        )

        assert returned == status_result
