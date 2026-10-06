from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from app.core.batch import BatchJobState
from app.crud.tts_evaluations.cron import poll_tts_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJob, BatchJobType
from app.models.job import JobStatus
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
)

GEMINI = "gemini-2.5-pro-preview-tts"


@pytest.fixture
def run(db: Session, user_api_key: TestAuthContext) -> EvaluationRun:
    run = create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[GEMINI, "bulbul:v3"],
    )
    run.status = "processing"
    db.add(run)
    db.commit()
    return run


def _terminal_batch_job(
    db: Session, run: EvaluationRun, state: BatchJobState, error: str | None = None
) -> BatchJob:
    job = BatchJob(
        provider="google-aistudio",
        job_type=BatchJobType.TTS_EVALUATION.value,
        config={"evaluation_run_id": run.id, "tts_provider": GEMINI},
        provider_batch_id="batches/1",
        provider_status=state.value,
        error_message=error,
        total_items=1,
        organization_id=run.organization_id,
        project_id=run.project_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


async def _poll(db: Session, run: EvaluationRun, expected_dispatches: int = 0) -> dict:
    # Gemini is never reached: batch jobs are either terminal or polled via the
    # patched poll_batch_status in the tests that need it.
    with patch(
        "app.crud.tts_evaluations.cron.start_tts_result_processing",
        return_value="celery-task",
    ) as dispatch:
        outcome = await poll_tts_run(
            session=db, run=run, batch_provider=MagicMock(), org_id=run.organization_id
        )
    assert dispatch.call_count == expected_dispatches
    if expected_dispatches:
        assert dispatch.call_args.kwargs["evaluation_run_id"] == run.id
        assert dispatch.call_args.kwargs["tts_provider"] == GEMINI
    return outcome


def _provider_reports(db: Session, state: BatchJobState):
    def fake_poll(*, session: Session, provider: object, batch_job: BatchJob) -> None:
        batch_job.provider_status = state.value
        session.add(batch_job)
        session.commit()

    return patch(
        "app.crud.evaluations.cron_utils.poll_batch_status", side_effect=fake_poll
    )


@pytest.mark.asyncio
class TestPollTTSRunFinalize:
    async def test_succeeded_batch_with_sync_rows_pending_stays_processing(
        self, db: Session, run: EvaluationRun
    ) -> None:
        _terminal_batch_job(db, run, BatchJobState.SUCCEEDED)
        create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)
        create_test_tts_result_row(db, run=run, provider="bulbul:v3")

        outcome = await _poll(db, run)

        assert outcome["action"] == "completed"
        assert outcome["current_status"] == "processing"
        db.refresh(run)
        assert run.status == "processing"

    async def test_all_done_completes_with_failure_count(
        self, db: Session, run: EvaluationRun
    ) -> None:
        _terminal_batch_job(db, run, BatchJobState.SUCCEEDED)
        create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)
        create_test_tts_result_row(
            db, run=run, provider="bulbul:v3", status=JobStatus.FAILED
        )

        outcome = await _poll(db, run)

        assert outcome["current_status"] == "completed"
        assert outcome["error"] == "1 synthesis(es) failed"
        db.refresh(run)
        assert run.status == "completed"
        assert run.error_message == "1 synthesis(es) failed"

    async def test_failed_batch_error_takes_precedence(
        self, db: Session, run: EvaluationRun
    ) -> None:
        _terminal_batch_job(db, run, BatchJobState.FAILED, error="quota exceeded")
        create_test_tts_result_row(db, run=run, status=JobStatus.FAILED)

        outcome = await _poll(db, run)

        assert outcome["action"] == "failed"
        assert outcome["current_status"] == "completed"
        assert outcome["error"] == f"{GEMINI}: quota exceeded"
        db.refresh(run)
        assert run.error_message == f"{GEMINI}: quota exceeded"

    async def test_run_vanishing_before_finalize_falls_back_to_poll_state(
        self, db: Session, run: EvaluationRun
    ) -> None:
        _terminal_batch_job(db, run, BatchJobState.FAILED, error="expired")

        with patch(
            "app.crud.tts_evaluations.cron.finalize_tts_run_status", return_value=None
        ):
            outcome = await _poll(db, run)

        assert outcome["current_status"] == "processing"
        assert outcome["error"] == f"{GEMINI}: expired"


@pytest.mark.asyncio
class TestPollTTSRunDispatch:
    async def test_no_batch_jobs_fails_run(
        self, db: Session, run: EvaluationRun
    ) -> None:
        outcome = await _poll(db, run)

        assert (outcome["action"], outcome["error"]) == (
            "failed",
            "No batch jobs found",
        )
        db.refresh(run)
        assert run.status == "failed"

    async def test_still_running_batch_is_no_change(
        self, db: Session, run: EvaluationRun
    ) -> None:
        _terminal_batch_job(db, run, BatchJobState.RUNNING)

        with _provider_reports(db, BatchJobState.RUNNING):
            outcome = await _poll(db, run)

        assert outcome["action"] == "no_change"
        db.refresh(run)
        assert run.status == "processing"

    async def test_newly_succeeded_batch_dispatches_processing(
        self, db: Session, run: EvaluationRun
    ) -> None:
        _terminal_batch_job(db, run, BatchJobState.RUNNING)
        create_test_tts_result_row(db, run=run)

        with _provider_reports(db, BatchJobState.SUCCEEDED):
            outcome = await _poll(db, run, expected_dispatches=1)

        assert (outcome["action"], outcome["current_status"]) == (
            "dispatched",
            "processing",
        )

    async def test_already_succeeded_with_pending_rows_redispatches(
        self, db: Session, run: EvaluationRun
    ) -> None:
        _terminal_batch_job(db, run, BatchJobState.SUCCEEDED)
        create_test_tts_result_row(db, run=run)

        outcome = await _poll(db, run, expected_dispatches=1)

        assert outcome["action"] == "dispatched"
