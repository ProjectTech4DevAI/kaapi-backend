from collections.abc import Iterator
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from app.core.batch import BatchJobState
from app.crud.tts_evaluations.cron import (
    poll_all_pending_tts_evaluations,
    poll_tts_run,
)
from app.models import EvaluationRun
from app.models.batch_job import BatchJob
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.core.config import settings
from app.services.tts_evaluations.constants import TTS_SYNC_STALE_GRACE_SECONDS
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_batch_job,
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
)

MODULE = "app.crud.tts_evaluations.cron"
GEMINI = "gemini-2.5-pro-preview-tts"
SARVAM = "bulbul:v3"


def _processing_run(
    db: Session, auth: TestAuthContext, models: list[str], stale: bool = False
) -> EvaluationRun:
    run = create_test_tts_run_with_dataset(
        db,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        models=models,
    )
    run.status = "processing"
    if stale:
        run.inserted_at -= timedelta(
            seconds=settings.CELERY_TASK_TIME_LIMIT + TTS_SYNC_STALE_GRACE_SECONDS + 60
        )
    db.add(run)
    db.commit()
    return run


@pytest.fixture
def dispatch() -> Iterator[MagicMock]:
    # Celery enqueue of result processing is the boundary.
    with patch(
        f"{MODULE}.start_tts_result_processing", return_value="celery-task"
    ) as dispatch:
        yield dispatch


@pytest.fixture
def gemini_client() -> Iterator[MagicMock]:
    with patch(f"{MODULE}.GeminiClient") as client:
        yield client


def _gemini_reports(state: BatchJobState | Exception) -> Any:
    """Stand-in for the Gemini status call made while polling a batch job."""

    def fake_poll(*, session: Session, batch_job: BatchJob, **_kwargs: Any) -> None:
        if isinstance(state, Exception):
            raise state
        batch_job.provider_status = state.value
        session.add(batch_job)
        session.commit()

    return patch(
        "app.crud.evaluations.cron_utils.poll_batch_status", side_effect=fake_poll
    )


async def _poll(db: Session, run: EvaluationRun) -> dict[str, Any]:
    return await poll_tts_run(session=db, run=run, org_id=run.organization_id)


def _refresh(db: Session, *rows: TTSResult | EvaluationRun) -> None:
    for row in rows:
        db.refresh(row)


@pytest.mark.asyncio
@pytest.mark.usefixtures("gemini_client")
class TestPollTTSRunGeminiBatches:
    @pytest.mark.parametrize(
        "initial_state", [BatchJobState.RUNNING, BatchJobState.SUCCEEDED]
    )
    async def test_succeeded_batch_dispatches_result_processing(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        dispatch: MagicMock,
        initial_state: BatchJobState,
    ) -> None:
        run = _processing_run(db, user_api_key, [GEMINI])
        batch_job = create_test_tts_batch_job(
            db, run=run, provider_status=initial_state.value
        )
        create_test_tts_result_row(db, run=run, provider=GEMINI)

        with _gemini_reports(BatchJobState.SUCCEEDED):
            outcome = await _poll(db, run)

        assert (outcome["action"], outcome["current_status"]) == (
            "dispatched",
            "processing",
        )
        dispatched = dispatch.call_args.kwargs
        assert (
            dispatched["evaluation_run_id"],
            dispatched["tts_provider"],
            dispatched["provider_batch_id"],
        ) == (run.id, GEMINI, batch_job.provider_batch_id)

    async def test_succeeded_batch_with_no_pending_rows_is_not_reprocessed(
        self, db: Session, user_api_key: TestAuthContext, dispatch: MagicMock
    ) -> None:
        run = _processing_run(db, user_api_key, [GEMINI])
        create_test_tts_batch_job(
            db, run=run, provider_status=BatchJobState.SUCCEEDED.value
        )
        create_test_tts_result_row(
            db, run=run, provider=GEMINI, status=JobStatus.SUCCESS
        )

        outcome = await _poll(db, run)

        dispatch.assert_not_called()
        assert outcome["action"] == "completed"
        _refresh(db, run)
        assert run.status == "completed"

    @pytest.mark.parametrize(
        ("state", "batch_error", "expected_error"),
        [
            (
                BatchJobState.FAILED,
                "quota exceeded",
                "Gemini batch ended: quota exceeded",
            ),
            (BatchJobState.CANCELLED, None, "Gemini batch ended: JOB_STATE_CANCELLED"),
            (BatchJobState.EXPIRED, None, "Gemini batch ended: JOB_STATE_EXPIRED"),
        ],
    )
    async def test_unsuccessful_batch_fails_only_that_models_rows(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        dispatch: MagicMock,
        state: BatchJobState,
        batch_error: str | None,
        expected_error: str,
    ) -> None:
        run = _processing_run(db, user_api_key, [GEMINI, SARVAM])
        create_test_tts_batch_job(
            db, run=run, provider_status=state.value, error_message=batch_error
        )
        gemini_row = create_test_tts_result_row(db, run=run, provider=GEMINI)
        sarvam_row = create_test_tts_result_row(db, run=run, provider=SARVAM)

        outcome = await _poll(db, run)

        _refresh(db, gemini_row, sarvam_row, run)
        assert (gemini_row.status, gemini_row.error_message) == (
            JobStatus.FAILED.value,
            expected_error,
        )
        assert sarvam_row.status == JobStatus.PENDING.value
        assert run.status == outcome["current_status"] == "processing"

    async def test_running_batch_leaves_rows_pending(
        self, db: Session, user_api_key: TestAuthContext, dispatch: MagicMock
    ) -> None:
        run = _processing_run(db, user_api_key, [GEMINI])
        create_test_tts_batch_job(
            db, run=run, provider_status=BatchJobState.PENDING.value
        )
        row = create_test_tts_result_row(db, run=run, provider=GEMINI)

        with _gemini_reports(BatchJobState.RUNNING):
            outcome = await _poll(db, run)

        assert outcome["action"] == "no_change"
        dispatch.assert_not_called()
        _refresh(db, row)
        assert row.status == JobStatus.PENDING.value

    async def test_gemini_client_failure_fails_only_gemini_rows(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        gemini_client: MagicMock,
    ) -> None:
        gemini_client.from_credentials.side_effect = ValueError("no gemini key")
        run = _processing_run(db, user_api_key, [GEMINI, SARVAM])
        create_test_tts_batch_job(
            db, run=run, provider_status=BatchJobState.RUNNING.value
        )
        gemini_row = create_test_tts_result_row(db, run=run, provider=GEMINI)
        sarvam_row = create_test_tts_result_row(db, run=run, provider=SARVAM)

        await _poll(db, run)

        _refresh(db, gemini_row, sarvam_row)
        assert (gemini_row.status, gemini_row.error_message) == (
            JobStatus.FAILED.value,
            "Gemini client initialization failed: no gemini key",
        )
        assert sarvam_row.status == JobStatus.PENDING.value

    async def test_batch_job_never_created_is_skipped_and_run_finalizes(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        # Regression: a job whose creation failed (no provider_batch_id) was polled
        # each tick, and the poll error kept the run stuck in processing.
        run = _processing_run(db, user_api_key, [GEMINI, SARVAM])
        create_test_tts_batch_job(
            db, run=run, provider_status="failed", provider_batch_id=None
        )
        create_test_tts_result_row(
            db, run=run, provider=GEMINI, status=JobStatus.FAILED
        )
        create_test_tts_result_row(
            db, run=run, provider=SARVAM, status=JobStatus.SUCCESS
        )

        with _gemini_reports(ValueError("batch name must not be None")):
            outcome = await _poll(db, run)

        assert outcome["action"] == "completed"
        _refresh(db, run)
        assert (run.status, run.error_message) == (
            "completed",
            "1 synthesis(es) failed",
        )


@pytest.mark.asyncio
class TestPollTTSRunStaleModels:
    @pytest.mark.parametrize(
        ("model", "expected_error"),
        [(SARVAM, "Synthesis did not complete"), (GEMINI, "No batch jobs found")],
    )
    @pytest.mark.parametrize("stale", [False, True], ids=["fresh", "stale"])
    async def test_pending_rows_fail_only_after_stale_window(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        model: str,
        expected_error: str,
        stale: bool,
    ) -> None:
        run = _processing_run(db, user_api_key, [model], stale=stale)
        row = create_test_tts_result_row(db, run=run, provider=model)

        await _poll(db, run)

        _refresh(db, row, run)
        if stale:
            assert (row.status, row.error_message) == (
                JobStatus.FAILED.value,
                expected_error,
            )
            assert run.status == "failed"
        else:
            assert row.status == JobStatus.PENDING.value
            assert run.status == "processing"


@pytest.mark.asyncio
class TestPollAllPendingTTSEvaluations:
    async def test_finalizes_sync_only_run_without_batch_job(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _processing_run(db, user_api_key, [SARVAM])
        assert run.batch_job_id is None
        create_test_tts_result_row(
            db, run=run, provider=SARVAM, status=JobStatus.SUCCESS
        )

        summary = await poll_all_pending_tts_evaluations(session=db)

        [detail] = [d for d in summary["details"] if d["run_id"] == run.id]
        assert detail["action"] == "completed"
        _refresh(db, run)
        assert run.status == "completed"
