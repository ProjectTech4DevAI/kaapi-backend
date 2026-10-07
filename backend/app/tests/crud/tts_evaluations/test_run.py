import pytest
from sqlmodel import Session

from app.crud.tts_evaluations.run import finalize_tts_run_status
from app.models import EvaluationRun
from app.models.job import JobStatus
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
)
from app.tests.utils.utils import get_non_existent_id

S, P, F = JobStatus.SUCCESS, JobStatus.PENDING, JobStatus.FAILED


@pytest.fixture
def run(db: Session, user_api_key: TestAuthContext) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


class TestFinalizeTTSRunStatus:
    @pytest.mark.parametrize(
        ("row_statuses", "expected_status", "expected_error"),
        [
            ([S, S], "completed", None),
            ([S, P, F], "processing", "1 synthesis(es) failed"),
            ([S, F, F], "completed", "2 synthesis(es) failed"),
            ([F, F], "failed", "2 synthesis(es) failed"),
            ([], "failed", None),
        ],
        ids=["all_success", "pending", "partial_failure", "all_failed", "no_rows"],
    )
    def test_status_derived_from_rows(
        self,
        db: Session,
        run: EvaluationRun,
        row_statuses: list[JobStatus],
        expected_status: str,
        expected_error: str | None,
    ) -> None:
        for status in row_statuses:
            create_test_tts_result_row(db, run=run, status=status)

        finalize_tts_run_status(session=db, run_id=run.id)

        db.refresh(run)
        assert (run.status, run.error_message) == (expected_status, expected_error)

    def test_explicit_error_message_wins_over_failed_count(
        self, db: Session, run: EvaluationRun
    ) -> None:
        create_test_tts_result_row(db, run=run, status=JobStatus.FAILED)

        finalized = finalize_tts_run_status(
            session=db, run_id=run.id, error_message="gemini: JOB_STATE_FAILED"
        )

        assert finalized.error_message == "gemini: JOB_STATE_FAILED"

    def test_missing_run_returns_none(self, db: Session) -> None:
        missing_id = get_non_existent_id(db, EvaluationRun)

        assert finalize_tts_run_status(session=db, run_id=missing_id) is None
