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


def _make_run(db: Session, auth: TestAuthContext) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db, organization_id=auth.organization_id, project_id=auth.project_id
    )


class TestFinalizeTTSRunStatus:
    def test_all_success_completes_without_error(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)
        create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)

        finalized = finalize_tts_run_status(session=db, run_id=run.id)

        assert finalized is not None
        assert finalized.status == "completed"
        assert finalized.error_message is None

    def test_pending_rows_keep_run_processing(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)
        create_test_tts_result_row(db, run=run, status=JobStatus.PENDING)

        finalized = finalize_tts_run_status(session=db, run_id=run.id)

        assert finalized.status == "processing"
        db.refresh(run)
        assert run.status == "processing"

    def test_failed_rows_produce_count_message(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)
        create_test_tts_result_row(db, run=run, status=JobStatus.FAILED)
        create_test_tts_result_row(db, run=run, status=JobStatus.FAILED)

        finalized = finalize_tts_run_status(session=db, run_id=run.id)

        assert finalized.status == "completed"
        assert finalized.error_message == "2 synthesis(es) failed"

    def test_explicit_error_message_wins_over_failed_count(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        create_test_tts_result_row(db, run=run, status=JobStatus.FAILED)

        finalized = finalize_tts_run_status(
            session=db, run_id=run.id, error_message="gemini: JOB_STATE_FAILED"
        )

        assert finalized.error_message == "gemini: JOB_STATE_FAILED"

    def test_run_with_no_results_completes(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)

        finalized = finalize_tts_run_status(session=db, run_id=run.id)

        assert finalized.status == "completed"
        assert finalized.error_message is None

    def test_missing_run_returns_none(self, db: Session) -> None:
        missing_id = get_non_existent_id(db, EvaluationRun)

        assert finalize_tts_run_status(session=db, run_id=missing_id) is None
