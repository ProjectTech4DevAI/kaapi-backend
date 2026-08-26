"""Tests for the eval-iteration cron resume dispatcher.

`dispatch_pending_evaluation_iteration_resumes` fans a `resume=True` Celery
call out to every thin `evaluation_iteration_run` row still `PROCESSING`; the
Celery enqueue itself is the only external boundary — the DB is real.
"""

from unittest.mock import patch

from sqlmodel import Session

from app.crud.evaluations.cron import dispatch_pending_evaluation_iteration_resumes
from app.crud.evaluations.iteration import (
    create_evaluation_iteration_run,
    update_evaluation_iteration_run,
)
from app.models.evaluation_iteration import (
    EvaluationIterationRunUpdate,
    EvaluationIterationStatusEnum,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)

_CALLBACK_URL = "https://example.com/callback"


def _make_iteration_run(
    db: Session,
    user_api_key: TestAuthContext,
    experiment_name: str,
    status: EvaluationIterationStatusEnum = EvaluationIterationStatusEnum.PROCESSING,
):
    dataset = create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    config = create_test_config(
        db=db, project_id=user_api_key.project_id, use_kaapi_schema=True
    )
    run = create_evaluation_iteration_run(
        session=db,
        dataset_id=dataset.id,
        experiment_name=experiment_name,
        config_id=config.id,
        initial_config_version=1,
        callback_url=_CALLBACK_URL,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    if status != EvaluationIterationStatusEnum.PROCESSING:
        run = update_evaluation_iteration_run(
            session=db,
            iteration_run=run,
            update=EvaluationIterationRunUpdate(status=status),
        )
    return run


class TestDispatchPendingEvaluationIterationResumes:
    def test_dispatches_exactly_one_resume_per_processing_row(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        processing = _make_iteration_run(db, user_api_key, "cron-processing")
        _make_iteration_run(
            db, user_api_key, "cron-completed", EvaluationIterationStatusEnum.COMPLETED
        )
        _make_iteration_run(
            db, user_api_key, "cron-failed", EvaluationIterationStatusEnum.FAILED
        )

        with patch("app.celery.utils.start_evaluation_iteration_round") as mock_start:
            summary = dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_start.assert_called_once_with(
            iteration_run_id=processing.id,
            resume=True,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        assert summary["total"] == 1
        assert summary["resumes_dispatched"] == 1

    def test_no_processing_rows_dispatches_nothing(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        _make_iteration_run(
            db,
            user_api_key,
            "cron-none-completed",
            EvaluationIterationStatusEnum.COMPLETED,
        )

        with patch("app.celery.utils.start_evaluation_iteration_round") as mock_start:
            summary = dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_start.assert_not_called()
        assert summary == {"total": 0, "resumes_dispatched": 0}
