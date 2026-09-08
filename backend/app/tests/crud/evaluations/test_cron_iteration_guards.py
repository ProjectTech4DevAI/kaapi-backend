"""In-flight and zombie guards on the eval-iteration resume dispatcher.

The tick used to fan a resume out to every PROCESSING row unconditionally, so a
graph step slower than the tick got a second one racing it on the same checkpoint
thread, and a row whose sub-job wedged collected resumes forever.

`mark_iteration_run_failed` is patched rather than exercised: it opens its own
`Session(engine)`, which cannot see this test's uncommitted rows.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from sqlmodel import Session

from app.core.config import settings
from app.core.util import now
from app.crud.evaluations.cron import dispatch_pending_evaluation_iteration_resumes
from app.crud.evaluations.iteration import create_evaluation_iteration_run
from app.models.evaluation_iteration import EvaluationIterationRun
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)

_CALLBACK_URL = "https://example.com/callback"
_START = "app.celery.utils.start_evaluation_iteration_round"
_REAPER = "app.services.evaluations.iteration_graph.mark_iteration_run_failed"


def _make_run(
    db: Session,
    user_api_key: TestAuthContext,
    experiment_name: str,
) -> EvaluationIterationRun:
    dataset = create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    config = create_test_config(
        db=db, project_id=user_api_key.project_id, use_kaapi_schema=True
    )
    return create_evaluation_iteration_run(
        session=db,
        dataset_id=dataset.id,
        experiment_name=experiment_name,
        config_id=config.id,
        initial_config_version=1,
        callback_url=_CALLBACK_URL,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


def _backdate(
    db: Session,
    run: EvaluationIterationRun,
    *,
    inserted_at: datetime | None = None,
    last_dispatched_at: datetime | None = None,
) -> EvaluationIterationRun:
    if inserted_at is not None:
        run.inserted_at = inserted_at
    if last_dispatched_at is not None:
        run.last_dispatched_at = last_dispatched_at
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


class TestInFlightGuard:
    def test_a_never_dispatched_loop_is_dispatched_and_stamped(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key, "guard-fresh")
        assert run.last_dispatched_at is None

        with patch(_START) as mock_start:
            summary = dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_start.assert_called_once()
        assert summary["resumes_dispatched"] == 1
        db.refresh(run)
        assert run.last_dispatched_at is not None

    def test_a_loop_dispatched_inside_the_cooldown_is_skipped(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        cooldown = settings.EVAL_ITERATION_DISPATCH_COOLDOWN_MINUTES
        run = _make_run(db, user_api_key, "guard-in-flight")
        stamp = now() - timedelta(minutes=cooldown // 2)
        _backdate(db, run, last_dispatched_at=stamp)

        with patch(_START) as mock_start:
            summary = dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_start.assert_not_called()
        assert summary["total"] == 1
        assert summary["resumes_dispatched"] == 0
        db.refresh(run)
        assert run.last_dispatched_at == stamp

    def test_a_loop_past_the_cooldown_is_dispatched_again(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        cooldown = settings.EVAL_ITERATION_DISPATCH_COOLDOWN_MINUTES
        run = _make_run(db, user_api_key, "guard-cooled-down")
        stale = now() - timedelta(minutes=cooldown + 1)
        _backdate(db, run, last_dispatched_at=stale)

        with patch(_START) as mock_start:
            summary = dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_start.assert_called_once()
        assert summary["resumes_dispatched"] == 1
        db.refresh(run)
        assert run.last_dispatched_at is not None
        assert run.last_dispatched_at > stale


class TestZombieReaper:
    def test_a_loop_past_the_stall_threshold_is_reaped_not_resumed(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key, "guard-zombie")
        _backdate(
            db,
            run,
            inserted_at=now()
            - timedelta(hours=settings.EVAL_ITERATION_STALL_THRESHOLD_HOURS + 1),
        )

        with patch(_START) as mock_start, patch(_REAPER) as mock_reaper:
            summary = dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_start.assert_not_called()
        mock_reaper.assert_called_once()
        kwargs = mock_reaper.call_args.kwargs
        assert kwargs["iteration_run_id"] == run.id
        assert kwargs["organization_id"] == user_api_key.organization_id
        assert kwargs["project_id"] == user_api_key.project_id
        assert "stalled" in kwargs["error_message"]
        assert summary == {"total": 1, "resumes_dispatched": 0}

    def test_a_young_loop_is_never_reaped(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        _make_run(db, user_api_key, "guard-young")

        with patch(_START), patch(_REAPER) as mock_reaper:
            dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_reaper.assert_not_called()

    def test_a_stale_dispatch_stamp_alone_never_reaps(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Only age since kickoff reaps — an old stamp just means it's due again."""
        run = _make_run(db, user_api_key, "guard-stale-stamp")
        _backdate(db, run, last_dispatched_at=now() - timedelta(days=30))

        with patch(_START) as mock_start, patch(_REAPER) as mock_reaper:
            dispatch_pending_evaluation_iteration_resumes(session=db)

        mock_reaper.assert_not_called()
        mock_start.assert_called_once()


class TestSummaryShape:
    def test_summary_keeps_exactly_the_two_documented_keys(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """`test_cron_iteration.py` asserts this dict by equality — keep it closed."""
        _make_run(db, user_api_key, "guard-shape")

        with patch(_START):
            summary = dispatch_pending_evaluation_iteration_resumes(session=db)

        assert set(summary) == {"total", "resumes_dispatched"}
