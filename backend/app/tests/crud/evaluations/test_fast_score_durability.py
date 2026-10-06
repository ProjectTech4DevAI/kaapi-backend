"""Score-unit durability ordering in `run_fast_evaluation`.

`save_score` must land before the completed transition. On a judged (v2) run the
score unit is the only copy of the per-row judge scores — nothing goes to
Langfuse to fall back on — so a crash between the two must leave the run
`processing`, not `completed` with its scores gone.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from app.crud.evaluations.fast import run_fast_evaluation
from app.models import EvaluationRun
from app.models.evaluation import RunModeEnum
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)
from app.tests.utils.utils import random_lower_string

_FAST = "app.crud.evaluations.fast"


@pytest.fixture
def judged_fast_run(db: Session, user_api_key: TestAuthContext) -> EvaluationRun:
    dataset = create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    config = create_test_config(
        db=db, project_id=user_api_key.project_id, use_kaapi_schema=True
    )
    run = EvaluationRun(
        run_name=f"run-{random_lower_string()}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=1,
        status="processing",
        run_mode=RunModeEnum.FAST,
        total_items=0,
        is_judge_run=True,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@pytest.fixture
def _stubbed_stages() -> Any:
    """Collapse stages 1-3 so only the stage 4/5/6 ordering is under test."""
    score = {"summary_scores": [{"name": "Judge", "avg": 4.0}], "overall": None}
    with (
        patch(f"{_FAST}._merge_response_chunks") as merge,
        patch(f"{_FAST}._stage3_score_and_trace") as stage3,
        patch(f"{_FAST}._cleanup_response_chunks"),
        patch(f"{_FAST}._sync_scores_to_langfuse", return_value=True),
    ):
        merge.side_effect = lambda *, session, eval_run: (eval_run, [])
        stage3.side_effect = lambda **kwargs: (kwargs["eval_run"], score, [])
        yield score


def _status(db: Session, eval_run_id: int) -> str:
    run = db.get(EvaluationRun, eval_run_id)
    assert run is not None
    return run.status


def _run(*, db: Session, eval_run: EvaluationRun) -> EvaluationRun:
    return run_fast_evaluation(
        session=db,
        openai_client=MagicMock(),
        langfuse=None,
        eval_run=eval_run,
    )


class TestScoreUnitPersistsBeforeCompletion:
    def test_save_score_sees_a_run_that_is_not_yet_completed(
        self, db: Session, judged_fast_run: EvaluationRun, _stubbed_stages: Any
    ) -> None:
        seen: list[str] = []

        def _capture(*, eval_run_id: int, **_: Any) -> SimpleNamespace:
            seen.append(_status(db, eval_run_id))
            return SimpleNamespace(id=eval_run_id)

        with patch(f"{_FAST}.save_score", side_effect=_capture):
            result = _run(db=db, eval_run=judged_fast_run)

        assert seen == ["processing"]
        assert result.status == "completed"

    def test_a_failing_save_score_leaves_the_run_uncompleted(
        self, db: Session, judged_fast_run: EvaluationRun, _stubbed_stages: Any
    ) -> None:
        with patch(f"{_FAST}.save_score", side_effect=RuntimeError("s3 down")):
            with pytest.raises(RuntimeError, match="s3 down"):
                _run(db=db, eval_run=judged_fast_run)

        db.expire_all()
        assert _status(db, judged_fast_run.id) == "processing"

    def test_a_vanished_run_raises_instead_of_completing(
        self, db: Session, judged_fast_run: EvaluationRun, _stubbed_stages: Any
    ) -> None:
        with patch(f"{_FAST}.save_score", return_value=None):
            with pytest.raises(RuntimeError, match="Score unit not persisted"):
                _run(db=db, eval_run=judged_fast_run)

        db.expire_all()
        assert _status(db, judged_fast_run.id) == "processing"

    def test_the_caller_still_gets_the_full_unit_back(
        self, db: Session, judged_fast_run: EvaluationRun, _stubbed_stages: Any
    ) -> None:
        with patch(
            f"{_FAST}.save_score", return_value=SimpleNamespace(id=judged_fast_run.id)
        ):
            result = _run(db=db, eval_run=judged_fast_run)

        assert result.score == _stubbed_stages

    def test_cleanup_runs_only_after_the_completed_transition(
        self, db: Session, judged_fast_run: EvaluationRun, _stubbed_stages: Any
    ) -> None:
        seen: list[str] = []

        def _capture(*, eval_run: EvaluationRun, **_: Any) -> None:
            seen.append(eval_run.status)

        with (
            patch(
                f"{_FAST}.save_score",
                return_value=SimpleNamespace(id=judged_fast_run.id),
            ),
            patch(f"{_FAST}._cleanup_response_chunks", side_effect=_capture),
        ):
            _run(db=db, eval_run=judged_fast_run)

        assert seen == ["completed"]
