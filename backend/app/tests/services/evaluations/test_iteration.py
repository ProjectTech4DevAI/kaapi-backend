"""Tests for the eval-iteration kickoff (`validate_and_start_evaluation_iteration`)
and the round-scoring helper (`compute_round_scores`).

The LangGraph loop itself (`iteration_graph.py`) is covered separately in
`test_iteration_graph.py`. External boundary here is the Celery enqueue helper
(`start_evaluation_iteration_round`) — the DB is real.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.evaluations.score import (
    GROUND_TRUTH_SCORE_NAME,
    KNOWLEDGE_BASE_SCORE_NAME,
    PROMPT_SCORE_NAME,
)
from app.models import Config, EvaluationDataset, EvaluationRun
from app.models.evaluation_iteration import (
    EvaluationIterationRun,
    EvaluationIterationStatusEnum,
)
from app.models.llm.request import ConfigBlob, build_kaapi_completion_config
from app.services.evaluations.iteration import (
    compute_round_scores,
    validate_and_start_evaluation_iteration,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)

_CALLBACK_URL = "https://example.com/callback"


def _summary_score(name: str, avg: float) -> dict:
    return {"name": name, "avg": avg}


def _eval_run_with_scores(scores: list[dict]) -> EvaluationRun:
    """A plain (non-persisted) EvaluationRun — compute_round_scores only reads .score/.id."""
    return EvaluationRun(
        id=1,
        run_name="scoring-test",
        dataset_name="d",
        dataset_id=1,
        organization_id=1,
        project_id=1,
        score={"summary_scores": scores},
    )


class TestComputeRoundScores:
    def test_missing_ground_truth_metric_returns_none(self) -> None:
        eval_run = _eval_run_with_scores([_summary_score(PROMPT_SCORE_NAME, 0.8)])
        assert compute_round_scores(eval_run) is None

    def test_missing_prompt_metric_returns_none(self) -> None:
        eval_run = _eval_run_with_scores([_summary_score(GROUND_TRUTH_SCORE_NAME, 0.8)])
        assert compute_round_scores(eval_run) is None

    def test_both_present_computes_mean_stop_score_and_kb_score(self) -> None:
        eval_run = _eval_run_with_scores(
            [
                _summary_score(GROUND_TRUTH_SCORE_NAME, 0.8),
                _summary_score(PROMPT_SCORE_NAME, 0.6),
                _summary_score(KNOWLEDGE_BASE_SCORE_NAME, 0.4),
            ]
        )
        stop_score, kb_score = compute_round_scores(eval_run)
        assert stop_score == pytest.approx(0.7)
        assert kb_score == pytest.approx(0.4)

    def test_kb_score_is_none_when_metric_absent(self) -> None:
        eval_run = _eval_run_with_scores(
            [
                _summary_score(GROUND_TRUTH_SCORE_NAME, 1.0),
                _summary_score(PROMPT_SCORE_NAME, 0.5),
            ]
        )
        stop_score, kb_score = compute_round_scores(eval_run)
        assert stop_score == pytest.approx(0.75)
        assert kb_score is None


def _make_dataset(*, db: Session, user_api_key: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        original_items_count=3,
        duplication_factor=1,
    )


def _make_text_config(db: Session, project_id: int) -> Config:
    blob = ConfigBlob(
        completion=build_kaapi_completion_config(
            provider="openai",
            type="text",
            params={"model": "gpt-4o-iteration-test", "temperature": 0.7},
        )
    )
    return create_test_config(
        db=db, project_id=project_id, use_kaapi_schema=True, config_blob=blob
    )


class TestValidateAndStartEvaluationIteration:
    def test_missing_dataset_raises_404(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        config = _make_text_config(db, user_api_key.project_id)

        with pytest.raises(HTTPException) as exc:
            validate_and_start_evaluation_iteration(
                session=db,
                dataset_id=999_999,
                experiment_name="missing-dataset",
                config_id=config.id,
                config_version=1,
                max_rounds=None,
                callback_url=_CALLBACK_URL,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )
        assert exc.value.status_code == 404

    def test_missing_config_raises_400(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        dataset = _make_dataset(db=db, user_api_key=user_api_key)

        with pytest.raises(HTTPException) as exc:
            validate_and_start_evaluation_iteration(
                session=db,
                dataset_id=dataset.id,
                experiment_name="missing-config",
                config_id=uuid4(),
                config_version=1,
                max_rounds=None,
                callback_url=_CALLBACK_URL,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )
        assert exc.value.status_code == 400

    def test_success_creates_thin_row_and_dispatches_round_1(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        with patch(
            "app.services.evaluations.iteration.start_evaluation_iteration_round",
            return_value="fake-task-id",
        ) as mock_start:
            iteration_run = validate_and_start_evaluation_iteration(
                session=db,
                dataset_id=dataset.id,
                experiment_name="iter-exp",
                config_id=config.id,
                config_version=1,
                max_rounds=None,
                callback_url=_CALLBACK_URL,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )

        assert iteration_run.id is not None
        assert iteration_run.status == EvaluationIterationStatusEnum.PROCESSING
        assert iteration_run.dataset_id == dataset.id
        assert iteration_run.config_id == config.id
        assert iteration_run.initial_config_version == 1
        assert iteration_run.callback_url == _CALLBACK_URL

        mock_start.assert_called_once_with(
            iteration_run_id=iteration_run.id,
            resume=False,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            max_rounds=settings.EVAL_ITERATION_MAX_ROUNDS_DEFAULT,
            config_version=1,
            trace_id="N/A",
        )

        persisted = db.get(EvaluationIterationRun, iteration_run.id)
        assert persisted is not None
        assert persisted.status == EvaluationIterationStatusEnum.PROCESSING

    def test_max_rounds_above_hard_cap_is_clamped(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        with patch(
            "app.services.evaluations.iteration.start_evaluation_iteration_round",
            return_value="fake-task-id",
        ) as mock_start:
            validate_and_start_evaluation_iteration(
                session=db,
                dataset_id=dataset.id,
                experiment_name="iter-clamp",
                config_id=config.id,
                config_version=1,
                max_rounds=settings.EVAL_ITERATION_MAX_ROUNDS_HARD_CAP + 50,
                callback_url=_CALLBACK_URL,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )

        assert (
            mock_start.call_args.kwargs["max_rounds"]
            == settings.EVAL_ITERATION_MAX_ROUNDS_HARD_CAP
        )

    def test_enqueue_failure_marks_row_failed_and_raises_500(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        with patch(
            "app.services.evaluations.iteration.start_evaluation_iteration_round",
            side_effect=RuntimeError("celery down"),
        ):
            with pytest.raises(HTTPException) as exc:
                validate_and_start_evaluation_iteration(
                    session=db,
                    dataset_id=dataset.id,
                    experiment_name="iter-enqueue-fail",
                    config_id=config.id,
                    config_version=1,
                    max_rounds=None,
                    callback_url=_CALLBACK_URL,
                    organization_id=user_api_key.organization_id,
                    project_id=user_api_key.project_id,
                )

        assert exc.value.status_code == 500
        failed = db.exec(
            select(EvaluationIterationRun).where(
                EvaluationIterationRun.experiment_name == "iter-enqueue-fail"
            )
        ).first()
        assert failed is not None
        assert failed.status == EvaluationIterationStatusEnum.FAILED
