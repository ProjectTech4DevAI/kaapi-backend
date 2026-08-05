"""Tests for app.services.evaluations.evaluation service functions."""

from collections.abc import Callable
from typing import Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.models import EvaluationRun
from app.services.evaluations.evaluation import (
    get_evaluation_with_scores,
    validate_and_start_batch_evaluation,
)


# A v2 judge run's deterministic run-level overall — not trace-derived, so the
# trace-merge reconstructions must preserve it rather than rebuild it away.
_OVERALL_BLOCK = {
    "overall_score": 3.3,
    "verdict": "Needs Refinement",
    "ai_summary": "The run performed well overall.",
    "breakdown": [
        {
            "name": "Adherence to Ground Truth",
            "key": "ground_truth",
            "score": 4,
            "weight": 0.5,
            "delta": 0.7,
            "verdict": "Good",
        }
    ],
}


class TestGetEvaluationWithScoresS3:
    """Test get_evaluation_with_scores() S3 retrieval."""

    @pytest.fixture
    def eval_run_factory(self) -> Callable[..., MagicMock]:
        """Factory that creates a MagicMock(spec=EvaluationRun) with given attrs."""

        def _factory(
            *,
            id: int,
            status: str,
            score: dict,
            score_trace_url: Optional[str] = None,
            dataset_name: Optional[str] = None,
            run_name: Optional[str] = None,
        ) -> MagicMock:
            eval_run = MagicMock(spec=EvaluationRun)
            eval_run.id = id
            eval_run.status = status
            eval_run.score = score
            eval_run.score_trace_url = score_trace_url
            eval_run.dataset_name = dataset_name
            eval_run.run_name = run_name
            return eval_run

        return _factory

    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.services.evaluations.evaluation.load_json_from_object_store")
    @patch("app.services.evaluations.evaluation.get_cloud_storage")
    def test_loads_traces_from_s3(
        self,
        mock_get_storage: MagicMock,
        mock_load: MagicMock,
        mock_get_eval: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """Verify traces loaded from S3 and score reconstructed."""
        eval_run = eval_run_factory(
            id=100,
            status="completed",
            score={"summary_scores": [{"name": "accuracy", "avg": 0.9}]},
            score_trace_url="s3://bucket/traces.json",
            dataset_name="test_dataset",
            run_name="test_run",
        )
        mock_get_eval.return_value = eval_run
        mock_get_storage.return_value = MagicMock()
        mock_load.return_value = [{"trace_id": "s3_trace"}]

        result, error = get_evaluation_with_scores(
            session=MagicMock(),
            evaluation_id=100,
            organization_id=1,
            project_id=1,
            get_trace_info=True,
            resync_score=False,
        )

        assert error is None
        mock_load.assert_called_once()
        assert result.score["traces"] == [{"trace_id": "s3_trace"}]
        assert result.score["summary_scores"] == [{"name": "accuracy", "avg": 0.9}]

    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.services.evaluations.evaluation.get_cloud_storage")
    def test_returns_db_traces_when_no_s3_url(
        self,
        mock_get_storage: MagicMock,
        mock_get_eval: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """Verify DB traces returned when no S3 URL."""
        eval_run = eval_run_factory(
            id=101,
            status="completed",
            score={
                "summary_scores": [{"name": "accuracy", "avg": 0.85}],
                "traces": [{"trace_id": "db_trace"}],
            },
        )
        mock_get_eval.return_value = eval_run

        result, error = get_evaluation_with_scores(
            session=MagicMock(),
            evaluation_id=101,
            organization_id=1,
            project_id=1,
            get_trace_info=True,
            resync_score=False,
        )

        assert error is None
        mock_get_storage.assert_not_called()
        assert result.score["traces"] == [{"trace_id": "db_trace"}]

    @patch("app.services.evaluations.evaluation.save_score")
    @patch("app.services.evaluations.evaluation.fetch_trace_scores_from_langfuse")
    @patch("app.services.evaluations.evaluation.get_langfuse_client")
    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.services.evaluations.evaluation.load_json_from_object_store")
    @patch("app.services.evaluations.evaluation.get_cloud_storage")
    def test_resync_merges_cache_with_langfuse(
        self,
        mock_get_storage: MagicMock,
        mock_load: MagicMock,
        mock_get_eval: MagicMock,
        mock_get_langfuse: MagicMock,
        mock_fetch_langfuse: MagicMock,
        mock_save_score: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """Verify resync=True re-fetches Langfuse and merges with cached traces."""
        eval_run = eval_run_factory(
            id=100,
            status="completed",
            score={"summary_scores": [{"name": "accuracy", "avg": 0.9}]},
            score_trace_url="s3://bucket/traces.json",
            dataset_name="test_dataset",
            run_name="test_run",
        )
        mock_get_eval.return_value = eval_run
        mock_get_storage.return_value = MagicMock()
        # Cache currently holds one trace; Langfuse returns a different one.
        mock_load.return_value = [{"trace_id": "old", "scores": []}]
        mock_get_langfuse.return_value = MagicMock()
        mock_fetch_langfuse.return_value = {
            "summary_scores": [],
            "traces": [{"trace_id": "new", "scores": []}],
        }
        mock_save_score.return_value = eval_run

        get_evaluation_with_scores(
            session=MagicMock(),
            evaluation_id=100,
            organization_id=1,
            project_id=1,
            get_trace_info=True,
            resync_score=True,
        )

        mock_load.assert_called_once()  # cache read so it can be merged
        mock_fetch_langfuse.assert_called_once()  # Langfuse re-fetched

        # Saved score must be the union of cached + fresh traces (step-forward).
        saved_score = mock_save_score.call_args.kwargs["score"]
        saved_ids = {t["trace_id"] for t in saved_score["traces"]}
        assert saved_ids == {"old", "new"}

    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.services.evaluations.evaluation.load_json_from_object_store")
    @patch("app.services.evaluations.evaluation.get_cloud_storage")
    def test_overall_block_survives_cached_trace_serve(
        self,
        mock_get_storage: MagicMock,
        mock_load: MagicMock,
        mock_get_eval: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """v2 judge run: the run-level `overall` block must survive the cached-serve
        trace reconstruction (the path the frontend hits), not get dropped when the
        score is rebuilt from summary_scores + traces."""
        eval_run = eval_run_factory(
            id=200,
            status="completed",
            score={
                "summary_scores": [{"name": "Adherence to Ground Truth", "avg": 4}],
                "overall": _OVERALL_BLOCK,
            },
            score_trace_url="s3://bucket/traces.json",
            dataset_name="test_dataset",
            run_name="test_run",
        )
        mock_get_eval.return_value = eval_run
        mock_get_storage.return_value = MagicMock()
        mock_load.return_value = [{"trace_id": "t1"}]

        result, error = get_evaluation_with_scores(
            session=MagicMock(),
            evaluation_id=200,
            organization_id=1,
            project_id=1,
            get_trace_info=True,
            resync_score=False,
        )

        assert error is None
        assert result.score["traces"] == [{"trace_id": "t1"}]
        overall = result.score["overall"]
        assert overall["overall_score"] == 3.3
        assert overall["verdict"] == "Needs Refinement"
        assert overall["breakdown"] == _OVERALL_BLOCK["breakdown"]

    @patch("app.services.evaluations.evaluation.save_score")
    @patch("app.services.evaluations.evaluation.fetch_trace_scores_from_langfuse")
    @patch("app.services.evaluations.evaluation.get_langfuse_client")
    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.services.evaluations.evaluation.load_json_from_object_store")
    @patch("app.services.evaluations.evaluation.get_cloud_storage")
    def test_overall_block_survives_resync_merge_and_persists(
        self,
        mock_get_storage: MagicMock,
        mock_load: MagicMock,
        mock_get_eval: MagicMock,
        mock_get_langfuse: MagicMock,
        mock_fetch_langfuse: MagicMock,
        mock_save_score: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """Resync path: the merged score handed to save_score (which re-persists to
        DB) must carry the run-level `overall` block, not just the merged traces."""
        eval_run = eval_run_factory(
            id=201,
            status="completed",
            score={
                "summary_scores": [{"name": "Adherence to Ground Truth", "avg": 4}],
                "overall": _OVERALL_BLOCK,
            },
            score_trace_url="s3://bucket/traces.json",
            dataset_name="test_dataset",
            run_name="test_run",
        )
        mock_get_eval.return_value = eval_run
        mock_get_storage.return_value = MagicMock()
        mock_load.return_value = [{"trace_id": "old", "scores": []}]
        mock_get_langfuse.return_value = MagicMock()
        mock_fetch_langfuse.return_value = {
            "summary_scores": [],
            "traces": [{"trace_id": "new", "scores": []}],
        }
        mock_save_score.return_value = eval_run

        get_evaluation_with_scores(
            session=MagicMock(),
            evaluation_id=201,
            organization_id=1,
            project_id=1,
            get_trace_info=True,
            resync_score=True,
        )

        saved_score = mock_save_score.call_args.kwargs["score"]
        assert {t["trace_id"] for t in saved_score["traces"]} == {"old", "new"}
        assert saved_score["overall"] == _OVERALL_BLOCK

    @patch("app.services.evaluations.evaluation.save_score")
    @patch("app.services.evaluations.evaluation.fetch_trace_scores_from_langfuse")
    @patch("app.services.evaluations.evaluation.get_langfuse_client")
    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.services.evaluations.evaluation.load_json_from_object_store")
    @patch("app.services.evaluations.evaluation.get_cloud_storage")
    def test_resync_skipped_when_cache_unreadable(
        self,
        mock_get_storage: MagicMock,
        mock_load: MagicMock,
        mock_get_eval: MagicMock,
        mock_get_langfuse: MagicMock,
        mock_fetch_langfuse: MagicMock,
        mock_save_score: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """If the cache pointer exists but cannot be read, resync must not overwrite it."""
        eval_run = eval_run_factory(
            id=100,
            status="completed",
            score={"summary_scores": []},
            score_trace_url="s3://bucket/traces.json",
            dataset_name="test_dataset",
            run_name="test_run",
        )
        mock_get_eval.return_value = eval_run
        mock_get_storage.return_value = MagicMock()
        mock_load.side_effect = Exception("S3 unavailable")

        result, error = get_evaluation_with_scores(
            session=MagicMock(),
            evaluation_id=100,
            organization_id=1,
            project_id=1,
            get_trace_info=True,
            resync_score=True,
        )

        assert error is not None
        mock_fetch_langfuse.assert_not_called()  # did not re-fetch
        mock_save_score.assert_not_called()  # did not overwrite the cache


_MODULE = "app.services.evaluations.evaluation"


def _dataset() -> MagicMock:
    dataset = MagicMock()
    dataset.id = 1
    dataset.name = "ds"
    dataset.langfuse_dataset_id = "lf_ds_1"
    dataset.object_store_url = None
    return dataset


def _config(provider: str) -> MagicMock:
    config = MagicMock()
    config.completion.provider = provider
    config.completion.type = "text"
    config.completion.params = {"model": "m"}
    return config


class TestValidateAndStartBatchEvaluation:
    """Test validate_and_start_batch_evaluation provider gating and queueing."""

    @patch(f"{_MODULE}.resolve_evaluation_config")
    @patch(f"{_MODULE}.get_dataset_by_id")
    def test_unsupported_provider_raises_422(
        self, mock_get_dataset, mock_resolve
    ) -> None:
        mock_get_dataset.return_value = _dataset()
        mock_resolve.return_value = (_config("anthropic"), None)

        with pytest.raises(HTTPException) as exc:
            validate_and_start_batch_evaluation(
                session=MagicMock(),
                dataset_id=1,
                experiment_name="exp",
                config_id=uuid4(),
                config_version=1,
                organization_id=1,
                project_id=2,
            )

        assert exc.value.status_code == 422
        assert "not supported" in exc.value.detail

    @patch(f"{_MODULE}.update_evaluation_run")
    @patch(f"{_MODULE}.start_evaluation_batch_submission")
    @patch(f"{_MODULE}.create_evaluation_run_or_409")
    @patch(f"{_MODULE}.resolve_evaluation_config")
    @patch(f"{_MODULE}.get_dataset_by_id")
    def test_queue_failure_marks_run_failed(
        self,
        mock_get_dataset,
        mock_resolve,
        mock_create_run,
        mock_submit,
        mock_update,
    ) -> None:
        """A failure enqueueing the celery task flips the run to failed (no raise)."""
        mock_get_dataset.return_value = _dataset()
        mock_resolve.return_value = (_config("openai"), None)

        eval_run = MagicMock()
        eval_run.id = 99
        mock_create_run.return_value = eval_run
        mock_submit.side_effect = Exception("broker down")
        mock_update.return_value = eval_run

        result = validate_and_start_batch_evaluation(
            session=MagicMock(),
            dataset_id=1,
            experiment_name="exp",
            config_id=uuid4(),
            config_version=1,
            organization_id=1,
            project_id=2,
        )

        assert result is eval_run
        update_arg = mock_update.call_args.kwargs["update"]
        assert update_arg.status == "failed"
        assert "Failed to queue batch submission" in update_arg.error_message

    @patch(f"{_MODULE}.start_evaluation_batch_submission")
    @patch(f"{_MODULE}.create_evaluation_run_or_409")
    @patch(f"{_MODULE}.resolve_evaluation_config")
    @patch(f"{_MODULE}.get_dataset_by_id")
    def test_success_returns_run(
        self, mock_get_dataset, mock_resolve, mock_create_run, mock_submit
    ) -> None:
        mock_get_dataset.return_value = _dataset()
        mock_resolve.return_value = (_config("google-aistudio"), None)
        eval_run = MagicMock()
        mock_create_run.return_value = eval_run
        mock_submit.return_value = "task-1"

        result = validate_and_start_batch_evaluation(
            session=MagicMock(),
            dataset_id=1,
            experiment_name="exp",
            config_id=uuid4(),
            config_version=1,
            organization_id=1,
            project_id=2,
        )

        assert result is eval_run
        mock_submit.assert_called_once()
