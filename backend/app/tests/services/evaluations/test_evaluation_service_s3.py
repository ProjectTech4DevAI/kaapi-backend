"""Tests for get_evaluation_with_scores() S3 retrieval."""

from collections.abc import Callable
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from app.models import EvaluationRun
from app.services.evaluations.evaluation import get_evaluation_with_scores


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
    def test_resync_bypasses_cache_and_fetches_langfuse(
        self,
        mock_get_storage: MagicMock,
        mock_load: MagicMock,
        mock_get_eval: MagicMock,
        mock_get_langfuse: MagicMock,
        mock_fetch_langfuse: MagicMock,
        mock_save_score: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """Verify resync=True skips S3/DB and fetches from Langfuse."""
        eval_run = eval_run_factory(
            id=100,
            status="completed",
            score={"summary_scores": [{"name": "accuracy", "avg": 0.9}]},
            score_trace_url="s3://bucket/traces.json",
            dataset_name="test_dataset",
            run_name="test_run",
        )
        mock_get_eval.return_value = eval_run
        mock_get_langfuse.return_value = MagicMock()
        mock_fetch_langfuse.return_value = {
            "summary_scores": [],
            "traces": [{"trace_id": "new"}],
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

        mock_load.assert_not_called()  # S3 skipped
        mock_fetch_langfuse.assert_called_once()  # Langfuse called
