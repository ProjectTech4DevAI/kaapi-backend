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

    @patch("app.services.evaluations.evaluation.save_score")
    @patch("app.services.evaluations.evaluation.fetch_trace_scores_from_langfuse")
    @patch("app.services.evaluations.evaluation.get_langfuse_client")
    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.services.evaluations.evaluation.load_json_from_object_store")
    @patch("app.services.evaluations.evaluation.get_cloud_storage")
    def test_resync_never_shrinks_pair_count(
        self,
        mock_get_storage: MagicMock,
        mock_load: MagicMock,
        mock_get_eval: MagicMock,
        mock_get_langfuse: MagicMock,
        mock_fetch_langfuse: MagicMock,
        mock_save_score: MagicMock,
        eval_run_factory: Callable[..., MagicMock],
    ) -> None:
        """A resync that returns fewer traces must not drop cached ones (29 -> 27 -> 29)."""
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
        # Cache has 29 traces; a transient Langfuse hiccup only returns 27.
        mock_load.return_value = [{"trace_id": str(i), "scores": []} for i in range(29)]
        mock_get_langfuse.return_value = MagicMock()
        mock_fetch_langfuse.return_value = {
            "summary_scores": [],
            "traces": [{"trace_id": str(i), "scores": []} for i in range(27)],
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

        saved_score = mock_save_score.call_args.kwargs["score"]
        assert len(saved_score["traces"]) == 29  # stayed at 29, not 27

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
