"""Tests for get_evaluation_with_scores() S3 retrieval."""

from unittest.mock import MagicMock, patch

import pytest

from app.models import EvaluationRun
from app.services.evaluations.evaluation import get_evaluation_with_scores


class TestGetEvaluationWithScoresS3:
    """Test get_evaluation_with_scores() S3 retrieval."""

    @pytest.fixture
    def completed_eval_run_with_s3(self):
        """Completed eval run with S3 URL."""
        eval_run = MagicMock(spec=EvaluationRun)
        eval_run.id = 100
        eval_run.status = "completed"
        eval_run.score = {"summary_scores": [{"name": "accuracy", "avg": 0.9}]}
        eval_run.score_trace_url = "s3://bucket/traces.json"
        eval_run.dataset_name = "test_dataset"
        eval_run.run_name = "test_run"
        return eval_run

    @pytest.fixture
    def completed_eval_run_with_db_traces(self):
        """Completed eval run with traces in DB."""
        eval_run = MagicMock(spec=EvaluationRun)
        eval_run.id = 101
        eval_run.status = "completed"
        eval_run.score = {
            "summary_scores": [{"name": "accuracy", "avg": 0.85}],
            "traces": [{"trace_id": "db_trace"}],
        }
        eval_run.score_trace_url = None
        return eval_run

    @patch("app.services.evaluations.evaluation.get_evaluation_run_by_id")
    @patch("app.core.storage_utils.load_json_from_object_store")
    @patch("app.core.cloud.storage.get_cloud_storage")
    def test_loads_traces_from_s3(
        self, mock_get_storage, mock_load, mock_get_eval, completed_eval_run_with_s3
    ) -> None:
        """Verify traces loaded from S3 and score reconstructed."""
        mock_get_eval.return_value = completed_eval_run_with_s3
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
    @patch("app.core.cloud.storage.get_cloud_storage")
    def test_returns_db_traces_when_no_s3_url(
        self, mock_get_storage, mock_get_eval, completed_eval_run_with_db_traces
    ) -> None:
        """Verify DB traces returned when no S3 URL."""
        mock_get_eval.return_value = completed_eval_run_with_db_traces

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
    @patch("app.core.storage_utils.load_json_from_object_store")
    @patch("app.core.cloud.storage.get_cloud_storage")
    def test_resync_bypasses_cache_and_fetches_langfuse(
        self,
        mock_get_storage,
        mock_load,
        mock_get_eval,
        mock_get_langfuse,
        mock_fetch_langfuse,
        mock_save_score,
        completed_eval_run_with_s3,
    ) -> None:
        """Verify resync=True skips S3/DB and fetches from Langfuse."""
        mock_get_eval.return_value = completed_eval_run_with_s3
        mock_get_langfuse.return_value = MagicMock()
        mock_fetch_langfuse.return_value = {"summary_scores": [], "traces": [{"trace_id": "new"}]}
        mock_save_score.return_value = completed_eval_run_with_s3

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
