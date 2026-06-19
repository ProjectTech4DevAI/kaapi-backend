"""Tests for save_score() S3 upload functionality."""

from unittest.mock import MagicMock, patch

import pytest

from app.crud.evaluations.core import persist_score_traces, save_score
from app.models import EvaluationRun


class TestSaveScoreS3Upload:
    """Test save_score() S3 upload functionality."""

    @pytest.fixture
    def mock_eval_run(self):
        """Create a mock EvaluationRun."""
        eval_run = MagicMock(spec=EvaluationRun)
        eval_run.id = 100
        return eval_run

    @patch("app.crud.evaluations.core.update_evaluation_run")
    @patch("app.crud.evaluations.core.get_evaluation_run_by_id")
    @patch("app.crud.evaluations.core.upload_jsonl_to_object_store")
    @patch("app.crud.evaluations.core.get_cloud_storage")
    @patch("app.core.db.engine")
    def test_uploads_traces_to_s3_and_stores_summary_only(
        self,
        mock_engine,
        mock_get_storage,
        mock_upload,
        mock_get_eval,
        mock_update,
        mock_eval_run,
    ) -> None:
        """Verify traces uploaded to S3, only summary_scores stored in DB."""
        mock_get_eval.return_value = mock_eval_run
        mock_get_storage.return_value = MagicMock()
        mock_upload.return_value = "s3://bucket/traces.json"

        score = {
            "summary_scores": [{"name": "accuracy", "avg": 0.9}],
            "traces": [{"trace_id": "t1"}],
        }

        save_score(eval_run_id=100, organization_id=1, project_id=1, score=score)

        # Verify upload was called with traces
        mock_upload.assert_called_once()
        assert mock_upload.call_args.kwargs["results"] == [{"trace_id": "t1"}]

        # Verify DB gets summary only, not traces
        update = mock_update.call_args.kwargs["update"]
        assert update.score == {"summary_scores": [{"name": "accuracy", "avg": 0.9}]}
        assert update.score_trace_url == "s3://bucket/traces.json"

    @patch("app.crud.evaluations.core.update_evaluation_run")
    @patch("app.crud.evaluations.core.get_evaluation_run_by_id")
    @patch("app.crud.evaluations.core.upload_jsonl_to_object_store")
    @patch("app.crud.evaluations.core.get_cloud_storage")
    @patch("app.core.db.engine")
    def test_fallback_to_db_when_s3_fails(
        self,
        mock_engine,
        mock_get_storage,
        mock_upload,
        mock_get_eval,
        mock_update,
        mock_eval_run,
    ) -> None:
        """Verify full score stored in DB when S3 upload fails."""
        mock_get_eval.return_value = mock_eval_run
        mock_get_storage.return_value = MagicMock()
        mock_upload.return_value = None  # S3 failed

        score = {
            "summary_scores": [{"name": "accuracy", "avg": 0.9}],
            "traces": [{"trace_id": "t1"}],
        }

        save_score(eval_run_id=100, organization_id=1, project_id=1, score=score)

        # Full score stored in DB as fallback
        update = mock_update.call_args.kwargs["update"]
        assert update.score == score
        assert update.score_trace_url is None

    @patch("app.crud.evaluations.core.update_evaluation_run")
    @patch("app.crud.evaluations.core.get_evaluation_run_by_id")
    @patch("app.crud.evaluations.core.upload_jsonl_to_object_store")
    @patch("app.crud.evaluations.core.get_cloud_storage")
    @patch("app.core.db.engine")
    def test_persist_score_traces_skips_score_column(
        self,
        mock_engine,
        mock_get_storage,
        mock_upload,
        mock_get_eval,
        mock_update,
        mock_eval_run,
    ) -> None:
        """persist_score_traces records the trace pointer but not `score`."""
        mock_get_eval.return_value = mock_eval_run
        mock_get_storage.return_value = MagicMock()
        mock_upload.return_value = "s3://bucket/traces.json"

        persist_score_traces(
            eval_run_id=100,
            organization_id=1,
            project_id=1,
            traces=[{"trace_id": "t1"}],
        )

        # Trace pointer is written, but `score` is left untouched so the run does
        # not surface an empty summary while still processing.
        update = mock_update.call_args.kwargs["update"]
        assert update.score_trace_url == "s3://bucket/traces.json"
        assert "score" not in update.model_dump(exclude_unset=True)

    @patch("app.crud.evaluations.core.update_evaluation_run")
    @patch("app.crud.evaluations.core.get_evaluation_run_by_id")
    @patch("app.crud.evaluations.core.upload_jsonl_to_object_store")
    @patch("app.crud.evaluations.core.get_cloud_storage")
    @patch("app.core.db.engine")
    def test_persist_score_traces_no_score_fallback_when_s3_fails(
        self,
        mock_engine,
        mock_get_storage,
        mock_upload,
        mock_get_eval,
        mock_update,
        mock_eval_run,
    ) -> None:
        """A skeleton persist that misses S3 must not write `score` either."""
        mock_get_eval.return_value = mock_eval_run
        mock_get_storage.return_value = MagicMock()
        mock_upload.return_value = None  # S3 failed

        persist_score_traces(
            eval_run_id=100,
            organization_id=1,
            project_id=1,
            traces=[{"trace_id": "t1"}],
        )

        # No fallback into `score`; cosine stays recoverable via per_item_scores.
        update = mock_update.call_args.kwargs["update"]
        assert "score" not in update.model_dump(exclude_unset=True)
        assert update.score_trace_url is None

    @patch("app.crud.evaluations.core.update_evaluation_run")
    @patch("app.crud.evaluations.core.get_evaluation_run_by_id")
    @patch("app.crud.evaluations.core.upload_jsonl_to_object_store")
    @patch("app.crud.evaluations.core.get_cloud_storage")
    @patch("app.core.db.engine")
    def test_no_s3_upload_when_no_traces(
        self,
        mock_engine,
        mock_get_storage,
        mock_upload,
        mock_get_eval,
        mock_update,
        mock_eval_run,
    ) -> None:
        """Verify S3 upload skipped when traces is empty."""
        mock_get_eval.return_value = mock_eval_run

        score = {"summary_scores": [{"name": "accuracy", "avg": 0.9}], "traces": []}

        save_score(eval_run_id=100, organization_id=1, project_id=1, score=score)

        mock_upload.assert_not_called()
