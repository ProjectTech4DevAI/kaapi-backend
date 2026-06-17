from datetime import datetime

import pytest

from app.models.evaluation import EvaluationRunPublic, RunModeEnum


def _make(status: str, error_message: str | None) -> EvaluationRunPublic:
    return EvaluationRunPublic(
        id=654,
        run_name="ai_cohort_v2",
        dataset_name="ai_cohort_v2",
        config_id=None,
        config_version=1,
        dataset_id=1,
        batch_job_id=None,
        embedding_batch_job_id=None,
        status=status,
        run_mode=RunModeEnum.FAST,
        object_store_url=None,
        score_trace_url=None,
        total_items=0,
        score=None,
        cost=None,
        error_message=error_message,
        organization_id=1,
        project_id=1,
        inserted_at=datetime(2026, 6, 17),
        updated_at=datetime(2026, 6, 17),
    )


STALE = "Checking failed: EvaluationRun 654 has no batch_job_id"


@pytest.mark.parametrize("status", ["completed", "processing", "pending"])
def test_non_failed_run_suppresses_stale_error_message(status):
    """A completed/in-flight fast run that the batch poller transiently marked
    must not surface the stale error_message to the API."""
    assert _make(status, STALE).error_message is None


def test_failed_run_keeps_error_message():
    assert _make("failed", STALE).error_message == STALE
