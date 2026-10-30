from unittest.mock import MagicMock, patch

import pytest

from app.celery import utils
from app.celery.tasks import job_execution

JOB_KWARGS = {"project_id": 1, "job_id": "job-1", "trace_id": "trace-1"}


@pytest.mark.parametrize(
    ("starter", "task_name", "call_kwargs", "expected_kwargs"),
    [
        ("start_llm_job", "run_llm_job", JOB_KWARGS, JOB_KWARGS),
        ("start_llm_chain_job", "run_llm_chain_job", JOB_KWARGS, JOB_KWARGS),
        ("start_guardrails_job", "run_guardrails_job", JOB_KWARGS, JOB_KWARGS),
        ("start_response_job", "run_response_job", JOB_KWARGS, JOB_KWARGS),
        ("start_doctransform_job", "run_doctransform_job", JOB_KWARGS, JOB_KWARGS),
        (
            "start_collection_setup_job",
            "run_collection_setup_job",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_collection_batch_job",
            "run_collection_batch_job",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_delete_collection_job",
            "run_delete_collection_job",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_evaluation_batch_submission",
            "run_evaluation_batch_submission",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_prompt_improvement",
            "run_prompt_improvement",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_stt_batch_submission",
            "run_stt_batch_submission",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_stt_metric_computation",
            "run_stt_metric_computation",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_tts_batch_submission",
            "run_tts_batch_submission",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_tts_result_processing",
            "run_tts_result_processing",
            JOB_KWARGS,
            JOB_KWARGS,
        ),
        (
            "start_fast_evaluation_chunk",
            "run_evaluation_fast_chunk",
            {"eval_run_id": 7, "chunk_index": 2},
            {"eval_run_id": 7, "chunk_index": 2, "trace_id": "N/A"},
        ),
        (
            "start_fast_evaluation_aggregate",
            "run_evaluation_fast_aggregate",
            {"eval_run_id": 7},
            {"eval_run_id": 7, "trace_id": "N/A"},
        ),
        (
            "start_evaluation_iteration_round",
            "run_evaluation_iteration_graph_step",
            {
                "iteration_run_id": 3,
                "resume": True,
                "organization_id": 1,
                "project_id": 2,
            },
            {
                "iteration_run_id": 3,
                "resume": True,
                "organization_id": 1,
                "project_id": 2,
                "trace_id": "N/A",
            },
        ),
    ],
)
def test_start_helper_enqueues_task_and_returns_its_id(
    starter: str,
    task_name: str,
    call_kwargs: dict[str, object],
    expected_kwargs: dict[str, object],
) -> None:
    task = getattr(job_execution, task_name)
    with patch.object(
        task, "apply_async", return_value=MagicMock(id="task-1")
    ) as apply_async:
        task_id = getattr(utils, starter)(**call_kwargs)

    assert task_id == "task-1"
    apply_async.assert_called_once_with(kwargs=expected_kwargs)
