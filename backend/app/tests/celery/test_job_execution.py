from unittest.mock import patch

import pytest

from app.celery.tasks import job_execution

SENTINEL = object()


@pytest.mark.parametrize(
    ("task_name", "service_target"),
    [
        ("run_llm_job", "app.services.llm.jobs.execute_job"),
        ("run_llm_chain_job", "app.services.llm.jobs.execute_chain_job"),
        ("run_response_job", "app.services.response.jobs.execute_job"),
        ("run_guardrails_job", "app.services.guardrails.jobs.execute_job"),
        ("run_doctransform_job", "app.services.doctransform.job.execute_job"),
        (
            "run_collection_setup_job",
            "app.services.collections.create_collection.execute_setup_job",
        ),
        (
            "run_collection_batch_job",
            "app.services.collections.create_collection.execute_batch_job",
        ),
        (
            "run_delete_collection_job",
            "app.services.collections.delete_collection.execute_job",
        ),
        (
            "run_evaluation_batch_submission",
            "app.services.evaluations.batch_job.execute_evaluation_batch_submission",
        ),
        (
            "run_prompt_improvement",
            "app.services.evaluations.prompt_improvement.execute_prompt_improvement",
        ),
        (
            "run_stt_batch_submission",
            "app.services.stt_evaluations.batch_job.execute_batch_submission",
        ),
        (
            "run_stt_metric_computation",
            "app.services.stt_evaluations.metric_job.execute_metric_computation",
        ),
        (
            "run_tts_batch_submission",
            "app.services.tts_evaluations.batch_job.execute_batch_submission",
        ),
        (
            "run_tts_result_processing",
            "app.services.tts_evaluations.batch_result_processing.execute_tts_result_processing",
        ),
    ],
)
class TestJobWrappers:
    def test_returns_service_result(self, task_name, service_target):
        task = getattr(job_execution, task_name)
        with (
            patch.object(job_execution, "_set_trace", lambda trace_id: None),
            patch(service_target, return_value=SENTINEL),
        ):
            result = task(project_id=1, job_id="job-1", trace_id="trace-1")

        assert result is SENTINEL


@pytest.mark.parametrize(
    ("task_name", "service_target", "task_kwargs"),
    [
        (
            "run_evaluation_fast_chunk",
            "app.services.evaluations.fast.execute_fast_evaluation_chunk",
            {"eval_run_id": 7, "chunk_index": 2},
        ),
        (
            "run_evaluation_fast_aggregate",
            "app.services.evaluations.fast.execute_fast_evaluation_aggregate",
            {"eval_run_id": 7},
        ),
        (
            "run_evaluation_iteration_graph_step",
            "app.services.evaluations.iteration_graph.execute_evaluation_iteration_graph_step",
            {
                "iteration_run_id": 3,
                "resume": True,
                "organization_id": 1,
                "project_id": 2,
            },
        ),
    ],
)
def test_evaluation_task_sets_trace_and_returns_service_result(
    task_name, service_target, task_kwargs
):
    task = getattr(job_execution, task_name)
    with patch(service_target, return_value=SENTINEL):
        result = task(trace_id="trace-1", **task_kwargs)

    assert result is SENTINEL
    assert job_execution.correlation_id.get() == "trace-1"


def test_assessment_pipeline_returns_service_result():
    with (
        patch.object(job_execution, "_set_trace", lambda trace_id: None),
        patch(
            "app.services.assessment.tasks.execute_assessment_pipeline",
            return_value=SENTINEL,
        ),
    ):
        result = job_execution.run_assessment_pipeline(
            run_id=1, organization_id=2, project_id=3, trace_id="trace-1"
        )

    assert result is SENTINEL


class TestAssessmentApiBatchReenqueue:
    def _run(self, requeue: bool):
        with (
            patch.object(job_execution, "_set_trace", lambda trace_id: None),
            patch(
                "app.services.assessment.api.batch.run_batch_stage",
                return_value={"requeue": requeue},
            ),
            patch.object(
                job_execution.run_assessment_api_batch, "apply_async"
            ) as apply_async,
        ):
            job_execution.run_assessment_api_batch(
                execution_id=1, organization_id=2, project_id=3, trace_id="trace-1"
            )
        return apply_async

    def test_reenqueue_passes_no_propagate_header(self):
        apply_async = self._run(requeue=True)

        apply_async.assert_called_once()
        assert apply_async.call_args.kwargs["headers"] == {
            "sentry-propagate-traces": False
        }

    def test_no_reenqueue_when_stage_not_requeued(self):
        apply_async = self._run(requeue=False)

        apply_async.assert_not_called()
