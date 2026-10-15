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
