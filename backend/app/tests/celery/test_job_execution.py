"""Tests for the LLM/response Celery task wrappers in tasks/job_execution.py.

The service entrypoints are mocked; no DB, OTel provider, or broker is used.
Celery tasks are callable and run synchronously when invoked directly (self is
bound), so we drive the real wrapper — including the gevent_timeout decorator
and suppress_db_instrumentation — end to end.
"""

from unittest.mock import patch

import pytest

from app.celery.tasks import job_execution
from app.core import telemetry

SENTINEL = object()


def _capture_suppression(captured: dict):
    """execute_* stand-in that records whether DB spans are suppressed at call time."""

    def _exec(**kwargs):
        captured["suppressed"] = telemetry._suppress_db_spans_var.get()
        return SENTINEL

    return _exec


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
        captured: dict = {}
        with (
            patch.object(job_execution, "_set_trace", lambda trace_id: None),
            patch(service_target, _capture_suppression(captured)),
        ):
            result = task(project_id=1, job_id="job-1", trace_id="trace-1")

        assert result is SENTINEL

    def test_db_spans_suppressed_during_execution(self, task_name, service_target):
        task = getattr(job_execution, task_name)
        captured: dict = {}
        with (
            patch.object(job_execution, "_set_trace", lambda trace_id: None),
            patch(service_target, _capture_suppression(captured)),
        ):
            task(project_id=1, job_id="job-1", trace_id="trace-1")

        assert captured["suppressed"] is True

    def test_suppression_resets_after_return(self, task_name, service_target):
        task = getattr(job_execution, task_name)
        captured: dict = {}
        with (
            patch.object(job_execution, "_set_trace", lambda trace_id: None),
            patch(service_target, _capture_suppression(captured)),
        ):
            task(project_id=1, job_id="job-1", trace_id="trace-1")

        assert telemetry._suppress_db_spans_var.get() is False
