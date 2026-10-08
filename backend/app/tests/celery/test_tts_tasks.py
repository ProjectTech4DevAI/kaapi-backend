from typing import Any
from unittest.mock import MagicMock, patch

from asgi_correlation_id import correlation_id

from app.celery.tasks import job_execution
from app.celery.utils import start_tts_sync_generation

SYNC_KWARGS: dict[str, Any] = {
    "organization_id": 3,
    "model": "bulbul:v3",
    "language_code": "hi-IN",
}


def test_start_tts_sync_generation_enqueues_task_kwargs() -> None:
    # apply_async is the broker boundary; nothing is published in tests.
    with patch.object(
        job_execution.run_tts_sync_generation,
        "apply_async",
        return_value=MagicMock(id="celery-123"),
    ) as apply_async:
        task_id = start_tts_sync_generation(
            project_id=7, job_id="42", trace_id="trace-abc", **SYNC_KWARGS
        )

    assert task_id == "celery-123"
    assert apply_async.call_args.kwargs["kwargs"] == {
        "project_id": 7,
        "job_id": "42",
        "trace_id": "trace-abc",
        **SYNC_KWARGS,
    }


def test_run_tts_sync_generation_forwards_kwargs_and_trace() -> None:
    seen: dict[str, Any] = {}

    def fake_execute(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs, correlation_id=correlation_id.get())
        return {"success": True, "processed": 2}

    with patch(
        "app.services.tts_evaluations.sync_generation.execute_tts_sync_generation",
        side_effect=fake_execute,
    ):
        result = job_execution.run_tts_sync_generation.apply(
            kwargs={
                "project_id": 7,
                "job_id": "42",
                "trace_id": "trace-abc",
                **SYNC_KWARGS,
            },
            task_id="task-xyz",
        ).get()

    assert result == {"success": True, "processed": 2}
    assert {k: seen[k] for k in ("project_id", "job_id", "task_id")} == {
        "project_id": 7,
        "job_id": "42",
        "task_id": "task-xyz",
    }
    assert {k: seen[k] for k in SYNC_KWARGS} == SYNC_KWARGS
    assert seen["correlation_id"] == "trace-abc"
