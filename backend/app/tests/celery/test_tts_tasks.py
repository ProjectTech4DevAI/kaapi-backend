from typing import Any
from unittest.mock import MagicMock, patch

from asgi_correlation_id import correlation_id

from app.celery.tasks import job_execution
from app.celery.utils import start_tts_sync_chunk

CHUNK_KWARGS: dict[str, Any] = {
    "organization_id": 3,
    "model": "bulbul:v3",
    "result_ids": [11, 12],
    "language_code": "hi-IN",
}


class TestStartTTSSyncChunk:
    def test_enqueues_task_with_kwargs_and_trace_headers(self) -> None:
        # apply_async is the broker boundary; nothing is published in tests.
        with patch.object(
            job_execution.run_tts_sync_chunk,
            "apply_async",
            return_value=MagicMock(id="celery-123"),
        ) as apply_async:
            task_id = start_tts_sync_chunk(
                project_id=7, job_id="42", trace_id="trace-abc", **CHUNK_KWARGS
            )

        assert task_id == "celery-123"
        call = apply_async.call_args.kwargs
        assert call["kwargs"] == {
            "project_id": 7,
            "job_id": "42",
            "trace_id": "trace-abc",
            **CHUNK_KWARGS,
        }
        assert "otel" in call["headers"]

    def test_trace_id_defaults_to_na(self) -> None:
        with patch.object(
            job_execution.run_tts_sync_chunk,
            "apply_async",
            return_value=MagicMock(id="celery-123"),
        ) as apply_async:
            start_tts_sync_chunk(project_id=7, job_id="42", **CHUNK_KWARGS)

        assert apply_async.call_args.kwargs["kwargs"]["trace_id"] == "N/A"


class TestRunTTSSyncChunk:
    def test_runs_chunk_with_task_context_and_trace(self) -> None:
        seen: dict[str, Any] = {}

        def fake_execute(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            seen["correlation_id"] = correlation_id.get()
            return {"success": True, "processed": 2}

        with patch(
            "app.services.tts_evaluations.sync_generation.execute_tts_sync_chunk",
            side_effect=fake_execute,
        ):
            result = job_execution.run_tts_sync_chunk.apply(
                kwargs={
                    "project_id": 7,
                    "job_id": "42",
                    "trace_id": "trace-abc",
                    **CHUNK_KWARGS,
                },
                task_id="task-xyz",
            ).get()

        assert result == {"success": True, "processed": 2}
        assert seen["project_id"] == 7
        assert seen["job_id"] == "42"
        assert seen["task_id"] == "task-xyz"
        assert seen["task_instance"] is not None
        assert seen["correlation_id"] == "trace-abc"
        for key, value in CHUNK_KWARGS.items():
            assert seen[key] == value

    def test_task_is_on_default_queue_at_low_priority(self) -> None:
        assert job_execution.run_tts_sync_chunk.queue == "default"
        assert job_execution.run_tts_sync_chunk.priority == 2
