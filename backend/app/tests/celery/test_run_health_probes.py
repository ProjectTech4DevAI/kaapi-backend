from typing import Any
from unittest.mock import patch

from app.celery.tasks import job_execution


def test_run_health_probes_task_returns_execute_result() -> None:
    canned: dict[str, Any] = {
        "elapsed_ms": 42,
        "total": 3,
        "ok": 3,
        "failed": 0,
        "results": [],
    }

    # the task lazy-imports execute_health_probes, so patch it at its source module
    with patch(
        "app.services.health_probes.execute_health_probes",
        return_value=canned,
    ) as execute_mock:
        result = job_execution.run_health_probes.apply(args=[]).get()

    assert result == canned
    execute_mock.assert_called_once_with()
