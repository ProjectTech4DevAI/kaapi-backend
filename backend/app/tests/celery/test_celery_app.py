import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy.pool import QueuePool

celery_app = importlib.import_module("app.celery.celery_app")


def _queue_pool_engine() -> MagicMock:
    pool = MagicMock(spec=QueuePool)
    pool.checkedout.return_value = 1
    pool.size.return_value = 5
    pool.overflow.return_value = 0
    return MagicMock(pool=pool)


class TestPoolStatusSignals:
    def test_prerun_and_postrun_log_pool_state(self, caplog) -> None:
        engine = _queue_pool_engine()
        task = SimpleNamespace(name="run_llm_job")
        with patch("app.core.db.engine", engine), caplog.at_level("INFO"):
            celery_app.log_pool_status(task=task)
            celery_app.log_pool_status_post(task=task)

        messages = [r.getMessage() for r in caplog.records]
        assert any("task=run_llm_job checked_out=1" in m for m in messages)
        assert any("POST task=run_llm_job" in m for m in messages)

    def test_failure_logs_exception_name(self, caplog) -> None:
        engine = _queue_pool_engine()
        with patch("app.core.db.engine", engine), caplog.at_level("WARNING"):
            celery_app.log_pool_status_failure(
                task_id="t1",
                exception=ValueError("x"),
                sender=SimpleNamespace(name="job"),
            )

        assert any(
            "FAILED task=job task_id=t1 exc=ValueError" in r.getMessage()
            for r in caplog.records
        )


class TestWorkerObservabilityInit:
    def test_initialises_sentry_and_telemetry_once(self) -> None:
        init_sentry = MagicMock()
        setup = MagicMock()
        with (
            patch.object(celery_app, "_sentry_initialized", False),
            patch.object(celery_app, "_telemetry_initialized", False),
            patch.object(celery_app, "_flush_hook_registered", False),
            patch.object(
                celery_app.settings, "SENTRY_DSN", "https://k@o.ingest.sentry.io/1"
            ),
            patch("app.core.telemetry.init_sentry", init_sentry),
            patch("app.core.telemetry.setup_telemetry", setup),
            patch.object(celery_app, "configure_logging", MagicMock()),
        ):
            celery_app._initialize_worker_observability()
            celery_app._initialize_worker_observability()

        init_sentry.assert_called_once()
        setup.assert_called_once_with(service_name="kaapi-celery")
