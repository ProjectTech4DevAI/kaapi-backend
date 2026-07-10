from unittest.mock import patch

from sqlmodel import Session

from app.celery.tasks import job_execution


class _NonClosingSession:
    """Wrapper so the task's `with Session(engine)` uses the transactional
    conftest session and doesn't close it."""

    def __init__(self, session: Session):
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_run_health_probes_task_returns_run_probes_result(db: Session):
    canned = {
        "elapsed_ms": 42,
        "total": 3,
        "ok": 3,
        "failed": 0,
        "results": [
            {
                "endpoint": "llm/call",
                "provider": "openai",
                "modality": "text",
                "model": "gpt-4o-1-mini",
                "ok": True,
                "latency_ms": 10,
                "error": None,
            }
        ],
    }

    captured: dict = {}

    def _fake_run_probes(*, session: Session) -> dict:
        captured["session"] = session
        return canned

    with (
        patch(
            "app.services.health_probes.run_probes",
            side_effect=_fake_run_probes,
        ),
        patch(
            "app.core.db.engine",
            new=object(),  # engine is only fed to Session() which we intercept below
        ),
        patch(
            "sqlmodel.Session",
            side_effect=lambda _engine: _NonClosingSession(db),
        ),
    ):
        result = job_execution.run_health_probes.apply(args=[]).get()

    assert result == canned
    assert captured["session"] is db


def test_run_health_probes_task_skipped_when_settings_unset(db: Session):
    """When run_probes reports skipped, the task returns the skip payload verbatim."""
    skipped = {
        "skipped": True,
        "reason": "health_probe_org_or_project_not_set",
    }

    with (
        patch(
            "app.services.health_probes.run_probes",
            return_value=skipped,
        ),
        patch(
            "app.core.db.engine",
            new=object(),
        ),
        patch(
            "sqlmodel.Session",
            side_effect=lambda _engine: _NonClosingSession(db),
        ),
    ):
        result = job_execution.run_health_probes.apply(args=[]).get()

    assert result == skipped
