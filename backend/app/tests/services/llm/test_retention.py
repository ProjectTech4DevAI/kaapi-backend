from datetime import datetime, timedelta

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.services.llm import retention
from app.services.llm.retention import redact_aged_llm_calls


def _stub_redact(
    monkeypatch: pytest.MonkeyPatch, calls: list[dict], returns: int = 0
) -> None:
    def fake_redact(**kwargs) -> int:
        calls.append(kwargs)
        return returns

    monkeypatch.setattr(retention, "redact_llm_calls", fake_redact)


@pytest.fixture
def redact_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    _stub_redact(monkeypatch, calls)
    return calls


def test_reports_rows_redacted_from_a_single_crud_call(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    _stub_redact(monkeypatch, calls, returns=4137)

    result = redact_aged_llm_calls(session=db)

    assert result.rows_redacted == 4137
    assert len(calls) == 1


def test_no_matching_rows(db: Session, redact_calls: list[dict]) -> None:
    result = redact_aged_llm_calls(session=db)

    assert result.rows_redacted == 0
    assert len(redact_calls) == 1


def test_cutoff_is_naive_utc(db: Session, redact_calls: list[dict]) -> None:
    # llm_call.updated_at is a naive TIMESTAMP; a tz-aware cutoff would raise on compare.
    result = redact_aged_llm_calls(session=db)

    cutoff = redact_calls[0]["cutoff"]
    assert cutoff.tzinfo is None
    assert result.cutoff == cutoff


def test_cutoff_trails_now_by_the_configured_rolling_window(
    db: Session, redact_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = datetime(2026, 3, 1, 12, 0, 0)
    monkeypatch.setattr(retention, "now", lambda: frozen)
    monkeypatch.setattr(settings, "DELETE_ROLLING_WINDOW_HOURS", 168)

    redact_aged_llm_calls(session=db)

    assert redact_calls[0]["cutoff"] == datetime(2026, 2, 22, 12, 0, 0)


def test_passes_session_to_crud(db: Session, redact_calls: list[dict]) -> None:
    redact_aged_llm_calls(session=db)

    assert redact_calls[0]["session"] is db


def test_default_rolling_window_is_one_week() -> None:
    assert settings.DELETE_ROLLING_WINDOW_TIMEDELTA == timedelta(days=7)
