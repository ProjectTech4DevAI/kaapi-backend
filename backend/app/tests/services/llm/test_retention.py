from datetime import datetime, timedelta

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.services.llm import retention
from app.services.llm.retention import (
    LLM_CALL_REDACTION_BATCH_SIZE,
    redact_aged_llm_calls,
)


@pytest.fixture
def batch_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Records the kwargs of each redact_llm_call_batch call; returns 0 by default."""
    calls: list[dict] = []

    def fake_batch(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(retention, "redact_llm_call_batch", fake_batch)
    return calls


def _stub_returns(
    monkeypatch: pytest.MonkeyPatch, returns: list[int], calls: list[dict]
) -> None:
    remaining = list(returns)

    def fake_batch(**kwargs) -> int:
        calls.append(kwargs)
        return remaining.pop(0)

    monkeypatch.setattr(retention, "redact_llm_call_batch", fake_batch)


def test_sums_rows_across_batches_until_a_batch_is_empty(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    _stub_returns(monkeypatch, [2000, 2000, 137, 0], calls)

    result = redact_aged_llm_calls(session=db)

    assert result.rows_redacted == 4137
    assert result.batches_run == 3
    assert len(calls) == 4


def test_no_matching_rows(db: Session, batch_calls: list[dict]) -> None:
    result = redact_aged_llm_calls(session=db)

    assert result.rows_redacted == 0
    assert result.batches_run == 0
    assert len(batch_calls) == 1


def test_cutoff_is_naive_utc(db: Session, batch_calls: list[dict]) -> None:
    # llm_call.updated_at is a naive TIMESTAMP; a tz-aware cutoff would raise on compare.
    result = redact_aged_llm_calls(session=db)

    cutoff = batch_calls[0]["cutoff"]
    assert cutoff.tzinfo is None
    assert result.cutoff == cutoff


def test_cutoff_trails_now_by_the_configured_rolling_window(
    db: Session, batch_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = datetime(2026, 3, 1, 12, 0, 0)
    monkeypatch.setattr(retention, "now", lambda: frozen)
    monkeypatch.setattr(settings, "DELETE_ROLLING_WINDOW_HOURS", 168)

    redact_aged_llm_calls(session=db)

    assert batch_calls[0]["cutoff"] == datetime(2026, 2, 22, 12, 0, 0)


def test_passes_session_and_module_batch_size_to_crud(
    db: Session, batch_calls: list[dict]
) -> None:
    redact_aged_llm_calls(session=db)

    assert batch_calls[0]["session"] is db
    assert batch_calls[0]["batch_size"] == LLM_CALL_REDACTION_BATCH_SIZE == 2000


def test_default_rolling_window_is_one_week() -> None:
    assert settings.DELETE_ROLLING_WINDOW_TIMEDELTA == timedelta(days=7)
