"""`retry_llm_call`, the result-based retry policy generation uses (`retry.py`).

`retry_openai_call` (embeddings, judge) is tenacity's stock exception-based
behaviour and is not re-asserted here.

Two behaviours of the result-based policy are load-bearing and are
asserted directly: a guardrail verdict is never retried (it is a content decision,
not a provider failure), and an exhausted retry *returns* the last result instead
of raising — a raise would propagate through `_run_in_pool`'s `future.result()`,
abort the whole chunk, and have the cron healer re-enqueue it, re-charging the
provider for every row.
"""

import logging
from collections.abc import Callable, Iterator

import pytest

from app.crud.evaluations.retry import (
    RETRY_MAX_ATTEMPTS,
    retry_llm_call,
)
from app.models.llm.response import Usage
from app.services.llm.chain.types import BlockResult, GuardrailOutcomeLabel

logger = logging.getLogger(__name__)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Record backoff instead of serving it.

    Patching `tenacity.nap.sleep` does NOT work: `BaseRetrying.__init__` binds that
    function as a default argument at import time. The underlying `time.sleep` is
    the only reachable seam.
    """
    recorded: list[float] = []
    monkeypatch.setattr("tenacity.nap.time.sleep", recorded.append)
    yield recorded


def _decorate(fn: Callable[[], BlockResult]) -> Callable[[], BlockResult]:
    return retry_llm_call(logger)(fn)


def _failure(error: str = "provider 503") -> BlockResult:
    return BlockResult(error=error)


def _success() -> BlockResult:
    return BlockResult(usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2))


def _guardrail(outcome: GuardrailOutcomeLabel) -> BlockResult:
    return BlockResult(error="uli_slur_match", guardrail_outcome=outcome)


class TestRetryLlmCall:
    def test_clean_success_runs_once(self, sleeps: list[float]) -> None:
        calls: list[int] = []
        expected = _success()

        @_decorate
        def call() -> BlockResult:
            calls.append(1)
            return expected

        assert call() is expected
        assert len(calls) == 1
        assert sleeps == []

    def test_transient_failure_then_success_runs_twice(
        self, sleeps: list[float]
    ) -> None:
        expected = _success()
        outcomes = [_failure(), expected]

        @_decorate
        def call() -> BlockResult:
            return outcomes.pop(0)

        assert call() is expected
        assert outcomes == []
        assert len(sleeps) == 1

    def test_exhaustion_returns_the_last_result_instead_of_raising(
        self, sleeps: list[float]
    ) -> None:
        attempts: list[BlockResult] = []

        @_decorate
        def call() -> BlockResult:
            attempts.append(_failure(f"provider 503 #{len(attempts)}"))
            return attempts[-1]

        result = call()

        assert isinstance(result, BlockResult)
        assert result is attempts[-1]
        assert result.error == "provider 503 #2"
        assert len(attempts) == RETRY_MAX_ATTEMPTS == 3
        assert len(sleeps) == RETRY_MAX_ATTEMPTS - 1

    @pytest.mark.parametrize("outcome", ["blocked", "rephrased"])
    def test_guardrail_verdict_is_never_retried(
        self, outcome: GuardrailOutcomeLabel, sleeps: list[float]
    ) -> None:
        calls: list[int] = []

        @_decorate
        def call() -> BlockResult:
            calls.append(1)
            return _guardrail(outcome)

        assert call().guardrail_outcome == outcome
        assert len(calls) == 1
        assert sleeps == []

    def test_raised_exception_propagates_without_retrying(
        self, sleeps: list[float]
    ) -> None:
        calls: list[int] = []

        @_decorate
        def call() -> BlockResult:
            calls.append(1)
            raise RuntimeError("config lookup exploded")

        with pytest.raises(RuntimeError, match="config lookup exploded"):
            call()

        assert len(calls) == 1
        assert sleeps == []
