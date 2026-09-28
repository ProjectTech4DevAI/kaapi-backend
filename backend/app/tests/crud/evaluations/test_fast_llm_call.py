"""Fast-eval generation against `execute_llm_call` (`crud/evaluations/fast.py`).

Stage 1 no longer calls the OpenAI SDK: every row goes through the same
`/llm/call` path production traffic does, so guardrails and the config's
prompt_template apply to eval runs too. These tests pin the `BlockResult` →
per-item-result mapping (the shape persisted to S3) and the kwargs the worker
hands `execute_llm_call`.

The LLM boundary is mocked at `_execute_llm_call_for_question` (the mapping
tests) or one level lower at `execute_llm_call` (the tests that need the real
retry decorator to run). No DB: the worker only reads the dataset item dict.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.crud.evaluations.fast import (
    _execute_llm_call_for_question,
    _llm_call_for_item,
)
from app.crud.evaluations.retry import RETRY_MAX_ATTEMPTS
from app.models.llm import QueryParams, Usage
from app.models.llm.request import LLMCallConfig
from app.services.llm.chain.types import BlockResult
from app.tests.utils.llm import text_llm_call_response

# The auth failures guardrails fails closed on; the message the service surfaces.
GUARDRAILS_AUTH_ERROR = "Guardrails service rejected the request (HTTP 401)"

QUESTION = "What is X?"
PROJECT_ID = 11
ORG_ID = 22

USAGE = Usage(input_tokens=5, output_tokens=7, total_tokens=12)
USAGE_DICT = {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Record tenacity's backoff instead of serving it.

    `tenacity.nap.sleep` is bound as a default argument at import time, so the
    underlying `time.sleep` is the only reachable seam.
    """
    recorded: list[float] = []
    monkeypatch.setattr("tenacity.nap.time.sleep", recorded.append)
    yield recorded


def _config() -> LLMCallConfig:
    return LLMCallConfig(id=uuid4(), version=3)


def _item(
    item_id: str = "item-1",
    *,
    question: str | None = QUESTION,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "input": {"question": question} if question is not None else {},
        "expected_output": {"answer": "golden"},
        "metadata": {"question_id": 7},
    }


def _run_item(
    result: BlockResult | Exception, item: dict[str, Any] | None = None
) -> tuple[dict[str, Any], MagicMock]:
    kwargs = (
        {"side_effect": result}
        if isinstance(result, Exception)
        else {"return_value": result}
    )
    with patch(
        "app.crud.evaluations.fast._execute_llm_call_for_question", **kwargs
    ) as mock_call:
        return (
            _llm_call_for_item(
                config=_config(),
                project_id=PROJECT_ID,
                organization_id=ORG_ID,
                item=item if item is not None else _item(),
            ),
            mock_call,
        )


class TestBlockResultMapping:
    def test_blocked_row_is_empty_output_not_failed_and_labelled(self) -> None:
        result, _ = _run_item(
            BlockResult(
                guardrail_outcome="blocked",
                error="uli_slur_match",
                response=None,
                usage=USAGE,
            )
        )

        assert result["generated_output"] == ""
        assert result["failed"] is False
        assert result["guardrail"] == "blocked: uli_slur_match"

    def test_blocked_row_keeps_the_tokens_the_provider_already_billed(self) -> None:
        result, _ = _run_item(
            BlockResult(guardrail_outcome="blocked", error="uli", usage=USAGE)
        )

        assert result["response_id"] is None
        assert result["retrieved_chunks"] == []
        assert result["usage"] == USAGE_DICT

    def test_rephrased_row_carries_the_rephrase_text(self) -> None:
        result, _ = _run_item(
            BlockResult(
                guardrail_outcome="rephrased",
                response=text_llm_call_response("please rephrase your question"),
                usage=USAGE,
            )
        )

        assert result["generated_output"] == "please rephrase your question"
        assert result["failed"] is False
        assert result["guardrail"] == "rephrased"

    def test_provider_error_without_a_verdict_fails_the_row(self) -> None:
        result, _ = _run_item(BlockResult(error="provider 503", usage=USAGE))

        assert result["failed"] is True
        assert result["generated_output"] == "ERROR: provider 503"
        assert result["guardrail"] is None

    @pytest.mark.parametrize("key", ["input_guardrail", "output_guardrail"])
    def test_success_with_guardrail_metadata_is_labelled_applied(
        self, key: str
    ) -> None:
        result, _ = _run_item(
            BlockResult(
                response=text_llm_call_response("answer text"),
                usage=USAGE,
                metadata={key: {"validators": []}},
            )
        )

        assert result["generated_output"] == "answer text"
        assert result["failed"] is False
        assert result["guardrail"] == "applied"

    def test_success_without_guardrail_metadata_has_no_label(self) -> None:
        result, _ = _run_item(
            BlockResult(
                response=text_llm_call_response("answer text"),
                usage=USAGE,
                metadata={"latency_ms": 12},
            )
        )

        assert result["generated_output"] == "answer text"
        assert result["failed"] is False
        assert result["guardrail"] is None


class TestRowIsolation:
    def test_missing_question_fails_the_row_without_calling_the_llm(self) -> None:
        result, mock_call = _run_item(
            BlockResult(response=text_llm_call_response()), item=_item(question=None)
        )

        assert result["failed"] is True
        assert result["generated_output"] == "ERROR: missing question in dataset item"
        assert mock_call.call_count == 0

    def test_raised_exception_fails_only_that_row(self) -> None:
        result, _ = _run_item(RuntimeError("config lookup exploded"))

        assert result["failed"] is True
        assert result["generated_output"] == "ERROR: config lookup exploded"


class TestGuardrailsAuthFailsClosed:
    def test_auth_rejection_fails_the_row_after_exhausting_retries(
        self, no_backoff: list[float]
    ) -> None:
        # A guardrails 401/403/422 is a fail-closed transport error, not a content
        # verdict: `guardrail_outcome` stays None, so the row must come back
        # `failed=True` rather than as a quietly unscoreable row — broken guardrail
        # credentials have to show up in the run's failure ratio.
        # (docs/wiki/modules/guardrails.md §9, invariant 8)
        with patch(
            "app.crud.evaluations.fast.execute_llm_call",
            return_value=BlockResult(
                error=GUARDRAILS_AUTH_ERROR, guardrail_outcome=None
            ),
        ) as mock_execute:
            result = _llm_call_for_item(
                config=_config(),
                project_id=PROJECT_ID,
                organization_id=ORG_ID,
                item=_item(),
            )

        assert result["failed"] is True
        assert result["generated_output"] == f"ERROR: {GUARDRAILS_AUTH_ERROR}"
        assert result["guardrail"] is None
        assert mock_execute.call_count == RETRY_MAX_ATTEMPTS


class TestExecuteLlmCallForQuestion:
    @staticmethod
    def _capture(outcomes: list[BlockResult]) -> list[dict[str, Any]]:
        """Run the worker, recording each attempt's kwargs plus the input value the
        attempt actually saw, then mutating the query as execute_llm_call does."""
        calls: list[dict[str, Any]] = []

        def _fake(**kwargs: Any) -> BlockResult:
            calls.append({**kwargs, "seen_input": kwargs["query"].input.content.value})
            kwargs["query"].input.content.value = "rewritten in place"
            return outcomes[len(calls) - 1]

        with patch("app.crud.evaluations.fast.execute_llm_call", side_effect=_fake):
            _execute_llm_call_for_question(
                config=_config(),
                question=QUESTION,
                project_id=PROJECT_ID,
                organization_id=ORG_ID,
            )
        return calls

    def test_sends_the_eval_specific_flags(self, no_backoff: list[float]) -> None:
        (call,) = self._capture([BlockResult(response=text_llm_call_response())])

        assert call["record_call"] is False
        assert call["include_provider_raw_response"] is True
        assert call["include_guardrail_metadata"] is True
        assert call["langfuse_credentials"] is None
        assert call["request_metadata"] is None
        assert call["project_id"] == PROJECT_ID
        assert call["organization_id"] == ORG_ID
        assert isinstance(call["query"], QueryParams)
        assert call["seen_input"] == QUESTION

    def test_each_retried_attempt_gets_a_fresh_query_and_job_id(
        self, no_backoff: list[float]
    ) -> None:
        """execute_llm_call rewrites `query.input.content.value` in place twice —
        prompt_template interpolation, then the guardrails' safe_text. Reusing the
        object across attempts would apply the template to already-templated text
        and resend already-sanitised input, so identity, not equality, is asserted.
        """
        first, second = self._capture(
            [
                BlockResult(error="provider 503"),
                BlockResult(response=text_llm_call_response()),
            ]
        )

        assert first["query"] is not second["query"]
        assert second["seen_input"] == QUESTION
        assert isinstance(second["job_id"], UUID)
        assert first["job_id"] != second["job_id"]
