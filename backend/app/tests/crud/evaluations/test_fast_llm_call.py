"""Fast-eval per-item workers: generation via `execute_llm_call`, embedding pairs."""

import itertools
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import openai
import pytest

from app.crud.evaluations.fast import (
    _embedding_call_for_pair,
    _execute_llm_call_for_question,
    _llm_call_for_item,
)
from app.models.llm import QueryParams, Usage
from app.models.llm.request import LLMCallConfig
from app.services.llm.chain.types import BlockResult
from app.tests.utils.llm import text_llm_call_response

GUARDRAILS_AUTH_ERROR = "Guardrails service rejected the request (HTTP 401)"

QUESTION = "What is X?"
PROJECT_ID = 11
ORG_ID = 22

USAGE = Usage(input_tokens=5, output_tokens=7, total_tokens=12)
USAGE_DICT = {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Record backoff; tenacity binds nap.sleep at import, so patch time.sleep."""
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
        assert result["input_to_llm"] is None
        assert result["output_from_llm"] is None

    def test_applied_row_carries_the_text_the_llm_saw_and_produced(self) -> None:
        result, _ = _run_item(
            BlockResult(
                response=text_llm_call_response("call [REDACTED]"),
                usage=USAGE,
                metadata={
                    "input_guardrail": {
                        "input_from_user": "my number is 98765",
                        "input_to_llm": "my number is [REDACTED]",
                        "validators": [],
                    },
                    "output_guardrail": {
                        "output_from_llm": "call 98765",
                        "output_to_user": "call [REDACTED]",
                        "validators": [],
                    },
                },
            )
        )

        assert result["guardrail"] == "applied"
        assert result["input_to_llm"] == "my number is [REDACTED]"
        assert result["output_from_llm"] == "call 98765"
        # The scored answer stays the post-guardrail text.
        assert result["generated_output"] == "call [REDACTED]"

    def test_only_the_side_that_applied_is_filled(self) -> None:
        result, _ = _run_item(
            BlockResult(
                response=text_llm_call_response("answer text"),
                usage=USAGE,
                metadata={"input_guardrail": {"input_to_llm": "redacted q"}},
            )
        )

        assert result["input_to_llm"] == "redacted q"
        assert result["output_from_llm"] is None

    def test_rephrased_row_does_not_report_the_canned_reply_as_llm_input(
        self,
    ) -> None:
        """On rephrase, input_to_llm is the canned reply, not what the model got."""
        result, _ = _run_item(
            BlockResult(
                guardrail_outcome="rephrased",
                response=text_llm_call_response("please rephrase"),
                usage=USAGE,
                metadata={"input_guardrail": {"input_to_llm": "please rephrase"}},
            )
        )

        assert result["input_to_llm"] is None
        assert result["output_from_llm"] is None


@pytest.fixture
def fast_clock() -> Iterator[MagicMock]:
    """1.5 s between the two reads, open-ended so an extra read never StopIterations.

    Patches `latency.time` (the whole module object), never global perf_counter:
    tenacity's own timing uses it.
    """
    with patch("app.crud.evaluations.latency.time") as mock_time:
        mock_time.perf_counter.side_effect = itertools.chain(
            [0.0, 1.5], itertools.repeat(1.5)
        )
        yield mock_time


class TestResponseLatency:
    def test_successful_call_records_wall_time_in_ms(
        self, fast_clock: MagicMock
    ) -> None:
        result, _ = _run_item(
            BlockResult(response=text_llm_call_response("answer"), usage=USAGE)
        )

        assert result["failed"] is False
        assert result["latency_ms"] == 1500.0

    def test_raised_exception_has_no_latency(self, fast_clock: MagicMock) -> None:
        result, _ = _run_item(RuntimeError("boom"))

        assert result["failed"] is True
        assert result["latency_ms"] is None

    def test_provider_error_has_no_latency(self, fast_clock: MagicMock) -> None:
        result, _ = _run_item(BlockResult(error="provider 503", usage=USAGE))

        assert result["failed"] is True
        assert result["latency_ms"] is None

    def test_retry_exhausted_row_has_no_latency(
        self, fast_clock: MagicMock, no_backoff: list[float]
    ) -> None:
        with patch(
            "app.crud.evaluations.fast.execute_llm_call",
            return_value=BlockResult(error="provider 503", retryable=True),
        ) as mock_execute:
            result = _llm_call_for_item(
                config=_config(),
                project_id=PROJECT_ID,
                organization_id=ORG_ID,
                item=_item(),
            )

        assert mock_execute.call_count == 3
        assert result["failed"] is True
        assert result["latency_ms"] is None

    def test_missing_question_has_no_latency(self, fast_clock: MagicMock) -> None:
        result, _ = _run_item(
            BlockResult(response=text_llm_call_response()), item=_item(question=None)
        )

        assert result["latency_ms"] is None

    @pytest.mark.parametrize(
        "block",
        [
            BlockResult(guardrail_outcome="blocked", error="uli", usage=USAGE),
            BlockResult(
                guardrail_outcome="rephrased",
                response=text_llm_call_response("please rephrase"),
                usage=USAGE,
            ),
        ],
        ids=["blocked", "rephrased"],
    )
    def test_guardrail_short_circuit_is_not_a_timed_call(
        self, fast_clock: MagicMock, block: BlockResult
    ) -> None:
        # Scored (failed=False) but may never have reached the LLM.
        result, _ = _run_item(block)

        assert result["failed"] is False
        assert result["latency_ms"] is None

    def test_guardrail_applied_row_is_still_timed(self, fast_clock: MagicMock) -> None:
        result, _ = _run_item(
            BlockResult(
                response=text_llm_call_response("answer"),
                usage=USAGE,
                metadata={"input_guardrail": {"input_to_llm": "redacted q"}},
            )
        )

        assert result["guardrail"] == "applied"
        assert result["latency_ms"] == 1500.0


class TestEmbeddingLatency:
    @staticmethod
    def _client(
        *, data: list[tuple[int, list[float]]] | None = None, error=None
    ) -> MagicMock:
        client = MagicMock()
        if error is not None:
            client.embeddings.create.side_effect = error
        else:
            client.embeddings.create.return_value = SimpleNamespace(
                data=[SimpleNamespace(index=i, embedding=v) for i, v in data or []],
                usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
            )
        return client

    def _embed(self, client: MagicMock, output_text: str = "out") -> dict[str, Any]:
        return _embedding_call_for_pair(
            openai_client=client,
            embedding_model="text-embedding-3-small",
            item_id="item-1",
            output_text=output_text,
            ground_truth="golden",
        )

    def test_successful_pair_records_wall_time_in_ms(
        self, fast_clock: MagicMock
    ) -> None:
        result = self._embed(self._client(data=[(0, [1.0]), (1, [2.0])]))

        assert result["failed"] is False
        assert result["latency_ms"] == 1500.0

    def test_openai_error_has_no_latency(self, fast_clock: MagicMock) -> None:
        # Base OpenAIError is not in the retryable set, so no backoff runs.
        result = self._embed(self._client(error=openai.OpenAIError("bad request")))

        assert result["failed"] is True
        assert result["latency_ms"] is None

    def test_short_payload_has_no_latency(self, fast_clock: MagicMock) -> None:
        result = self._embed(self._client(data=[(0, [1.0])]))

        assert result["failed"] is True
        assert result["latency_ms"] is None

    def test_empty_side_is_not_called_or_timed(self, fast_clock: MagicMock) -> None:
        client = self._client(data=[(0, [1.0]), (1, [2.0])])
        result = self._embed(client, output_text="")

        assert client.embeddings.create.call_count == 0
        assert result["latency_ms"] is None


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
    def test_auth_rejection_fails_the_row_on_the_first_attempt(
        self, no_backoff: list[float]
    ) -> None:
        # Fail-closed transport error: row fails, no retry (each retry re-bills).
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
        assert mock_execute.call_count == 1
        assert no_backoff == []


class TestExecuteLlmCallForQuestion:
    @staticmethod
    def _capture(outcomes: list[BlockResult]) -> list[dict[str, Any]]:
        """Record each attempt's kwargs/input, then mutate the query in place."""
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
        """execute_llm_call mutates the query in place; each attempt needs a new one."""
        first, second = self._capture(
            [
                BlockResult(error="provider 503", retryable=True),
                BlockResult(response=text_llm_call_response()),
            ]
        )

        assert first["query"] is not second["query"]
        assert second["seen_input"] == QUESTION
        assert isinstance(second["job_id"], UUID)
        assert first["job_id"] != second["job_id"]
