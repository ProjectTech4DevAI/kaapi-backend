"""`generate_run_ai_summary` — the best-effort natural-language note on a run.

The function builds its own Anthropic client from the platform-owned
ANTHROPIC_API_KEY and makes one `messages.create` call (the external boundary,
mocked here). The user message is a diagnostic brief: run name, duplication
factor, the evaluated AI config, and the per-question judge traces as JSON —
question, golden answer, generated answer, and each judge score with its
rationale. Trace bookkeeping (`data_type`, `verdict`, `unscoreable`) stays out of
the payload. Every failure mode (provider error, generic error, unparseable
payload, empty output) must resolve to None WITHOUT raising, leaving the
deterministic overall to persist.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from app.core.config import settings
from app.crud.evaluations.score import TraceData
from app.crud.evaluations.summary import generate_run_ai_summary

_MODEL = "claude-sonnet-4-6"
_RUN_NAME = "run-x"
_CONFIG_PROMPT = "You are a farming helpline bot. Always answer in Hindi."
_TRACES_MARKER = "## Per-question judge traces (JSON)\n"


def _traces() -> list[TraceData]:
    return [
        {
            "trace_id": "item_1_1",
            "question": "How much urea per acre?",
            "llm_answer": "About 50 kg per acre.",
            "question_id": 1,
            "ground_truth_answer": "Roughly 45-55 kg per acre.",
            "category": "fertiliser",
            "scores": [
                {
                    "name": "Adherence to Ground Truth",
                    "value": 4,
                    "data_type": "NUMERIC",
                    "comment": "conveys the same dosage range",
                    "verdict": "Good",
                },
                {
                    "name": "Adherence to Prompt",
                    "value": 1,
                    "data_type": "NUMERIC",
                    "comment": "answered in English, not Hindi",
                    "verdict": "Needs Improvement",
                },
            ],
        },
        {
            "trace_id": "item_2_1",
            "question": "Is the helpline open on Sunday?",
            "llm_answer": "Information not available.",
            "question_id": 2,
            "ground_truth_answer": "Yes, 9am to 1pm.",
            "category": "general",
            # No comment key: the judge left no rationale for this placeholder.
            "scores": [
                {
                    "name": "Adherence to Knowledge Base",
                    "value": "N/A",
                    "data_type": "CATEGORICAL",
                    "unscoreable": True,
                }
            ],
        },
    ]


def _message(summary: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text=json.dumps({"summary": summary})),
        ]
    )


def _http_error(exc_type: type, *, status_code: int) -> Exception:
    return exc_type(
        message="provider said no",
        response=MagicMock(status_code=status_code, request=MagicMock(), headers={}),
        body=None,
    )


def _call(
    client: MagicMock,
    *,
    duplication_factor: int = 5,
    config_prompt: str = _CONFIG_PROMPT,
    traces: list[TraceData] | None = None,
) -> str | None:
    # ANTHROPIC_API_KEY is unset in .env.test, which would short-circuit before
    # the client is ever built.
    with (
        patch.object(settings, "ANTHROPIC_API_KEY", "sk-test"),
        patch(
            "app.crud.evaluations.summary.ClaudeProvider.create_client",
            return_value=client,
        ),
    ):
        return generate_run_ai_summary(
            model=_MODEL,
            run_name=_RUN_NAME,
            duplication_factor=duplication_factor,
            config_prompt=config_prompt,
            traces=_traces() if traces is None else traces,
        )


def _brief_for(**kwargs) -> str:
    client = MagicMock()
    client.messages.create.return_value = _message("A note.")
    _call(client, **kwargs)
    return client.messages.create.call_args.kwargs["messages"][0]["content"]


def _payload_from(brief: str) -> list[dict]:
    return json.loads(brief.split(_TRACES_MARKER, 1)[1])


class TestHappyPath:
    def test_returns_the_summary_field_stripped(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = _message(
            "  Consistently grounded and on-tone across the set.  "
        )
        assert _call(client) == "Consistently grounded and on-tone across the set."

    def test_call_params_carry_the_model_token_cap_and_json_schema(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = _message("A note.")
        _call(client)

        params = client.messages.create.call_args.kwargs
        assert params["model"] == _MODEL
        assert params["max_tokens"] == 3000
        assert params["messages"][0]["role"] == "user"
        assert params["output_config"]["format"]["type"] == "json_schema"


class TestTraceBrief:
    def test_header_carries_run_name_duplication_factor_and_config_prompt(self) -> None:
        brief = _brief_for(duplication_factor=5)
        assert f"Run: {_RUN_NAME}" in brief
        assert "Duplication factor: 5" in brief
        assert _CONFIG_PROMPT in brief

    def test_duplication_factor_of_one_renders_plainly(self) -> None:
        assert "Duplication factor: 1" in _brief_for(duplication_factor=1)

    def test_blank_config_prompt_falls_back_to_the_placeholder(self) -> None:
        brief = _brief_for(config_prompt="")
        assert "(no instructions configured)" in brief
        assert _CONFIG_PROMPT not in brief

    def test_each_trace_carries_its_qa_and_scored_rationales(self) -> None:
        payload = _payload_from(_brief_for())

        assert [t["question_id"] for t in payload] == [1, 2]
        first = payload[0]
        assert first["question"] == "How much urea per acre?"
        assert first["ground_truth_answer"] == "Roughly 45-55 kg per acre."
        assert first["llm_answer"] == "About 50 kg per acre."
        assert first["scores"] == [
            {
                "name": "Adherence to Ground Truth",
                "value": 4,
                "rationale": "conveys the same dosage range",
            },
            {
                "name": "Adherence to Prompt",
                "value": 1,
                "rationale": "answered in English, not Hindi",
            },
        ]

    def test_score_without_a_comment_gets_an_empty_rationale(self) -> None:
        payload = _payload_from(_brief_for())
        assert payload[1]["scores"] == [
            {"name": "Adherence to Knowledge Base", "value": "N/A", "rationale": ""}
        ]

    def test_trace_bookkeeping_keys_never_reach_the_model(self) -> None:
        brief = _brief_for()
        payload = _payload_from(brief)

        assert set(payload[0]) == {
            "question_id",
            "question",
            "ground_truth_answer",
            "llm_answer",
            "scores",
        }
        for trace in payload:
            for score in trace["scores"]:
                assert set(score) == {"name", "value", "rationale"}
        for leaked in ("data_type", "verdict", "unscoreable", "NUMERIC", "category"):
            assert leaked not in brief

    def test_empty_traces_render_an_empty_payload(self) -> None:
        assert _payload_from(_brief_for(traces=[])) == []

    def test_devanagari_qa_survives_unescaped(self) -> None:
        traces = _traces()
        traces[0]["question"] = "एक एकड़ में कितनी यूरिया डालें?"
        traces[0]["llm_answer"] = "लगभग 50 किलो प्रति एकड़।"
        brief = _brief_for(traces=traces)

        assert "एक एकड़ में कितनी यूरिया डालें?" in brief
        # Everything else in the brief is ASCII, so any \u escape means ensure_ascii.
        assert "\\u" not in brief
        assert _payload_from(brief)[0]["llm_answer"] == "लगभग 50 किलो प्रति एकड़।"


class TestMissingApiKey:
    def test_empty_api_key_returns_none_without_building_a_client(self) -> None:
        with (
            patch.object(settings, "ANTHROPIC_API_KEY", ""),
            patch(
                "app.crud.evaluations.summary.ClaudeProvider.create_client"
            ) as create_client,
        ):
            result = generate_run_ai_summary(
                model=_MODEL,
                run_name=_RUN_NAME,
                duplication_factor=5,
                config_prompt=_CONFIG_PROMPT,
                traces=_traces(),
            )

        assert result is None
        create_client.assert_not_called()


class TestFailureIsNonFatal:
    @pytest.mark.parametrize(
        "exception_factory",
        [
            pytest.param(
                lambda: _http_error(anthropic.AuthenticationError, status_code=401),
                id="authentication_error",
            ),
            pytest.param(
                lambda: _http_error(anthropic.RateLimitError, status_code=429),
                id="rate_limit_error",
            ),
            pytest.param(
                lambda: anthropic.APITimeoutError(request=MagicMock()),
                id="timeout_error",
            ),
            pytest.param(
                lambda: anthropic.APIConnectionError(request=MagicMock()),
                id="connection_error",
            ),
            # 4xx and 5xx take different log branches; both must still return None.
            pytest.param(
                lambda: _http_error(anthropic.APIStatusError, status_code=400),
                id="status_error_4xx",
            ),
            pytest.param(
                lambda: _http_error(anthropic.APIStatusError, status_code=503),
                id="status_error_5xx",
            ),
            pytest.param(lambda: RuntimeError("unexpected shape"), id="generic_error"),
        ],
    )
    def test_provider_errors_return_none_without_raising(
        self, exception_factory
    ) -> None:
        client = MagicMock()
        client.messages.create.side_effect = exception_factory()
        assert _call(client) is None

    @pytest.mark.parametrize("summary", ["", "   \n\t"])
    def test_empty_or_whitespace_summary_returns_none(self, summary: str) -> None:
        client = MagicMock()
        client.messages.create.return_value = _message(summary)
        assert _call(client) is None

    def test_non_json_text_block_returns_none_without_raising(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="not json at all")]
        )
        assert _call(client) is None

    def test_response_without_a_text_block_returns_none_without_raising(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", text=None)]
        )
        assert _call(client) is None
