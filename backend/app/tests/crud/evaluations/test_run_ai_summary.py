"""`generate_run_ai_summary` — the best-effort natural-language note on a run.

The function builds its own Anthropic client from the platform-owned
ANTHROPIC_API_KEY and makes one `messages.create` call (the external boundary,
mocked here). The user message is a qualitative brief (band words + plain area
names + consistency phrases + a repetition line) — never raw scores or the
internal "Adherence to X" labels. Every failure mode (provider error, generic
error, unparseable payload, empty output) must resolve to None WITHOUT raising,
leaving the deterministic overall to persist.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from app.core.config import settings
from app.crud.evaluations.score import OverallSummary
from app.crud.evaluations.summary import _consistency_read, generate_run_ai_summary

_MODEL = "claude-sonnet-4-6"


def _overall() -> OverallSummary:
    return {
        "overall_score": 3.3,
        "verdict": "Needs Refinement",
        "ai_summary": None,
        "breakdown": [
            {
                "name": "Adherence to Ground Truth",
                "key": "ground_truth",
                "score": 4,
                "weight": 0.5,
                "delta": 0.7,
                "verdict": "Good",
            },
            {
                "name": "Adherence to Knowledge Base",
                "key": "knowledge_base",
                "score": 3,
                "weight": 0.3,
                "delta": -0.3,
                "verdict": "Needs Refinement",
            },
            {
                "name": "Adherence to Prompt",
                "key": "prompt",
                "score": 2,
                "weight": 0.2,
                "delta": -1.3,
                "verdict": "Needs Refinement",
            },
        ],
    }


def _summary_scores() -> list[dict]:
    # std per dimension drives the consistency read (0–5 spread; cutoffs 0.5 / 1.0);
    # names match the overall's dims.
    return [
        {"name": "Adherence to Ground Truth", "avg": 4.0, "std": 0.3},
        {"name": "Adherence to Knowledge Base", "avg": 3.0, "std": 0.8},
        {"name": "Adherence to Prompt", "avg": 2.0, "std": 1.5},
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
    summary_scores: list[dict] | None = None,
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
            overall=_overall(),
            run_name="run-x",
            summary_scores=summary_scores
            if summary_scores is not None
            else _summary_scores(),
            duplication_factor=duplication_factor,
        )


class TestHappyPath:
    def test_returns_the_summary_field_stripped(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = _message(
            "  Consistently grounded and on-tone across the set.  "
        )
        assert _call(client) == "Consistently grounded and on-tone across the set."


class TestQualitativeBrief:
    def _brief_for(self, *, duplication_factor: int) -> str:
        client = MagicMock()
        client.messages.create.return_value = _message("A note.")
        _call(client, duplication_factor=duplication_factor)
        return client.messages.create.call_args.kwargs["messages"][0]["content"]

    def test_user_message_carries_plain_area_and_consistency_phrases_and_token_cap(
        self,
    ) -> None:
        client = MagicMock()
        client.messages.create.return_value = _message("A note.")
        _call(client, duplication_factor=5)

        params = client.messages.create.call_args.kwargs
        assert params["model"] == _MODEL
        assert params["max_tokens"] == 2000
        assert params["messages"][0]["role"] == "user"
        assert params["output_config"]["format"]["type"] == "json_schema"

        brief = params["messages"][0]["content"]
        assert "Accuracy against the expected answers" in brief
        assert "Grounding in the source material" in brief
        assert "Tone and instruction-following" in brief

        assert "answers stayed consistent" in brief  # std 0.3
        assert "answers were mostly consistent, with some variation" in brief  # 0.8
        assert "answers varied" in brief  # 1.5

    def test_brief_leaks_no_raw_scores_or_internal_labels(self) -> None:
        brief = self._brief_for(duplication_factor=5)
        assert "Adherence to" not in brief
        assert "verdict" not in brief
        # Raw score / weight / delta values must not cross into the brief.
        for leaked in ("3.3", "0.7", "-1.3", "0.5", "0.3"):
            assert leaked not in brief

    def test_repetition_line_states_the_repeat_count_when_gt_one(self) -> None:
        brief = self._brief_for(duplication_factor=5)
        assert "Each question was asked 5 times." in brief

    def test_repetition_line_softens_when_asked_once(self) -> None:
        brief = self._brief_for(duplication_factor=1)
        assert "asked once (no repetition to speak of)" in brief
        assert "Each question was asked 1 times" not in brief


class TestConsistencyRead:
    @pytest.mark.parametrize(
        ("std", "expected"),
        [
            (0.3, "answers stayed consistent"),
            (0.5, "answers stayed consistent"),
            (0.8, "answers were mostly consistent, with some variation"),
            (1.0, "answers were mostly consistent, with some variation"),
            (1.5, "answers varied"),
            (None, "consistency unknown"),
        ],
    )
    def test_bands_and_boundaries(self, std, expected) -> None:
        assert _consistency_read(std) == expected


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
                overall=_overall(),
                run_name="run-x",
                summary_scores=_summary_scores(),
                duplication_factor=5,
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
