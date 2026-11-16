import httpx
import pytest

from app.services.chatbot.agents.coordinator.decisions import (
    DecisionResult,
    decide_intent,
)
from app.services.chatbot.exceptions import ChatbotProviderError
from app.services.chatbot.types import ChatbotMessageDict, CoordinatorRoute
from app.tests.utils.chatbot import (
    COORDINATOR_PROMPT,
    decisions_answer,
    patch_decisions_api,
)

API_KEY = "sk-test-decisions"
ROUTES = [
    CoordinatorRoute(
        intent="Enroll", description="Sign up a new student", target_node_id="enroll"
    ),
    CoordinatorRoute(intent="Feedback", description=None, target_node_id="feedback"),
]


def _decide(
    *,
    history: list[ChatbotMessageDict] | None = None,
    in_progress_intent: str | None = None,
) -> DecisionResult:
    return decide_intent(
        api_key=API_KEY,
        system_prompt=COORDINATOR_PROMPT,
        routes=ROUTES,
        history=history or [],
        user_message="I want to enroll my son",
        in_progress_intent=in_progress_intent,
    )


class TestDecideIntent:
    def test_confident_choice_is_the_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_decisions_api(
            monkeypatch, httpx.Response(200, json=decisions_answer("Enroll", 0.9))
        )

        result = _decide()

        assert result == DecisionResult(
            intent="Enroll",
            confidence=0.9,
            probabilities={"Enroll": 0.9, "other": 0.1},
        )

    def test_other_choice_has_no_intent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_decisions_api(
            monkeypatch, httpx.Response(200, json=decisions_answer("other", 0.95))
        )

        assert _decide().intent is None

    def test_choice_below_threshold_has_no_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_decisions_api(
            monkeypatch, httpx.Response(200, json=decisions_answer("Enroll", 0.55))
        )

        result = _decide()

        assert result.intent is None
        assert result.confidence == 0.55


class TestDecideIntentErrors:
    @pytest.mark.parametrize(
        ("outcome", "fragment"),
        [
            (
                httpx.Response(
                    200, json={"answers": [{"type": "refusal", "name": "intent"}]}
                ),
                "refused",
            ),
            (httpx.Response(503, text="overloaded"), "code: 503"),
            (httpx.Response(401, text="bad key"), "code: 401"),
            (httpx.ReadTimeout("slow"), "timed out"),
            (httpx.ConnectError("dns"), "ConnectError"),
            (httpx.Response(200, text="not json"), "malformed"),
            (httpx.Response(200, json={"answers": []}), "malformed"),
        ],
        ids=[
            "refusal",
            "5xx",
            "4xx",
            "timeout",
            "connect-error",
            "invalid-json",
            "no-answers",
        ],
    )
    def test_failure_becomes_provider_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outcome: httpx.Response | Exception,
        fragment: str,
    ) -> None:
        patch_decisions_api(monkeypatch, outcome)

        with pytest.raises(ChatbotProviderError, match=fragment):
            _decide()


class TestDecideIntentRequest:
    def test_request_lists_every_route_plus_other_and_sends_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = patch_decisions_api(
            monkeypatch, httpx.Response(200, json=decisions_answer("Enroll", 0.9))
        )

        _decide()

        [request] = fake.requests
        assert request["url"] == "https://api.openai.com/v1/decisions"
        assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
        body = request["json"]
        assert body["model"] == "gpt-6-luna"
        [question] = body["questions"]
        assert (question["type"], question["name"]) == ("choice", "intent")
        assert [c["value"] for c in question["choices"]] == [
            "Enroll",
            "Feedback",
            "other",
        ]
        assert question["choices"][0]["description"] == "Sign up a new student"
        assert "description" not in question["choices"][1]
        assert body["input"][0] == {"role": "developer", "content": COORDINATOR_PROMPT}
        assert body["input"][-1] == {
            "role": "user",
            "content": "I want to enroll my son",
        }

    @pytest.mark.parametrize("in_progress_intent", ["Feedback", None])
    def test_instructions_name_the_in_progress_intent_only_when_set(
        self, monkeypatch: pytest.MonkeyPatch, in_progress_intent: str | None
    ) -> None:
        fake = patch_decisions_api(
            monkeypatch, httpx.Response(200, json=decisions_answer("Enroll", 0.9))
        )

        _decide(in_progress_intent=in_progress_intent)

        instructions = fake.requests[0]["json"]["questions"][0]["instructions"]
        assert ("in the middle of 'Feedback'" in instructions) is (
            in_progress_intent is not None
        )

    def test_history_is_trimmed_to_the_last_twelve_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = patch_decisions_api(
            monkeypatch, httpx.Response(200, json=decisions_answer("Enroll", 0.9))
        )
        history: list[ChatbotMessageDict] = [
            {"role": "user" if i % 2 else "assistant", "content": f"turn {i}"}
            for i in range(20)
        ]

        _decide(history=history)

        sent = fake.requests[0]["json"]["input"]
        # developer prompt + 12 history messages + the new user message
        assert [m["content"] for m in sent[1:-1]] == [f"turn {i}" for i in range(8, 20)]
