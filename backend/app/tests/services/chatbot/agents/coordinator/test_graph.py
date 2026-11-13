import anthropic
import httpx
import pytest
from langchain_core.exceptions import OutputParserException

from app.services.chatbot.agents.coordinator.graph import (
    CoordinatorDecision,
    CoordinatorRoute,
    RouteDecision,
    run_coordinator_turn,
)
from app.services.chatbot.agents.coordinator.prompts import (
    FALLBACK_CLARIFYING_QUESTION,
)
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.types import ChatbotMessageDict
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.chatbot import (
    ANTHROPIC_REQUEST,
    COORDINATOR_PROMPT,
    ScriptedChatModel,
    patch_chat_model,
)

MODEL = "claude-sonnet-4-6"
ROUTES = [
    CoordinatorRoute(
        intent="Enroll", description="Sign up a new student", target_node_id="enroll"
    ),
    CoordinatorRoute(intent="Feedback", description=None, target_node_id="feedback"),
]
CLARIFY = CoordinatorDecision(
    target_node_id=None, clarifying_question="Do you want to enroll or give feedback?"
)


def _run(
    *,
    user_message: str | None = "I want to enroll my son",
    history: list[ChatbotMessageDict] | None = None,
    in_progress_intent: str | None = None,
) -> CoordinatorDecision:
    return run_coordinator_turn(
        system_prompt=COORDINATOR_PROMPT,
        history=history or [],
        user_message=user_message,
        routes=ROUTES,
        model_settings=ChatbotModelSettings(model=MODEL),
        in_progress_intent=in_progress_intent,
    )


def _scripted(monkeypatch: pytest.MonkeyPatch, *outcomes: object) -> ScriptedChatModel:
    return patch_chat_model(
        monkeypatch,
        ScriptedChatModel(replies=["should never be used"], extractions=list(outcomes)),
    )


class TestRouting:
    @pytest.mark.parametrize(
        ("intent", "target"),
        [
            ("Enroll", "enroll"),
            ("Feedback", "feedback"),
            ("  enroll ", "enroll"),
            ("FEEDBACK", "feedback"),
        ],
        ids=["exact", "second-route", "padded-lowercase", "uppercase"],
    )
    def test_matching_intent_routes_to_its_target(
        self, monkeypatch: pytest.MonkeyPatch, intent: str, target: str
    ) -> None:
        _scripted(monkeypatch, RouteDecision(intent=intent))

        decision = _run()

        assert decision == CoordinatorDecision(
            target_node_id=target, clarifying_question=None
        )

    @pytest.mark.parametrize(
        "intent", [None, "Billing"], ids=["null-intent", "unknown-intent"]
    )
    def test_no_matching_intent_asks_the_models_question(
        self, monkeypatch: pytest.MonkeyPatch, intent: str | None
    ) -> None:
        _scripted(
            monkeypatch,
            RouteDecision(
                intent=intent, clarifying_question=CLARIFY.clarifying_question
            ),
        )

        assert _run(user_message="hmm") == CLARIFY

    @pytest.mark.parametrize("question", [None, "   "], ids=["missing", "blank"])
    def test_missing_question_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, question: str | None
    ) -> None:
        _scripted(monkeypatch, RouteDecision(intent=None, clarifying_question=question))

        decision = _run(user_message="hmm")

        assert decision == CoordinatorDecision(
            target_node_id=None, clarifying_question=FALLBACK_CLARIFYING_QUESTION
        )

    def test_opening_turn_clarifies_even_when_model_names_an_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The opening turn only carries the synthetic kickoff, so there's nothing to route on.
        _scripted(
            monkeypatch,
            RouteDecision(intent="Enroll", clarifying_question="Hi! How can I help?"),
        )

        decision = _run(user_message=None)

        assert decision == CoordinatorDecision(
            target_node_id=None, clarifying_question="Hi! How can I help?"
        )

    def test_routing_is_one_json_schema_call_listing_every_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = _scripted(monkeypatch, RouteDecision(intent="Enroll"))

        _run()

        assert chat_model.structured_output_schemas == [RouteDecision]
        # Forced tool use is rejected while thinking is on.
        assert chat_model.structured_output_methods == ["json_schema"]
        assert chat_model.calls == []
        [sent] = chat_model.extraction_calls
        system_prompt = str(sent[0].content)
        assert system_prompt.startswith(COORDINATOR_PROMPT)
        assert "- Enroll: Sign up a new student" in system_prompt.splitlines()
        assert "- Feedback" in system_prompt.splitlines()
        assert sent[-1].content == "I want to enroll my son"

    def test_missing_message_with_history_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = _scripted(monkeypatch)

        with pytest.raises(ChatbotInputError, match="`message` is required"):
            _run(user_message=None, history=[{"role": "assistant", "content": "Hi!"}])
        assert chat_model.extraction_calls == []


class TestInProgressRouting:
    @pytest.mark.parametrize(
        "intent", [None, "Billing"], ids=["null-intent", "unknown-intent"]
    )
    def test_no_matching_intent_continues_the_in_progress_route(
        self, monkeypatch: pytest.MonkeyPatch, intent: str | None
    ) -> None:
        _scripted(
            monkeypatch,
            RouteDecision(
                intent=intent, clarifying_question=CLARIFY.clarifying_question
            ),
        )

        decision = _run(user_message="Ravi", in_progress_intent="Enroll")

        assert decision == CoordinatorDecision(
            target_node_id="enroll", clarifying_question=None
        )

    def test_matching_another_intent_switches_away_from_the_in_progress_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _scripted(monkeypatch, RouteDecision(intent="Feedback"))

        decision = _run(user_message="I have feedback", in_progress_intent="Enroll")

        assert decision == CoordinatorDecision(
            target_node_id="feedback", clarifying_question=None
        )

    def test_opening_turn_still_clarifies_with_a_route_in_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _scripted(
            monkeypatch,
            RouteDecision(intent=None, clarifying_question="Hi! How can I help?"),
        )

        decision = _run(user_message=None, in_progress_intent="Enroll")

        assert decision == CoordinatorDecision(
            target_node_id=None, clarifying_question="Hi! How can I help?"
        )

    @pytest.mark.parametrize(
        ("in_progress_intent", "expected_in_prompt"),
        [(None, False), ("Enroll", True)],
        ids=["nothing-in-progress", "enroll-in-progress"],
    )
    def test_routing_prompt_names_the_in_progress_intent_only_when_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        in_progress_intent: str | None,
        expected_in_prompt: bool,
    ) -> None:
        chat_model = _scripted(monkeypatch, RouteDecision(intent="Enroll"))

        _run(in_progress_intent=in_progress_intent)

        [sent] = chat_model.extraction_calls
        system_prompt = str(sent[0].content)
        assert ("The user is in the middle of 'Enroll'." in system_prompt) is (
            expected_in_prompt
        )


class TestRoutingErrors:
    @pytest.mark.parametrize(
        ("outcome", "expected_message"),
        [
            (
                OutputParserException("not valid JSON"),
                "[ANTHROPIC] Coordinator routing returned malformed structured output",
            ),
            (
                {"intent": "Enroll"},
                "[ANTHROPIC] Coordinator routing returned an unexpected output shape",
            ),
            (
                ValueError("effort is not supported for this model"),
                f"[KAAPI] Unsupported model configuration for '{MODEL}': "
                "effort is not supported for this model",
            ),
            (
                anthropic.APITimeoutError(request=ANTHROPIC_REQUEST),
                "(code: APITimeoutError)",
            ),
            (
                anthropic.APIConnectionError(request=ANTHROPIC_REQUEST),
                "(code: APIConnectionError)",
            ),
            (
                anthropic.APIStatusError(
                    "Overloaded",
                    response=httpx.Response(
                        529, request=ANTHROPIC_REQUEST, headers={"request-id": "req_9"}
                    ),
                    body=None,
                ),
                "[ANTHROPIC] Chat model call failed: Overloaded "
                "(code: 529, request_id: req_9)",
            ),
        ],
        ids=[
            "output_parser",
            "wrong_type",
            "value_error",
            "timeout",
            "connection",
            "status",
        ],
    )
    def test_routing_failure_becomes_provider_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outcome: object,
        expected_message: str,
    ) -> None:
        _scripted(monkeypatch, outcome)

        with pytest.raises(ChatbotProviderError) as exc_info:
            _run()

        assert expected_message in str(exc_info.value)
