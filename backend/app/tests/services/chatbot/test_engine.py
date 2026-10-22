import anthropic
import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.chatbot.engine import run_chatbot_turn
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.chatbot import ScriptedChatModel, patch_chat_model

SYSTEM_PROMPT = "Collect the user's name and city."
SUMMARY_TEXT = "User is Asha from Pune."

_ANTHROPIC_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _alternating_history(turns: int) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for i in range(turns):
        history.append({"role": "user", "content": f"user {i}"})
        history.append({"role": "assistant", "content": f"bot {i}"})
    return history


def _model_settings(history_token_limit: int = 4000) -> ChatbotModelSettings:
    return ChatbotModelSettings(
        model="claude-sonnet-4-6", history_token_limit=history_token_limit
    )


class TestRunChatbotTurn:
    @pytest.mark.asyncio
    async def test_first_call_returns_greeting_without_kickoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Hi! What's your name?"])
        patch_chat_model(monkeypatch, chat_model)

        transcript = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message=None,
            model_settings=_model_settings(),
        )

        assert transcript == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! What's your name?"},
        ]
        # Anthropic rejects a conversation that doesn't open with a user turn.
        sent = chat_model.calls[0]
        assert isinstance(sent[0], SystemMessage)
        assert isinstance(sent[1], HumanMessage)
        assert sent[1].content == "Begin the conversation."

    @pytest.mark.asyncio
    async def test_follow_up_turn_appends_user_message_and_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Nice to meet you, Asha."])
        patch_chat_model(monkeypatch, chat_model)
        history = [
            {"role": "system", "content": "stale prompt from the previous call"},
            {"role": "assistant", "content": "Hi! What's your name?"},
        ]

        transcript = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            user_message="Asha",
            model_settings=_model_settings(),
        )

        assert transcript == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! What's your name?"},
            {"role": "user", "content": "Asha"},
            {"role": "assistant", "content": "Nice to meet you, Asha."},
        ]
        sent_contents = [m.content for m in chat_model.calls[0]]
        assert "stale prompt from the previous call" not in sent_contents

    @pytest.mark.asyncio
    async def test_missing_message_with_history_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel()
        patch_chat_model(monkeypatch, chat_model)

        with pytest.raises(ChatbotInputError, match="`message` is required"):
            await run_chatbot_turn(
                system_prompt=SYSTEM_PROMPT,
                history=[{"role": "assistant", "content": "Hi!"}],
                user_message=None,
                model_settings=_model_settings(),
            )
        assert chat_model.calls == []

    @pytest.mark.asyncio
    async def test_unsupported_role_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel()
        patch_chat_model(monkeypatch, chat_model)

        with pytest.raises(ChatbotInputError, match="Unsupported message role: tool"):
            await run_chatbot_turn(
                system_prompt=SYSTEM_PROMPT,
                history=[{"role": "tool", "content": "{}"}],
                user_message="hello",
                model_settings=_model_settings(),
            )
        assert chat_model.calls == []


class TestProviderErrorMapping:
    @pytest.mark.parametrize(
        ("error", "expected_message"),
        [
            (
                anthropic.APITimeoutError(request=_ANTHROPIC_REQUEST),
                "(code: APITimeoutError)",
            ),
            (
                anthropic.APIConnectionError(request=_ANTHROPIC_REQUEST),
                "(code: APIConnectionError)",
            ),
            (
                anthropic.APIStatusError(
                    "Overloaded",
                    response=httpx.Response(
                        529,
                        request=_ANTHROPIC_REQUEST,
                        headers={"request-id": "req_123"},
                    ),
                    body=None,
                ),
                "[ANTHROPIC] Chat model call failed: Overloaded (code: 529, request_id: req_123)",
            ),
        ],
        ids=["timeout", "connection", "status"],
    )
    @pytest.mark.asyncio
    async def test_anthropic_errors_become_provider_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: anthropic.APIError,
        expected_message: str,
    ) -> None:
        patch_chat_model(monkeypatch, ScriptedChatModel(error=error))

        with pytest.raises(ChatbotProviderError) as exc_info:
            await run_chatbot_turn(
                system_prompt=SYSTEM_PROMPT,
                history=[],
                user_message="hello",
                model_settings=_model_settings(),
            )

        assert expected_message in str(exc_info.value)
        assert exc_info.value.__cause__ is error


class TestHistorySummarization:
    @pytest.mark.asyncio
    async def test_long_history_over_budget_is_summarized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=[SUMMARY_TEXT, "Got it."], token_count=5000
        )
        patch_chat_model(monkeypatch, chat_model)
        history = _alternating_history(turns=4)

        transcript = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            user_message="user 4",
            model_settings=_model_settings(history_token_limit=4000),
        )

        summarize_call, reply_call = chat_model.calls
        # 9 conversation messages, keep-recent=6 lands on an assistant turn, so the
        # window widens back to the preceding user turn: [user 0, bot 0] get summarized.
        assert [m.content for m in summarize_call[1:]] == ["user 0", "bot 0"]
        assert [m.content for m in reply_call] == [
            SYSTEM_PROMPT,
            f"[Earlier conversation summary] {SUMMARY_TEXT}",
            "user 1",
            "bot 1",
            "user 2",
            "bot 2",
            "user 3",
            "bot 3",
            "user 4",
        ]
        # Summarization only trims the model's view; the caller keeps the full transcript.
        assert transcript == [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": "user 4"},
            {"role": "assistant", "content": "Got it."},
        ]

    @pytest.mark.asyncio
    async def test_long_history_within_budget_is_sent_whole(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Got it."], token_count=100)
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=_alternating_history(turns=4),
            user_message="user 4",
            model_settings=_model_settings(history_token_limit=4000),
        )

        assert len(chat_model.calls) == 1
        assert len(chat_model.calls[0]) == 10

    @pytest.mark.asyncio
    async def test_short_history_never_summarized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Got it."], token_count=10_000)
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=_alternating_history(turns=3)[:5],
            user_message="more",
            model_settings=_model_settings(history_token_limit=10),
        )

        assert len(chat_model.calls) == 1
        assert len(chat_model.calls[0]) == 7

    @pytest.mark.asyncio
    async def test_no_user_turn_to_split_on_skips_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Got it."], token_count=10_000)
        patch_chat_model(monkeypatch, chat_model)
        # Only the injected kickoff precedes the recent window, so widening the
        # window to start on a user turn consumes everything there was to summarize.
        assistant_only = [
            {"role": "assistant", "content": f"bot {i}"} for i in range(7)
        ]

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=assistant_only,
            user_message="hello",
            model_settings=_model_settings(history_token_limit=10),
        )

        assert len(chat_model.calls) == 1
        assert len(chat_model.calls[0]) == 10
