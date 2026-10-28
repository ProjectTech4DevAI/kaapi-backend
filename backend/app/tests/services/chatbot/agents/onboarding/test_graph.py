import anthropic
import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import ChatbotCompletionStatusEnum
from app.services.chatbot.agents.onboarding.graph import run_chatbot_turn
from app.services.chatbot.agents.onboarding.prompts import EXTRACTION_INSTRUCTION
from app.services.chatbot.agents.onboarding.types import FieldExtraction
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.chatbot import (
    ScriptedChatModel,
    extracted_fields,
    patch_chat_model,
)

SYSTEM_PROMPT = "Collect the user's name and city."
SUMMARY_TEXT = "User is Asha from Pune."
PROGRESS_PREFIX = "[Field collection progress] "
COLLECTED_PREFIX = "Already collected (do not ask for these again): "

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

        result = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message=None,
            model_settings=_model_settings(),
        )

        assert result.reply == "Hi! What's your name?"
        assert result.sources == []
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

        result = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            user_message="Asha",
            model_settings=_model_settings(),
        )

        assert result.reply == "Nice to meet you, Asha."
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
                ValueError("thinking.type=disabled is not supported"),
                "[KAAPI] Unsupported model configuration for 'claude-sonnet-4-6': "
                "thinking.type=disabled is not supported",
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
        ids=["timeout", "connection", "value_error", "status"],
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

        result = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            user_message="user 4",
            model_settings=_model_settings(history_token_limit=4000),
        )

        summarize_call, reply_call = chat_model.calls
        # 9 conversation messages, keep-recent=6 lands on an assistant turn, so the
        # window widens back to the preceding user turn: [user 0, bot 0] get summarized,
        # sent as one user message so the request never ends on an assistant turn.
        instruction, transcript_message = summarize_call
        assert isinstance(instruction, SystemMessage)
        assert isinstance(transcript_message, HumanMessage)
        assert transcript_message.content == (
            "<conversation>\nUser: user 0\nAssistant: bot 0\n</conversation>"
        )
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
        # The reply is the turn's answer, never the summary text.
        assert result.reply == "Got it."

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

    @pytest.mark.asyncio
    async def test_long_history_summary_is_shared_by_extraction_and_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=[SUMMARY_TEXT, "Got it."],
            token_count=5000,
            extractions=[
                FieldExtraction(
                    collected_fields=extracted_fields({"name": "Asha"}),
                    missing_fields=["city"],
                    is_complete=False,
                )
            ],
        )
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=_alternating_history(turns=4),
            user_message="user 4",
            model_settings=_model_settings(history_token_limit=4000),
        )

        # Exactly one summarize call: the extraction node's view is reused for the reply.
        summarize_call, reply_call = chat_model.calls
        assert summarize_call[-1].content == (
            "<conversation>\nUser: user 0\nAssistant: bot 0\n</conversation>"
        )
        summary_note = f"[Earlier conversation summary] {SUMMARY_TEXT}"
        recent = ["user 1", "bot 1", "user 2", "bot 2", "user 3", "bot 3", "user 4"]
        [extraction_call] = chat_model.extraction_calls
        assert [m.content for m in extraction_call[1:]] == [summary_note, *recent]
        # The progress note joins the leading system block, after the summary note.
        assert [m.content for m in reply_call] == [
            SYSTEM_PROMPT,
            summary_note,
            f'{PROGRESS_PREFIX}{COLLECTED_PREFIX}{{"name": "Asha"}}. '
            "Still missing: city. Ask only for the missing fields.",
            *recent,
        ]
        assert all(isinstance(m, SystemMessage) for m in reply_call[:3])


class TestFieldExtraction:
    @pytest.mark.asyncio
    async def test_greeting_turn_skips_extraction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["Hi! What's your name?"],
            extractions=[
                FieldExtraction(
                    collected_fields=extracted_fields({"name": "invented"}),
                    missing_fields=[],
                    is_complete=True,
                )
            ],
        )
        patch_chat_model(monkeypatch, chat_model)

        result = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message=None,
            model_settings=_model_settings(),
            collected_fields={"name": "Asha"},
        )

        assert chat_model.extraction_calls == []
        assert result.collected_fields == {"name": "Asha"}
        assert result.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS
        [reply_call] = chat_model.calls
        assert [m.content for m in reply_call] == [
            SYSTEM_PROMPT,
            "Begin the conversation.",
        ]

    @pytest.mark.asyncio
    async def test_extraction_uses_native_json_schema_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Which city?"])
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message="I'm Asha",
            model_settings=_model_settings(),
        )

        # Forced tool use is rejected by Anthropic when thinking is on.
        assert chat_model.structured_output_methods == ["json_schema"]

    @pytest.mark.asyncio
    async def test_extraction_sees_flow_prompt_prior_fields_and_conversation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Which city?"])
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[{"role": "assistant", "content": "Hi! What's your name?"}],
            user_message="I'm Asha",
            model_settings=_model_settings(),
            collected_fields={"age": "30"},
        )

        [extraction_call] = chat_model.extraction_calls
        instruction, *conversation = extraction_call
        assert isinstance(instruction, SystemMessage)
        assert instruction.text.startswith(EXTRACTION_INSTRUCTION)
        assert f"<flow_prompt>\n{SYSTEM_PROMPT}\n</flow_prompt>" in instruction.text
        assert (
            '<previously_collected>\n{"age": "30"}\n</previously_collected>'
            in instruction.text
        )
        # The flow prompt is embedded in the instruction, not repeated as a message.
        assert [m.content for m in conversation] == [
            "Begin the conversation.",
            "Hi! What's your name?",
            "I'm Asha",
        ]

    @pytest.mark.asyncio
    async def test_extracted_fields_merge_over_prior_with_corrections_winning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["Noted."],
            extractions=[
                FieldExtraction(
                    collected_fields=extracted_fields({"city": "Mumbai", "age": "30"}),
                    missing_fields=[],
                    is_complete=False,
                )
            ],
        )
        patch_chat_model(monkeypatch, chat_model)

        result = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[{"role": "assistant", "content": "Anything to change?"}],
            user_message="Actually I live in Mumbai now, and I'm 30",
            model_settings=_model_settings(),
            collected_fields={"name": "Asha", "city": "Pune"},
        )

        assert result.collected_fields == {
            "name": "Asha",
            "city": "Mumbai",
            "age": "30",
        }

    @pytest.mark.parametrize(
        ("is_complete", "missing_fields", "expected_status"),
        [
            (True, [], ChatbotCompletionStatusEnum.COMPLETE),
            (True, ["city"], ChatbotCompletionStatusEnum.IN_PROGRESS),
            (False, [], ChatbotCompletionStatusEnum.IN_PROGRESS),
        ],
        ids=["complete", "complete_but_lists_missing", "not_complete"],
    )
    @pytest.mark.asyncio
    async def test_completion_requires_flag_and_no_missing_fields(
        self,
        monkeypatch: pytest.MonkeyPatch,
        is_complete: bool,
        missing_fields: list[str],
        expected_status: ChatbotCompletionStatusEnum,
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["Thanks!"],
            extractions=[
                FieldExtraction(
                    collected_fields=extracted_fields({"name": "Asha"}),
                    missing_fields=missing_fields,
                    is_complete=is_complete,
                )
            ],
        )
        patch_chat_model(monkeypatch, chat_model)

        result = await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message="I'm Asha",
            model_settings=_model_settings(),
        )

        assert result.completion_status == expected_status


class TestProgressNote:
    @pytest.mark.parametrize(
        ("extraction", "expected_note"),
        [
            (
                FieldExtraction(
                    collected_fields=extracted_fields({"name": "Asha"}),
                    missing_fields=["city"],
                    is_complete=False,
                ),
                f'{COLLECTED_PREFIX}{{"name": "Asha"}}. '
                "Still missing: city. Ask only for the missing fields.",
            ),
            (
                FieldExtraction(
                    collected_fields=extracted_fields({}),
                    missing_fields=["name", "city"],
                    is_complete=False,
                ),
                "Still missing: name, city. Ask only for the missing fields.",
            ),
            (
                FieldExtraction(
                    collected_fields=extracted_fields({"name": "Asha", "city": "Pune"}),
                    missing_fields=[],
                    is_complete=True,
                ),
                f'{COLLECTED_PREFIX}{{"name": "Asha", "city": "Pune"}}. '
                "Every required field is collected; confirm and wrap up as the "
                "flow prompt describes.",
            ),
        ],
        ids=["collected_and_missing", "missing_only", "complete"],
    )
    @pytest.mark.asyncio
    async def test_note_follows_system_prompt_in_reply_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        extraction: FieldExtraction,
        expected_note: str,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Reply."], extractions=[extraction])
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[{"role": "assistant", "content": "Hi! What's your name?"}],
            user_message="I'm Asha from Pune",
            model_settings=_model_settings(),
        )

        [reply_call] = chat_model.calls
        assert [type(m) for m in reply_call[:2]] == [SystemMessage, SystemMessage]
        assert [m.content for m in reply_call] == [
            SYSTEM_PROMPT,
            f"{PROGRESS_PREFIX}{expected_note}",
            "Begin the conversation.",
            "Hi! What's your name?",
            "I'm Asha from Pune",
        ]

    @pytest.mark.asyncio
    async def test_no_note_when_nothing_collected_missing_or_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["Reply."],
            extractions=[
                FieldExtraction(
                    collected_fields=extracted_fields({}),
                    missing_fields=[],
                    is_complete=False,
                )
            ],
        )
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message="hello",
            model_settings=_model_settings(),
        )

        [reply_call] = chat_model.calls
        assert [m.content for m in reply_call] == [SYSTEM_PROMPT, "hello"]

    @pytest.mark.asyncio
    async def test_prior_fields_survive_into_note_when_extraction_adds_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Which city?"])
        patch_chat_model(monkeypatch, chat_model)

        await run_chatbot_turn(
            system_prompt=SYSTEM_PROMPT,
            history=[{"role": "assistant", "content": "Which city?"}],
            user_message="hmm",
            model_settings=_model_settings(),
            collected_fields={"name": "Asha"},
        )

        [reply_call] = chat_model.calls
        assert reply_call[1].content == (
            f'{PROGRESS_PREFIX}{COLLECTED_PREFIX}{{"name": "Asha"}}.'
        )


class TestFieldExtractionSchema:
    def test_collected_fields_is_a_list_of_name_value_pairs(self) -> None:
        # Anthropic's json_schema mode forces additionalProperties=false on every
        # object, so a free-form dict here would only ever come back empty.
        schema = FieldExtraction.model_json_schema()
        collected = schema["properties"]["collected_fields"]
        item = schema["$defs"][collected["items"]["$ref"].split("/")[-1]]

        assert collected["type"] == "array"
        assert set(item["properties"]) == {"name", "value"}
        assert set(item["required"]) == {"name", "value"}


class TestExtractionErrors:
    @pytest.mark.parametrize(
        ("extraction_outcome", "expected_message"),
        [
            (
                OutputParserException("not valid JSON"),
                "[ANTHROPIC] Field extraction returned malformed structured output",
            ),
            (
                {"collected_fields": {}, "missing_fields": [], "is_complete": True},
                "[ANTHROPIC] Field extraction returned an unexpected output shape",
            ),
            (
                ValueError("effort is not supported for this model"),
                "[KAAPI] Unsupported model configuration for 'claude-sonnet-4-6': "
                "effort is not supported for this model",
            ),
            (
                anthropic.APITimeoutError(request=_ANTHROPIC_REQUEST),
                "(code: APITimeoutError)",
            ),
        ],
        ids=["output_parser", "wrong_type", "value_error", "anthropic_timeout"],
    )
    @pytest.mark.asyncio
    async def test_extraction_failure_becomes_provider_error_before_reply(
        self,
        monkeypatch: pytest.MonkeyPatch,
        extraction_outcome: object,
        expected_message: str,
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["should never be used"], extractions=[extraction_outcome]
        )
        patch_chat_model(monkeypatch, chat_model)

        with pytest.raises(ChatbotProviderError) as exc_info:
            await run_chatbot_turn(
                system_prompt=SYSTEM_PROMPT,
                history=[],
                user_message="I'm Asha",
                model_settings=_model_settings(),
            )

        assert expected_message in str(exc_info.value)
        assert chat_model.calls == []
