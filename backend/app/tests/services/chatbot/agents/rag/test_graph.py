import anthropic
import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.services.chatbot.agents.rag.graph import run_rag_turn
from app.services.chatbot.agents.rag.prompts import RAG_INSTRUCTION
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.chatbot import (
    FakeOpenAIClient,
    FakeVectorStores,
    ScriptedChatModel,
    patch_chat_model,
    search_hit,
    tool_call_reply,
)

GOAL = "Answer questions about school fees only from the documents."
SUMMARY_TEXT = "User asked about admissions."

_ANTHROPIC_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
_OPENAI_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/vector_stores")


def _model_settings(
    *, instructions: str | None = None, history_token_limit: int = 4000
) -> ChatbotModelSettings:
    return ChatbotModelSettings(
        model="claude-sonnet-4-6",
        instructions=instructions,
        knowledge_base_ids=("vs_fees",),
        history_token_limit=history_token_limit,
    )


def _fees_client() -> FakeOpenAIClient:
    return FakeOpenAIClient(
        vector_stores=FakeVectorStores(
            hits={
                "vs_fees": [
                    search_hit(
                        file_id="file-fees",
                        filename="fees.pdf",
                        score=0.9,
                        texts=("Tuition is 100 per term.",),
                    ),
                    search_hit(file_id="file-faq", filename="faq.pdf", score=0.4),
                ]
            }
        )
    )


async def _run(
    chat_model: ScriptedChatModel,
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: FakeOpenAIClient | None = None,
    user_message: str | None = "What is the tuition?",
    history: list[dict[str, str]] | None = None,
    model_settings: ChatbotModelSettings | None = None,
    collected_fields: dict[str, str] | None = None,
):
    patch_chat_model(monkeypatch, chat_model)
    return await run_rag_turn(
        system_prompt=GOAL,
        history=history or [],
        user_message=user_message,
        model_settings=model_settings or _model_settings(),
        openai_client=client or _fees_client(),
        collected_fields=collected_fields,
    )


class TestRunRagTurn:
    @pytest.mark.asyncio
    async def test_answers_without_searching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _fees_client()
        chat_model = ScriptedChatModel(replies=["Hello! Ask me about fees."])

        result = await _run(
            chat_model,
            monkeypatch,
            client=client,
            user_message="hi",
            collected_fields={"name": "Asha"},
        )

        assert result.reply == "Hello! Ask me about fees."
        assert result.sources == []
        assert result.is_complete is False
        assert result.collected_fields == {"name": "Asha"}
        assert client.vector_stores.calls == []
        assert chat_model.bound_tool_names[0] == ["search_knowledge_base"]

    @pytest.mark.asyncio
    async def test_tool_round_feeds_passages_back_and_returns_deduped_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _fees_client()
        chat_model = ScriptedChatModel(
            replies=[
                tool_call_reply("tuition fee", "fee payment deadline"),
                "Tuition is 100 per term.",
            ]
        )

        result = await _run(chat_model, monkeypatch, client=client)

        assert result.reply == "Tuition is 100 per term."
        # Both searches hit the same two files; each file is reported once.
        assert result.sources == [
            {"file_id": "file-fees", "filename": "fees.pdf"},
            {"file_id": "file-faq", "filename": "faq.pdf"},
        ]
        assert [call["query"] for call in client.vector_stores.calls] == [
            "tuition fee",
            "fee payment deadline",
        ]
        tool_results = [m for m in chat_model.calls[1] if isinstance(m, ToolMessage)]
        assert len(tool_results) == 2
        assert (
            '<source file="fees.pdf">\nTuition is 100 per term.\n</source>'
            in tool_results[0].content
        )

    @pytest.mark.asyncio
    async def test_fourth_agent_call_is_forced_to_answer_after_three_tool_rounds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _fees_client()
        chat_model = ScriptedChatModel(
            replies=[
                tool_call_reply("tuition"),
                tool_call_reply("tuition 2024"),
                tool_call_reply("tuition per term"),
                "Tuition is 100 per term.",
            ]
        )

        result = await _run(chat_model, monkeypatch, client=client)

        assert result.reply == "Tuition is 100 per term."
        assert chat_model.call_tool_choices == [None, None, None, {"type": "none"}]
        assert len(client.vector_stores.calls) == 3

    @pytest.mark.asyncio
    async def test_invalid_tool_args_are_handed_back_to_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _fees_client()
        chat_model = ScriptedChatModel(
            replies=[tool_call_reply(""), "Could you rephrase that?"]
        )

        result = await _run(chat_model, monkeypatch, client=client)

        assert result.reply == "Could you rephrase that?"
        assert result.sources == []
        assert client.vector_stores.calls == []
        [tool_result] = [m for m in chat_model.calls[1] if isinstance(m, ToolMessage)]
        assert tool_result.status == "error"

    @pytest.mark.asyncio
    async def test_search_failure_fails_the_turn_instead_of_reaching_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeOpenAIClient(
            vector_stores=FakeVectorStores(
                error=openai.APITimeoutError(request=_OPENAI_REQUEST)
            )
        )
        chat_model = ScriptedChatModel(
            replies=[tool_call_reply("tuition"), "should never be used"]
        )

        with pytest.raises(ChatbotProviderError, match="search timed out"):
            await _run(chat_model, monkeypatch, client=client)

        assert len(chat_model.calls) == 1

    @pytest.mark.asyncio
    async def test_empty_final_reply_is_a_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ChatbotProviderError, match="EmptyReply"):
            await _run(ScriptedChatModel(replies=["  "]), monkeypatch)

    @pytest.mark.asyncio
    async def test_missing_message_with_history_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel()

        with pytest.raises(ChatbotInputError, match="`message` is required"):
            await _run(
                chat_model,
                monkeypatch,
                user_message=None,
                history=[{"role": "assistant", "content": "Hi!"}],
            )
        assert chat_model.calls == []


class TestRagSystemPrompt:
    @pytest.mark.asyncio
    async def test_config_instructions_and_goal_follow_the_rag_instruction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Namaste!"])

        await _run(
            chat_model,
            monkeypatch,
            model_settings=_model_settings(instructions="Answer in Hindi."),
        )

        system_message = chat_model.calls[0][0]
        assert isinstance(system_message, SystemMessage)
        assert system_message.content == (
            f"{RAG_INSTRUCTION}"
            "\n\n<config_instructions>\nAnswer in Hindi.\n</config_instructions>"
            f"\n\n<goal>\n{GOAL}\n</goal>"
        )

    @pytest.mark.asyncio
    async def test_without_config_instructions_only_goal_is_added(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Hello!"])

        await _run(chat_model, monkeypatch)

        system_message = chat_model.calls[0][0]
        assert system_message.content == f"{RAG_INSTRUCTION}\n\n<goal>\n{GOAL}\n</goal>"


class TestRagHistorySummarization:
    @pytest.mark.asyncio
    async def test_history_over_budget_is_summarized_before_the_tool_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        history: list[dict[str, str]] = []
        for i in range(4):
            history.append({"role": "user", "content": f"user {i}"})
            history.append({"role": "assistant", "content": f"bot {i}"})
        chat_model = ScriptedChatModel(
            replies=[SUMMARY_TEXT, tool_call_reply("tuition"), "Tuition is 100."],
            token_count=5000,
        )

        result = await _run(
            chat_model,
            monkeypatch,
            history=history,
            model_settings=_model_settings(history_token_limit=4000),
        )

        assert result.reply == "Tuition is 100."
        # One summary call, then two agent calls; the tool round is never summarized.
        _summary_call, first_agent_call, second_agent_call = chat_model.calls
        assert first_agent_call[1].content == (
            f"[Earlier conversation summary] {SUMMARY_TEXT}"
        )
        assert isinstance(first_agent_call[-1], HumanMessage)
        assert first_agent_call[-1].content == "What is the tuition?"
        assert isinstance(second_agent_call[-1], ToolMessage)
        assert "user 0" not in [m.content for m in second_agent_call]


class TestRagProviderErrorMapping:
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
                "[ANTHROPIC] Chat model call failed: Overloaded "
                "(code: 529, request_id: req_123)",
            ),
        ],
        ids=["timeout", "connection", "value_error", "status"],
    )
    @pytest.mark.asyncio
    async def test_anthropic_errors_become_provider_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
        expected_message: str,
    ) -> None:
        with pytest.raises(ChatbotProviderError) as exc_info:
            await _run(ScriptedChatModel(error=error), monkeypatch)

        assert expected_message in str(exc_info.value)
        assert exc_info.value.__cause__ is error
