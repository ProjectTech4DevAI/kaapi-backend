import pytest

from app.models import ChatbotFlowNodeTypeEnum
from app.services.chatbot.agents.registry import run_agent_turn
from app.services.chatbot.exceptions import ChatbotInputError
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.chatbot import (
    FakeOpenAIClient,
    FakeVectorStores,
    ScriptedChatModel,
    patch_chat_model,
    search_hit,
    tool_call_reply,
)

SYSTEM_PROMPT = "Collect the user's name and city."
_MODEL_SETTINGS = ChatbotModelSettings(
    model="claude-sonnet-4-6", knowledge_base_ids=("vs_fees",)
)


class TestRunAgentTurn:
    @pytest.mark.asyncio
    async def test_rag_without_openai_client_rejected_before_calling_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["should never be used"])
        patch_chat_model(monkeypatch, chat_model)

        with pytest.raises(ChatbotInputError, match="needs an OpenAI client"):
            await run_agent_turn(
                node_type=ChatbotFlowNodeTypeEnum.RAG,
                system_prompt=SYSTEM_PROMPT,
                history=[],
                user_message="hello",
                model_settings=_MODEL_SETTINGS,
            )
        assert chat_model.calls == []

    @pytest.mark.asyncio
    async def test_onboarding_runs_field_extraction_and_ignores_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Which city?"])
        patch_chat_model(monkeypatch, chat_model)
        client = FakeOpenAIClient()

        result = await run_agent_turn(
            node_type=ChatbotFlowNodeTypeEnum.ONBOARDING,
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message="I'm Asha",
            model_settings=_MODEL_SETTINGS,
            openai_client=client,
        )

        assert result.reply == "Which city?"
        # Only the onboarding agent runs the structured-output extraction step.
        assert len(chat_model.extraction_calls) == 1
        assert chat_model.bound_tool_names == []
        assert client.vector_stores.calls == []

    @pytest.mark.asyncio
    async def test_rag_searches_with_the_given_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=[tool_call_reply("tuition"), "Tuition is 100."]
        )
        patch_chat_model(monkeypatch, chat_model)
        client = FakeOpenAIClient(
            vector_stores=FakeVectorStores(
                hits={
                    "vs_fees": [
                        search_hit(file_id="file-fees", filename="fees.pdf", score=0.9)
                    ]
                }
            )
        )

        result = await run_agent_turn(
            node_type=ChatbotFlowNodeTypeEnum.RAG,
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_message="What is the tuition?",
            model_settings=_MODEL_SETTINGS,
            openai_client=client,
        )

        assert result.reply == "Tuition is 100."
        assert result.sources == [{"file_id": "file-fees", "filename": "fees.pdf"}]
        assert chat_model.extraction_calls == []
