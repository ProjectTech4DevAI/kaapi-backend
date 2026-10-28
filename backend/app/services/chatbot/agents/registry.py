import logging
from typing import Any

from openai import OpenAI

from app.models.chatbot import ChatbotAgentTypeEnum
from app.services.chatbot.agents.onboarding.graph import run_chatbot_turn
from app.services.chatbot.agents.rag.graph import run_rag_turn
from app.services.chatbot.exceptions import ChatbotInputError
from app.services.chatbot.types import ChatbotMessageDict, ChatbotTurnResult
from app.services.chatbot.utils import ChatbotModelSettings

logger = logging.getLogger(__name__)


async def run_agent_turn(
    *,
    agent_type: ChatbotAgentTypeEnum,
    system_prompt: str,
    history: list[ChatbotMessageDict],
    user_message: str | None,
    model_settings: ChatbotModelSettings,
    collected_fields: dict[str, Any] | None = None,
    openai_client: OpenAI | None = None,
) -> ChatbotTurnResult:
    """Run one turn with the agent the session was created for."""
    if agent_type == ChatbotAgentTypeEnum.ONBOARDING:
        return await run_chatbot_turn(
            system_prompt=system_prompt,
            history=history,
            user_message=user_message,
            model_settings=model_settings,
            collected_fields=collected_fields,
        )

    if agent_type == ChatbotAgentTypeEnum.RAG:
        if openai_client is None:
            error_message = (
                "[KAAPI] The rag agent needs an OpenAI client to search its "
                "knowledge bases."
            )
            logger.error(f"[run_agent_turn] {error_message}")
            raise ChatbotInputError(error_message)
        return await run_rag_turn(
            system_prompt=system_prompt,
            history=history,
            user_message=user_message,
            model_settings=model_settings,
            openai_client=openai_client,
            collected_fields=collected_fields,
        )

    error_message = f"[KAAPI] Unsupported chatbot agent type '{agent_type}'."
    logger.error(f"[run_agent_turn] {error_message}")
    raise ChatbotInputError(error_message)
