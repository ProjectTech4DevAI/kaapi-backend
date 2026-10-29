"""Onboarding turn: extract the flow's fields, then reply -- two LangGraph nodes over a LangChain chat model."""

import asyncio
import json
from typing import Any

import anthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ValidationError

from app.services.chatbot.agents.onboarding.prompts import (
    EXTRACTION_FLOW_PROMPT_TEMPLATE,
    EXTRACTION_INSTRUCTION,
    EXTRACTION_METHOD,
    EXTRACTION_PRIOR_FIELDS_TEMPLATE,
    PROGRESS_COLLECTED_TEMPLATE,
    PROGRESS_COMPLETE_NOTE,
    PROGRESS_MISSING_TEMPLATE,
    PROGRESS_NOTE_PREFIX,
)
from app.services.chatbot.agents.onboarding.types import (
    FieldExtraction,
    OnboardingState,
)
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.history import (
    extract_final_reply,
    has_user_turn,
    insert_after_leading_system,
    summarize_if_needed,
    to_langchain_messages,
)
from app.services.chatbot.types import ChatbotMessageDict, ChatbotTurnResult
from app.services.chatbot.utils import ChatbotModelSettings, get_chat_model

ExtractionModel = Runnable[LanguageModelInput, dict | BaseModel]

EXTRACT_FIELDS_NODE = "extract_fields_node"
CHATBOT_NODE = "chatbot_node"


def _format_fields(fields: dict[str, Any]) -> str:
    return json.dumps(fields, ensure_ascii=False, default=str)


def _build_extraction_messages(
    *,
    flow_prompt: str,
    prior_fields: dict[str, Any],
    conversation_view: list[BaseMessage],
) -> list[BaseMessage]:
    instruction = (
        EXTRACTION_INSTRUCTION
        + EXTRACTION_FLOW_PROMPT_TEMPLATE.format(flow_prompt=flow_prompt)
        + EXTRACTION_PRIOR_FIELDS_TEMPLATE.format(
            prior_fields=_format_fields(prior_fields)
        )
    )
    # The view's own first message is the flow prompt, already embedded above.
    return [SystemMessage(content=instruction), *conversation_view[1:]]


def _build_progress_note(state: OnboardingState) -> SystemMessage | None:
    if "is_complete" not in state:
        return None

    note = ""
    collected = state["collected_fields"]
    if collected:
        note += PROGRESS_COLLECTED_TEMPLATE.format(collected=_format_fields(collected))
    missing = state.get("missing_fields", [])
    if state["is_complete"]:
        note += PROGRESS_COMPLETE_NOTE
    elif missing:
        note += PROGRESS_MISSING_TEMPLATE.format(missing=", ".join(missing))
    if not note:
        return None
    return SystemMessage(content=f"{PROGRESS_NOTE_PREFIX}{note.strip()}")


def _invoke_extraction(
    extraction_model: ExtractionModel,
    messages: list[BaseMessage],
) -> FieldExtraction:
    try:
        extraction = extraction_model.invoke(messages)
    except (OutputParserException, ValidationError) as e:
        error_message = (
            "[ANTHROPIC] Field extraction returned malformed structured output — "
            f"retry the turn. If persistent, contact Kaapi. (code: {type(e).__name__})"
        )
        raise ChatbotProviderError(error_message) from e

    if not isinstance(extraction, FieldExtraction):
        error_message = (
            "[ANTHROPIC] Field extraction returned an unexpected output shape — retry "
            "the turn. If persistent, contact Kaapi. (code: UnexpectedOutputType)"
        )
        raise ChatbotProviderError(error_message)
    return extraction


def build_onboarding_graph(
    chat_model: BaseChatModel,
    extraction_model: ExtractionModel,
    *,
    flow_prompt: str,
    history_token_limit: int,
) -> CompiledStateGraph:
    """extract_fields_node -> chatbot_node; both models are fixed for the graph's lifetime."""

    def extract_fields_node(state: OnboardingState) -> dict[str, Any]:
        if not has_user_turn(state["messages"]):
            return {}

        conversation_view = summarize_if_needed(
            chat_model, state["messages"], history_token_limit=history_token_limit
        )
        prior_fields = state["collected_fields"]
        extraction = _invoke_extraction(
            extraction_model,
            _build_extraction_messages(
                flow_prompt=flow_prompt,
                prior_fields=prior_fields,
                conversation_view=conversation_view,
            ),
        )
        extracted = {field.name: field.value for field in extraction.collected_fields}
        merged_fields = {**prior_fields, **extracted}
        is_complete = extraction.is_complete and not extraction.missing_fields
        return {
            "context_messages": conversation_view,
            "collected_fields": merged_fields,
            "missing_fields": extraction.missing_fields,
            "is_complete": is_complete,
        }

    def chatbot_node(state: OnboardingState) -> dict[str, list[BaseMessage]]:
        messages = state.get("context_messages")
        if messages is None:
            messages = summarize_if_needed(
                chat_model, state["messages"], history_token_limit=history_token_limit
            )
        progress_note = _build_progress_note(state)
        if progress_note is not None:
            messages = insert_after_leading_system(messages, progress_note)
        response = chat_model.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(OnboardingState)
    graph.add_node(EXTRACT_FIELDS_NODE, extract_fields_node)
    graph.add_node(CHATBOT_NODE, chatbot_node)
    graph.add_edge(START, EXTRACT_FIELDS_NODE)
    graph.add_edge(EXTRACT_FIELDS_NODE, CHATBOT_NODE)
    graph.add_edge(CHATBOT_NODE, END)
    return graph.compile()


def _get_extraction_model(chat_model: BaseChatModel) -> ExtractionModel:
    return chat_model.with_structured_output(FieldExtraction, method=EXTRACTION_METHOD)


def _to_provider_error(error: Exception, *, model: str) -> ChatbotProviderError:
    if isinstance(error, anthropic.APITimeoutError):
        return ChatbotProviderError(
            "[KAAPI] Request timed out — retry smaller. If persistent, contact Kaapi. "
            "(code: APITimeoutError)"
        )
    if isinstance(error, anthropic.APIConnectionError):
        return ChatbotProviderError(
            "[KAAPI] Network/DNS issue reaching provider — check connectivity. "
            "If persistent, contact Kaapi. (code: APIConnectionError)"
        )
    if isinstance(error, anthropic.APIStatusError):
        return ChatbotProviderError(
            f"[ANTHROPIC] Chat model call failed: {error.message} "
            f"(code: {error.status_code}, request_id: {error.request_id})"
        )
    return ChatbotProviderError(
        f"[KAAPI] Unsupported model configuration for '{model}': {error} "
        "— update the config's model, thinking, or effort. (code: ValueError)"
    )


async def run_chatbot_turn(
    *,
    system_prompt: str,
    history: list[ChatbotMessageDict],
    user_message: str | None,
    model_settings: ChatbotModelSettings,
    collected_fields: dict[str, Any] | None = None,
) -> ChatbotTurnResult:
    """Run one stateless turn: update the collected fields, then reply."""
    if user_message is None and history:
        raise ChatbotInputError(
            "[KAAPI] `message` is required once `history` is non-empty; omit it only "
            "on the first call to get the opening greeting."
        )

    messages = to_langchain_messages(
        system_prompt=system_prompt, history=history, user_message=user_message
    )
    prior_fields = dict(collected_fields) if collected_fields else {}

    initial_state: OnboardingState = {
        "messages": messages,
        "collected_fields": prior_fields,
    }
    try:
        chat_model = get_chat_model(model_settings)
        graph = build_onboarding_graph(
            chat_model,
            _get_extraction_model(chat_model),
            flow_prompt=system_prompt,
            history_token_limit=model_settings.history_token_limit,
        )
        # ChatAnthropic.invoke is sync; keep it off the event loop.
        result = await asyncio.to_thread(graph.invoke, initial_state)
    except (anthropic.APIConnectionError, anthropic.APIStatusError, ValueError) as e:
        raise _to_provider_error(e, model=model_settings.model) from e

    return ChatbotTurnResult(
        reply=extract_final_reply(result["messages"]),
        collected_fields=result["collected_fields"],
        is_complete=result.get("is_complete", False),
    )
