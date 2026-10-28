"""RAG turn: the model answers from the knowledge base, searching it through a tool loop."""

import asyncio
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.tool_node import ToolInvocationError
from openai import OpenAI

from app.services.chatbot.agents.rag.prompts import (
    CONFIG_INSTRUCTIONS_TEMPLATE,
    GOAL_TEMPLATE,
    RAG_INSTRUCTION,
)
from app.services.chatbot.agents.rag.retriever import build_search_tool
from app.services.chatbot.agents.rag.types import KnowledgeBasePassage, RagState
from app.services.chatbot.history import (
    extract_final_reply,
    summarize_if_needed,
    to_langchain_messages,
    validate_turn_input,
)
from app.services.chatbot.provider_errors import CHAT_MODEL_ERRORS, to_provider_error
from app.services.chatbot.types import (
    ChatbotMessageDict,
    ChatbotSource,
    ChatbotTurnResult,
)
from app.services.chatbot.utils import ChatbotModelSettings, get_chat_model

ToolCallingModel = Runnable[LanguageModelInput, BaseMessage]

PREPARE_CONTEXT_NODE = "prepare_context_node"
AGENT_NODE = "agent_node"
# tools_condition routes to this exact node name.
TOOLS_NODE = "tools"

MAX_TOOL_ROUNDS = 3
# Forced tool use is rejected with thinking on, so the model picks; "none" makes it
# answer. Only ChatAnthropic passes a dict tool_choice through as-is.
TOOL_CHOICE_NONE: dict[str, str] = {"type": "none"}


def build_rag_system_prompt(*, goal: str, instructions: str | None) -> str:
    system_prompt = RAG_INSTRUCTION
    if instructions:
        system_prompt += CONFIG_INSTRUCTIONS_TEMPLATE.format(instructions=instructions)
    system_prompt += GOAL_TEMPLATE.format(goal=goal)
    return system_prompt


def _count_tool_rounds(messages: list[BaseMessage]) -> int:
    # Stored history is plain text, so every tool call in state belongs to this turn.
    rounds = 0
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            rounds += 1
    return rounds


def build_rag_graph(
    chat_model: ChatAnthropic,
    search_tool: BaseTool,
    *,
    history_token_limit: int,
) -> CompiledStateGraph:
    """prepare_context_node -> agent_node <-> tools, ending once the model stops calling tools."""
    tool_model: ToolCallingModel = chat_model.bind_tools([search_tool])
    # Tools stay bound so the prior tool_use blocks in the history remain valid.
    answer_model: ToolCallingModel = chat_model.bind_tools(
        [search_tool], tool_choice=TOOL_CHOICE_NONE
    )

    def prepare_context_node(state: RagState) -> dict[str, list[BaseMessage]]:
        # Summarized once up front so the tool loop never splits a tool_use/tool_result pair.
        messages = state["messages"]
        context_view = summarize_if_needed(
            chat_model, messages, history_token_limit=history_token_limit
        )
        if context_view is messages:
            return {}
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *context_view]}

    def agent_node(state: RagState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        model = tool_model
        if _count_tool_rounds(messages) >= MAX_TOOL_ROUNDS:
            model = answer_model
        response = model.invoke(messages)
        return {"messages": [response]}

    # Retrieval failures must fail the turn (and send a failure webhook), not be fed
    # back to the model as a tool result; only bad tool arguments are handed back.
    tool_node = ToolNode([search_tool], handle_tool_errors=ToolInvocationError)

    graph = StateGraph(RagState)
    graph.add_node(PREPARE_CONTEXT_NODE, prepare_context_node)
    graph.add_node(AGENT_NODE, agent_node)
    graph.add_node(TOOLS_NODE, tool_node)
    graph.add_edge(START, PREPARE_CONTEXT_NODE)
    graph.add_edge(PREPARE_CONTEXT_NODE, AGENT_NODE)
    graph.add_conditional_edges(
        AGENT_NODE, tools_condition, {TOOLS_NODE: TOOLS_NODE, END: END}
    )
    graph.add_edge(TOOLS_NODE, AGENT_NODE)
    return graph.compile()


def _collect_sources(messages: list[BaseMessage]) -> list[ChatbotSource]:
    sources: list[ChatbotSource] = []
    seen_file_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or not message.artifact:
            continue
        passages: list[KnowledgeBasePassage] = message.artifact
        for passage in passages:
            if passage.file_id in seen_file_ids:
                continue
            seen_file_ids.add(passage.file_id)
            sources.append({"file_id": passage.file_id, "filename": passage.filename})
    return sources


async def run_rag_turn(
    *,
    system_prompt: str,
    history: list[ChatbotMessageDict],
    user_message: str | None,
    model_settings: ChatbotModelSettings,
    openai_client: OpenAI,
    collected_fields: dict[str, Any] | None = None,
) -> ChatbotTurnResult:
    """Run one stateless turn: answer the user from the configured knowledge bases."""
    validate_turn_input(history=history, user_message=user_message)

    rag_system_prompt = build_rag_system_prompt(
        goal=system_prompt, instructions=model_settings.instructions
    )
    initial_state: RagState = {
        "messages": to_langchain_messages(
            system_prompt=rag_system_prompt, history=history, user_message=user_message
        ),
    }
    search_tool = build_search_tool(
        client=openai_client,
        knowledge_base_ids=model_settings.knowledge_base_ids,
        max_num_results=model_settings.max_num_results,
    )
    try:
        graph = build_rag_graph(
            get_chat_model(model_settings),
            search_tool,
            history_token_limit=model_settings.history_token_limit,
        )
        # ChatAnthropic.invoke and the OpenAI search are sync; keep them off the event loop.
        result = await asyncio.to_thread(graph.invoke, initial_state)
    except CHAT_MODEL_ERRORS as e:
        raise to_provider_error(e, model=model_settings.model) from e

    final_messages: list[BaseMessage] = result["messages"]
    return ChatbotTurnResult(
        reply=extract_final_reply(final_messages),
        collected_fields=dict(collected_fields) if collected_fields else {},
        is_complete=False,
        sources=_collect_sources(final_messages),
    )
