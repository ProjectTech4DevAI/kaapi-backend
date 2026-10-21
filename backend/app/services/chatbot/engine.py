"""v0 chatbot: a single LangGraph node wrapping a LangChain chat model."""

import asyncio

import anthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.models.chatbot import ChatbotMessageRoleEnum
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.types import ChatbotMessageDict, ChatbotState
from app.services.chatbot.utils import (
    FLOW_KEEP_RECENT_MESSAGES,
    ChatbotModelSettings,
    get_chat_model,
)

CHATBOT_NODE = "chatbot_node"

_SUMMARY_NOTE_PREFIX = "[Earlier conversation summary] "
_SUMMARIZE_INSTRUCTION = (
    "Summarize the conversation so far in 2-3 sentences. Keep every fact the "
    "user has already given -- this note replaces the original messages."
)

_KICKOFF_MESSAGE_ID = "chatbot-kickoff"
_KICKOFF_MESSAGE_CONTENT = "Begin the conversation."


def _summarize_if_needed(
    chat_model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    history_token_limit: int,
) -> list[BaseMessage]:
    """Trim the view sent to the model once history crosses the token budget."""
    if len(messages) <= FLOW_KEEP_RECENT_MESSAGES + 1:  # +1 for the system message
        return messages
    if chat_model.get_num_tokens_from_messages(messages) <= history_token_limit:
        return messages

    system_message, *rest = messages
    to_summarize, recent = (
        rest[:-FLOW_KEEP_RECENT_MESSAGES],
        rest[-FLOW_KEEP_RECENT_MESSAGES:],
    )
    while recent and to_summarize and not isinstance(recent[0], HumanMessage):
        recent.insert(0, to_summarize.pop())
    if not to_summarize:
        return messages

    summary = chat_model.invoke(
        [SystemMessage(content=_SUMMARIZE_INSTRUCTION), *to_summarize]
    )
    summary_note = SystemMessage(content=f"{_SUMMARY_NOTE_PREFIX}{summary.text}")
    return [system_message, summary_note, *recent]


def build_chatbot_graph(
    chat_model: BaseChatModel, *, history_token_limit: int
) -> CompiledStateGraph:
    """One node, no branching -- `chat_model` is fixed for the graph's lifetime."""

    def chatbot_node(state: ChatbotState) -> dict[str, list[BaseMessage]]:
        messages = _summarize_if_needed(
            chat_model, state["messages"], history_token_limit=history_token_limit
        )
        response = chat_model.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(ChatbotState)
    graph.add_node(CHATBOT_NODE, chatbot_node)
    graph.add_edge(START, CHATBOT_NODE)
    graph.add_edge(CHATBOT_NODE, END)
    return graph.compile()


def _to_langchain_messages(
    *, system_prompt: str, history: list[ChatbotMessageDict], user_message: str | None
) -> list[BaseMessage]:
    conversation: list[BaseMessage] = []
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        if role == ChatbotMessageRoleEnum.USER:
            conversation.append(HumanMessage(content=content))
        elif role == ChatbotMessageRoleEnum.ASSISTANT:
            conversation.append(AIMessage(content=content))
        elif role != ChatbotMessageRoleEnum.SYSTEM:
            raise ChatbotInputError(f"Unsupported message role: {role}")

    if user_message is not None:
        conversation.append(HumanMessage(content=user_message))

    if not conversation or not isinstance(conversation[0], HumanMessage):
        kickoff = HumanMessage(content=_KICKOFF_MESSAGE_CONTENT, id=_KICKOFF_MESSAGE_ID)
        conversation.insert(0, kickoff)

    return [SystemMessage(content=system_prompt), *conversation]


def _to_message_dicts(
    *, system_prompt: str, messages: list[BaseMessage]
) -> list[ChatbotMessageDict]:
    transcript: list[ChatbotMessageDict] = [
        {"role": ChatbotMessageRoleEnum.SYSTEM.value, "content": system_prompt}
    ]
    for message in messages:
        if message.id == _KICKOFF_MESSAGE_ID:
            continue
        if isinstance(message, HumanMessage):
            role = ChatbotMessageRoleEnum.USER
        elif isinstance(message, AIMessage):
            role = ChatbotMessageRoleEnum.ASSISTANT
        else:
            continue
        transcript.append({"role": role.value, "content": message.text})
    return transcript


async def run_chatbot_turn(
    *,
    system_prompt: str,
    history: list[ChatbotMessageDict],
    user_message: str | None,
    model_settings: ChatbotModelSettings,
) -> list[ChatbotMessageDict]:
    """Run one stateless turn and return the full updated transcript.

    `user_message` may be None only on the very first call (empty `history`),
    which yields the bot's opening greeting.
    """
    if user_message is None and history:
        raise ChatbotInputError(
            "[KAAPI] `message` is required once `history` is non-empty; omit it only "
            "on the first call to get the opening greeting."
        )

    messages = _to_langchain_messages(
        system_prompt=system_prompt, history=history, user_message=user_message
    )

    graph = build_chatbot_graph(
        get_chat_model(model_settings),
        history_token_limit=model_settings.history_token_limit,
    )
    try:
        # ChatAnthropic.invoke is sync; keep it off the event loop.
        result = await asyncio.to_thread(graph.invoke, {"messages": messages})
    except anthropic.APITimeoutError as e:
        raise ChatbotProviderError(
            "[KAAPI] Request timed out — retry smaller. If persistent, contact Kaapi. "
            "(code: APITimeoutError)"
        ) from e
    except anthropic.APIConnectionError as e:
        raise ChatbotProviderError(
            "[KAAPI] Network/DNS issue reaching provider — check connectivity. "
            "If persistent, contact Kaapi. (code: APIConnectionError)"
        ) from e
    except anthropic.APIStatusError as e:
        raise ChatbotProviderError(
            f"[ANTHROPIC] Chat model call failed: {e.message} (code: {e.status_code}, "
            f"request_id: {e.request_id})"
        ) from e

    transcript = _to_message_dicts(
        system_prompt=system_prompt, messages=result["messages"]
    )
    return transcript
