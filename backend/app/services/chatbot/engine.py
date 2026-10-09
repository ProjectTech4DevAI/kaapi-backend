"""v0 chatbot: a single LangGraph node wrapping a LangChain chat model.

Unlike a field-by-field collector, this mirrors OCS's own approach: the
admin supplies the full prompt (goal + fields + rules, written as free text --
the same shape as OCS's `prompt` field) and the model manages the conversation
itself, reading the full message history every turn. There is no field-by-field
state machine in code here.

No tools in v0 -- OCS's node also lets the admin attach tools (e.g.
set-session-state); that is deliberately left out here and is the natural next
step once this shape is proven.

No checkpointer: like OCS, there is no LangGraph-level pause/resume. The caller
keeps the running message list itself and passes the full history into
`graph.invoke(...)` on every turn -- a fresh, complete run each time.
"""

import asyncio
import logging
from typing import Annotated, TypedDict

import anthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from app.models.chatbot import ChatbotMessageRoleEnum
from app.services.chatbot.utils import (
    FLOW_KEEP_RECENT_MESSAGES,
    ChatbotModelSettings,
    get_chat_model,
)

logger = logging.getLogger(__name__)

CHATBOT_NODE = "chatbot_node"

_SUMMARY_NOTE_PREFIX = "[Earlier conversation summary] "
_SUMMARIZE_INSTRUCTION = (
    "Summarize the conversation so far in 2-3 sentences. Keep every fact the "
    "user has already given -- this note replaces the original messages."
)

# Anthropic rejects a request whose first non-system turn isn't a user turn, so
# the opening greeting (and every later call, whose history starts with that
# greeting) needs a synthetic user turn in front. It's stripped from the
# returned transcript by this fixed id, so callers never see it.
_KICKOFF_MESSAGE_ID = "chatbot-kickoff"
_KICKOFF_MESSAGE_CONTENT = "Begin the conversation."

ChatbotMessageDict = dict[str, str]


class ChatbotState(TypedDict):
    # `add_messages` is LangGraph's prebuilt reducer for LangChain message
    # lists: it appends new messages and merges by `.id` instead of naively
    # concatenating, which is what lets callers keep re-sending the running
    # history without duplicating turns.
    messages: Annotated[list[BaseMessage], add_messages]


class ChatbotInputError(ValueError):
    """The turn request can't produce a valid model call (caller's fault)."""


class ChatbotProviderError(Exception):
    """The chat-model provider failed or was unreachable."""


def _summarize_if_needed(
    chat_model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    history_token_limit: int,
) -> list[BaseMessage]:
    """Trim the view sent to the model once history crosses the token budget.

    Simplification: this only shrinks what THIS call sends to Anthropic: the
    caller's own running history keeps growing untouched, so a long-enough
    conversation re-summarizes from scratch each turn rather than compacting
    state permanently. OCS's real history middleware persists the compression
    point instead -- worth doing if this gets reused past v0.
    """
    if len(messages) <= FLOW_KEEP_RECENT_MESSAGES + 1:  # +1 for the system message
        return messages
    if chat_model.get_num_tokens_from_messages(messages) <= history_token_limit:
        return messages

    system_message, *rest = messages
    to_summarize, recent = (
        rest[:-FLOW_KEEP_RECENT_MESSAGES],
        rest[-FLOW_KEEP_RECENT_MESSAGES:],
    )
    # Anthropic requires the first non-system turn to be a user turn. `rest`
    # always ends on this turn's unanswered human message, so it has odd
    # length; cutting an even-sized window from its tail always lands one
    # message short, opening on an assistant turn instead. Shift the boundary
    # back until `recent` starts on a human message.
    while recent and to_summarize and not isinstance(recent[0], HumanMessage):
        recent.insert(0, to_summarize.pop())
    if not to_summarize:
        return messages

    summary = chat_model.invoke(
        [SystemMessage(content=_SUMMARIZE_INSTRUCTION), *to_summarize]
    )
    logger.info(
        f"[_summarize_if_needed] Compressed history | "
        f"from: {len(to_summarize)} messages, to: 1 summary note"
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
    # `system_prompt` is the single source of the system turn; any system entry
    # echoed back in `history` (the previous response includes it) is dropped
    # so it isn't sent twice.
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
        # `.text` drops thinking/tool blocks; the wire shape carries text only.
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
        logger.warning(
            f"[run_chatbot_turn] [KAAPI] Missing user message on a non-initial turn | "
            f"history_count: {len(history)}"
        )
        raise ChatbotInputError(
            "[KAAPI] `message` is required once `history` is non-empty; omit it only "
            "on the first call to get the opening greeting."
        )

    messages = _to_langchain_messages(
        system_prompt=system_prompt, history=history, user_message=user_message
    )
    logger.info(
        f"[run_chatbot_turn] Starting turn | history_count: {len(history)}, "
        f"has_user_message: {user_message is not None}, model: {model_settings.model}"
    )

    graph = build_chatbot_graph(
        get_chat_model(model_settings),
        history_token_limit=model_settings.history_token_limit,
    )
    try:
        # ChatAnthropic.invoke is sync; keep it off the event loop.
        result = await asyncio.to_thread(graph.invoke, {"messages": messages})
    except anthropic.APITimeoutError as e:
        logger.error(
            f"[run_chatbot_turn] [KAAPI] Request to Anthropic timed out (code: APITimeoutError) | "
            f"provider=anthropic, model={model_settings.model}",
            exc_info=True,
        )
        raise ChatbotProviderError(
            "[KAAPI] Request timed out — retry smaller. If persistent, contact Kaapi. "
            "(code: APITimeoutError)"
        ) from e
    except anthropic.APIConnectionError as e:
        logger.error(
            f"[run_chatbot_turn] [KAAPI] Could not reach Anthropic (code: APIConnectionError) | "
            f"provider=anthropic, model={model_settings.model}",
            exc_info=True,
        )
        raise ChatbotProviderError(
            "[KAAPI] Network/DNS issue reaching provider — check connectivity. "
            "If persistent, contact Kaapi. (code: APIConnectionError)"
        ) from e
    except anthropic.APIStatusError as e:
        # 5xx is provider-side (alert-worthy); 4xx is caller's fault (noise if alerted)
        log = logger.error if e.status_code >= 500 else logger.warning
        log(
            f"[run_chatbot_turn] [ANTHROPIC] {e.message} (code: {e.status_code}) | "
            f"provider=anthropic, model={model_settings.model}, request_id={e.request_id}",
            exc_info=True,
        )
        raise ChatbotProviderError(
            f"[ANTHROPIC] Chat model call failed: {e.message} (code: {e.status_code}, "
            f"request_id: {e.request_id})"
        ) from e

    transcript = _to_message_dicts(
        system_prompt=system_prompt, messages=result["messages"]
    )
    logger.info(
        f"[run_chatbot_turn] Turn completed | transcript_count: {len(transcript)}, "
        f"reply_length: {len(transcript[-1]['content'])}"
    )
    return transcript
