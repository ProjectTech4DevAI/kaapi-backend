"""Conversation-history plumbing shared by every chatbot agent graph."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.models.chatbot import ChatbotMessageRoleEnum
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.types import ChatbotMessageDict
from app.services.chatbot.utils import FLOW_KEEP_RECENT_MESSAGES

logger = logging.getLogger(__name__)

SUMMARY_NOTE_PREFIX = "[Earlier conversation summary] "
SUMMARIZE_INSTRUCTION = (
    "Summarize the conversation so far in 2-3 sentences. Keep every fact the "
    "user has already given -- this note replaces the original messages."
)
SUMMARIZE_TRANSCRIPT_TEMPLATE = "<conversation>\n{rendered}\n</conversation>"
TRANSCRIPT_USER_LABEL = "User"
TRANSCRIPT_ASSISTANT_LABEL = "Assistant"

KICKOFF_MESSAGE_ID = "chatbot-kickoff"
KICKOFF_MESSAGE_CONTENT = "Begin the conversation."


def summarize_if_needed(
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

    rendered = _render_transcript(to_summarize)
    summary = chat_model.invoke(
        [
            SystemMessage(content=SUMMARIZE_INSTRUCTION),
            HumanMessage(
                content=SUMMARIZE_TRANSCRIPT_TEMPLATE.format(rendered=rendered)
            ),
        ]
    )
    summary_note = SystemMessage(content=f"{SUMMARY_NOTE_PREFIX}{summary.text}")
    return [system_message, summary_note, *recent]


def _render_transcript(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if message.id == KICKOFF_MESSAGE_ID:
            continue
        speaker = (
            TRANSCRIPT_USER_LABEL
            if isinstance(message, HumanMessage)
            else TRANSCRIPT_ASSISTANT_LABEL
        )
        lines.append(f"{speaker}: {message.text}")
    return "\n".join(lines)


def has_user_turn(messages: list[BaseMessage]) -> bool:
    for message in messages:
        if isinstance(message, HumanMessage) and message.id != KICKOFF_MESSAGE_ID:
            return True
    return False


def insert_after_leading_system(
    messages: list[BaseMessage], note: SystemMessage
) -> list[BaseMessage]:
    # Adjacent system messages merge into Anthropic's top-level `system`; a
    # trailing one is rejected as "non-consecutive" on pre-Opus-5 models.
    insert_at = 0
    while insert_at < len(messages) and isinstance(messages[insert_at], SystemMessage):
        insert_at += 1
    return [*messages[:insert_at], note, *messages[insert_at:]]


def to_langchain_messages(
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
        kickoff = HumanMessage(content=KICKOFF_MESSAGE_CONTENT, id=KICKOFF_MESSAGE_ID)
        conversation.insert(0, kickoff)

    return [SystemMessage(content=system_prompt), *conversation]


def extract_final_reply(messages: list[BaseMessage]) -> str:
    """Text of the turn's last message, which every agent graph ends on."""
    reply = ""
    if messages and isinstance(messages[-1], AIMessage):
        reply = messages[-1].text
    if not reply.strip():
        error_message = (
            "[ANTHROPIC] Chat model returned no reply text — retry the turn. "
            "If persistent, contact Kaapi. (code: EmptyReply)"
        )
        logger.warning(f"[extract_final_reply] {error_message}")
        raise ChatbotProviderError(error_message)
    return reply
