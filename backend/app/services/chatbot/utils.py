"""Model/runtime settings for the v0 LangGraph + LangChain chatbot node.

Like OCS's `LLMResponseWithPrompt` node, the model and reasoning settings come
from per-request input -- here a saved Kaapi config (see
`services/chatbot/config.py`) plus optional token overrides on the request.
Only Anthropic is wired up, since the node's chat model is `ChatAnthropic`.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic

from app.core.config import settings

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_THINKING: dict[str, Any] = {"type": "adaptive"}

# OCS's `history_mode=summarize`, simplified: once the running history crosses
# this many tokens, everything except the system prompt and the most recent
# turns is summarized into one note before the next model call.
DEFAULT_HISTORY_TOKEN_LIMIT = 4000
FLOW_KEEP_RECENT_MESSAGES = 6


def default_thinking() -> dict[str, Any]:
    # Fresh copy so one request can't mutate the module-level default.
    return dict(DEFAULT_THINKING)


@dataclass(frozen=True)
class ChatbotModelSettings:
    model: str
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    # None lets the provider pick its own default effort.
    reasoning_effort: AnthropicEffort | None = None
    thinking: dict[str, Any] = field(default_factory=default_thinking)
    history_token_limit: int = DEFAULT_HISTORY_TOKEN_LIMIT


def get_chat_model(model_settings: ChatbotModelSettings) -> ChatAnthropic:
    """LangChain's provider abstraction, so swapping providers later is a
    config change here, not a rewrite of the node.
    """
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "[get_chat_model] ANTHROPIC_API_KEY is not set; export it or set it in .env"
        )
    kwargs: dict[str, Any] = {
        "model": model_settings.model,
        "max_tokens": model_settings.max_tokens,
        "thinking": model_settings.thinking,
        "anthropic_api_key": settings.ANTHROPIC_API_KEY,
    }
    if model_settings.reasoning_effort is not None:
        kwargs["reasoning_effort"] = model_settings.reasoning_effort
    return ChatAnthropic(**kwargs)
