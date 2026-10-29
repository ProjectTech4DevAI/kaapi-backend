"""Model/runtime settings for the v0 LangGraph + LangChain chatbot node."""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_THINKING: dict[str, Any] = {"type": "adaptive"}

DEFAULT_HISTORY_TOKEN_LIMIT = 4000
FLOW_KEEP_RECENT_MESSAGES = 6


def default_thinking() -> dict[str, Any]:
    return dict(DEFAULT_THINKING)


@dataclass(frozen=True)
class ChatbotModelSettings:
    model: str
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    reasoning_effort: AnthropicEffort | None = None
    thinking: dict[str, Any] = field(default_factory=default_thinking)
    history_token_limit: int = DEFAULT_HISTORY_TOKEN_LIMIT
    api_key: str = field(default="", repr=False, compare=False)


def get_chat_model(model_settings: ChatbotModelSettings) -> ChatAnthropic:
    """LangChain's provider abstraction, so swapping providers later is a
    config change here, not a rewrite of the node.
    """
    if not model_settings.api_key:
        raise RuntimeError(
            "[get_chat_model] model_settings has no api_key; build it with "
            "resolve_chatbot_model_settings so the project's Anthropic credentials are loaded"
        )
    kwargs: dict[str, Any] = {
        "model": model_settings.model,
        "max_tokens": model_settings.max_tokens,
        "thinking": model_settings.thinking,
        "anthropic_api_key": model_settings.api_key,
    }
    if model_settings.reasoning_effort is not None:
        kwargs["reasoning_effort"] = model_settings.reasoning_effort
    return ChatAnthropic(**kwargs)
