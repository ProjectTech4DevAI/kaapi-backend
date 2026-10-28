"""Anthropic chat-model error mapping shared by every chatbot agent."""

import logging

import anthropic

from app.services.chatbot.exceptions import ChatbotProviderError

logger = logging.getLogger(__name__)

PROVIDER_SERVER_ERROR_STATUS = 500

# ValueError covers langchain_anthropic's client-side rejection of a model config
# (e.g. thinking/effort the model doesn't support) before any request is sent.
CHAT_MODEL_ERRORS: tuple[type[Exception], ...] = (
    anthropic.APIConnectionError,
    anthropic.APIStatusError,
    ValueError,
)


def to_provider_error(error: Exception, *, model: str) -> ChatbotProviderError:
    # APITimeoutError subclasses APIConnectionError, so it must be checked first.
    if isinstance(error, anthropic.APITimeoutError):
        error_message = (
            "[KAAPI] Request timed out — retry smaller. If persistent, contact Kaapi. "
            "(code: APITimeoutError)"
        )
        logger.error(
            f"[to_provider_error] {error_message} | provider=anthropic, model={model}",
            exc_info=error,
        )
        return ChatbotProviderError(error_message)
    if isinstance(error, anthropic.APIConnectionError):
        error_message = (
            "[KAAPI] Network/DNS issue reaching provider — check connectivity. "
            "If persistent, contact Kaapi. (code: APIConnectionError)"
        )
        logger.error(
            f"[to_provider_error] {error_message} | provider=anthropic, model={model}",
            exc_info=error,
        )
        return ChatbotProviderError(error_message)
    if isinstance(error, anthropic.APIStatusError):
        error_message = (
            f"[ANTHROPIC] Chat model call failed: {error.message} "
            f"(code: {error.status_code}, request_id: {error.request_id})"
        )
        # 5xx is provider-side (alert-worthy); 4xx is caller's fault (noise if alerted)
        log = (
            logger.error
            if error.status_code >= PROVIDER_SERVER_ERROR_STATUS
            else logger.warning
        )
        log(
            f"[to_provider_error] {error_message} | provider=anthropic, model={model}",
            exc_info=error,
        )
        return ChatbotProviderError(error_message)

    error_message = (
        f"[KAAPI] Unsupported model configuration for '{model}': {error} "
        "— update the config's model, thinking, or effort. (code: ValueError)"
    )
    logger.warning(
        f"[to_provider_error] {error_message} | provider=anthropic, model={model}"
    )
    return ChatbotProviderError(error_message)
