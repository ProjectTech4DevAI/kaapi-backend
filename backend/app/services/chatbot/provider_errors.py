"""Anthropic chat-model error mapping shared by every chatbot agent."""

import logging
from typing import TypeVar

import anthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from app.services.chatbot.exceptions import ChatbotProviderError

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

PROVIDER_SERVER_ERROR_STATUS = 500

CHAT_MODEL_ERRORS: tuple[type[Exception], ...] = (
    anthropic.APIConnectionError,
    anthropic.APIStatusError,
    ValueError,
)


def to_provider_error(error: Exception, *, model: str) -> ChatbotProviderError:
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


def invoke_structured(
    structured_model: Runnable[LanguageModelInput, dict | BaseModel],
    messages: list[BaseMessage],
    schema: type[SchemaT],
    *,
    call_name: str,
) -> SchemaT:
    """Invoke a `with_structured_output` model, mapping bad output to ChatbotProviderError.

    Parser/validation errors are ValueError subclasses; mapped here so the caller's
    ValueError catch doesn't mislabel them as a model-configuration problem.
    """
    try:
        output = structured_model.invoke(messages)
    except (OutputParserException, ValidationError) as e:
        error_message = (
            f"[ANTHROPIC] {call_name} returned malformed structured output — "
            f"retry the turn. If persistent, contact Kaapi. (code: {type(e).__name__})"
        )
        logger.warning(f"[invoke_structured] {error_message}", exc_info=True)
        raise ChatbotProviderError(error_message) from e

    if not isinstance(output, schema):
        error_message = (
            f"[ANTHROPIC] {call_name} returned an unexpected output shape — retry "
            "the turn. If persistent, contact Kaapi. (code: UnexpectedOutputType)"
        )
        logger.warning(f"[invoke_structured] {error_message}")
        raise ChatbotProviderError(error_message)
    return output
