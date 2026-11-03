from dataclasses import replace
from typing import Any, get_args
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlmodel import Session

from app.crud.config import ConfigVersionCrud
from app.crud.credentials import get_provider_credential
from app.models import ConfigVersion
from app.models.llm.constants import Provider
from app.models.llm.request import ConfigBlob, KaapiTextCompletionConfig
from app.services.chatbot.exceptions import ChatbotInputError
from app.services.chatbot.utils import (
    DEFAULT_HISTORY_TOKEN_LIMIT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicEffort,
    ChatbotModelSettings,
    default_thinking,
)

_ANTHROPIC_EFFORT_LEVELS: tuple[AnthropicEffort, ...] = get_args(AnthropicEffort)
THINKING_TYPE_ENABLED = "enabled"
CREDENTIAL_API_KEY = "api_key"
ANTHROPIC_CREDENTIALS_MISSING_MESSAGE = (
    "[KAAPI] Anthropic credentials are not configured for this project; add an "
    "Anthropic API key in the project's credentials."
)


def get_anthropic_api_key(
    *, session: Session, organization_id: int, project_id: int
) -> str:
    """The project's Anthropic API key; the chatbot never falls back to a platform key."""
    credentials = get_provider_credential(
        session=session,
        org_id=organization_id,
        project_id=project_id,
        provider=Provider.ANTHROPIC.value,
    )
    if not isinstance(credentials, dict) or not credentials.get(CREDENTIAL_API_KEY):
        raise ChatbotInputError(ANTHROPIC_CREDENTIALS_MISSING_MESSAGE)
    return credentials[CREDENTIAL_API_KEY]


def get_chatbot_config_version(
    *, session: Session, project_id: int, config_id: UUID, config_version: int | None
) -> ConfigVersion:
    """The requested version, or the latest live one when `config_version` is None."""
    version_crud = ConfigVersionCrud(
        session=session, config_id=config_id, project_id=project_id
    )
    if config_version is not None:
        return version_crud.exists_or_raise(version_number=config_version)

    latest = version_crud.read_latest()
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for config '{config_id}'",
        )
    return latest


def _to_anthropic_effort(effort: str | None) -> AnthropicEffort | None:
    if effort is None:
        return None
    for level in _ANTHROPIC_EFFORT_LEVELS:
        if level == effort:
            return level
    raise ChatbotInputError(
        f"[KAAPI] Config param 'effort' value '{effort}' is not an Anthropic effort "
        f"level ({', '.join(_ANTHROPIC_EFFORT_LEVELS)}); update the config."
    )


def _validate_thinking_budget(thinking: dict[str, Any], *, max_tokens: int) -> None:
    if thinking.get("type") != THINKING_TYPE_ENABLED:
        return
    budget = thinking.get("budget_tokens")
    if not isinstance(budget, int) or budget >= max_tokens:
        raise ChatbotInputError(
            f"[KAAPI] Config thinking.budget_tokens ({budget}) must be an integer below "
            f"the chatbot's max output tokens ({max_tokens}); lower the budget or use "
            "thinking type 'adaptive'."
        )


def resolve_chatbot_model_settings(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    config_id: UUID,
    config_version: int | None,
    max_tokens: int | None,
    history_token_limit: int | None,
) -> ChatbotModelSettings:
    """Turn a saved config (+ request overrides) and the project's Anthropic key into chat-model settings."""
    version_row = get_chatbot_config_version(
        session=session,
        project_id=project_id,
        config_id=config_id,
        config_version=config_version,
    )
    model_settings = build_chatbot_model_settings(
        version_row, max_tokens=max_tokens, history_token_limit=history_token_limit
    )
    api_key = get_anthropic_api_key(
        session=session, organization_id=organization_id, project_id=project_id
    )
    return replace(model_settings, api_key=api_key)


def build_chatbot_model_settings(
    version_row: ConfigVersion,
    *,
    max_tokens: int | None,
    history_token_limit: int | None,
) -> ChatbotModelSettings:
    try:
        blob = ConfigBlob.model_validate(version_row.config_blob)
    except ValidationError as e:
        raise ChatbotInputError(
            f"[KAAPI] Stored configuration blob is invalid: {e}"
        ) from e

    completion = blob.completion
    if (
        not isinstance(completion, KaapiTextCompletionConfig)
        or completion.provider != Provider.ANTHROPIC
    ):
        raise ChatbotInputError(
            "[KAAPI] Chatbot only supports configs with completion provider "
            f"'{Provider.ANTHROPIC.value}' and type 'text'; got provider "
            f"'{completion.provider}', type '{completion.type}'."
        )

    params = completion.params
    if not params.model:
        raise ChatbotInputError(
            "[KAAPI] Config is missing completion.params.model; set a model on the config."
        )

    thinking = params.thinking if params.thinking is not None else default_thinking()
    resolved_max_tokens = (
        max_tokens if max_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS
    )
    _validate_thinking_budget(thinking, max_tokens=resolved_max_tokens)
    model_settings = ChatbotModelSettings(
        model=params.model,
        max_tokens=resolved_max_tokens,
        reasoning_effort=_to_anthropic_effort(params.effort),
        thinking=thinking,
        history_token_limit=(
            history_token_limit
            if history_token_limit is not None
            else DEFAULT_HISTORY_TOKEN_LIMIT
        ),
    )
    return model_settings
