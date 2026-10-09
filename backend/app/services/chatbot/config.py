import logging
from typing import get_args
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlmodel import Session

from app.crud.config import ConfigVersionCrud
from app.models import ConfigVersion
from app.models.llm.constants import Provider
from app.models.llm.request import ConfigBlob, KaapiTextCompletionConfig
from app.services.chatbot.engine import ChatbotInputError
from app.services.chatbot.utils import (
    DEFAULT_HISTORY_TOKEN_LIMIT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicEffort,
    ChatbotModelSettings,
    default_thinking,
)

logger = logging.getLogger(__name__)

_ANTHROPIC_EFFORT_LEVELS: tuple[AnthropicEffort, ...] = get_args(AnthropicEffort)


def _get_config_version(
    *, session: Session, project_id: int, config_id: UUID, config_version: int | None
) -> ConfigVersion:
    # The crud constructor and `exists_or_raise` already 404 when the config or
    # version is outside this project, so ownership is enforced here.
    version_crud = ConfigVersionCrud(
        session=session, config_id=config_id, project_id=project_id
    )
    if config_version is not None:
        return version_crud.exists_or_raise(version_number=config_version)

    latest = version_crud.read_latest()
    if latest is None:
        logger.warning(
            f"[_get_config_version] Config has no versions | "
            f"config_id: {config_id}, project_id: {project_id}"
        )
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
    # Kaapi's "none"/"minimal" have no Anthropic equivalent; dropping them
    # silently would let the provider default (often higher) take over.
    raise ChatbotInputError(
        f"[KAAPI] Config param 'effort' value '{effort}' is not an Anthropic effort "
        f"level ({', '.join(_ANTHROPIC_EFFORT_LEVELS)}); update the config."
    )


def resolve_chatbot_model_settings(
    *,
    session: Session,
    project_id: int,
    config_id: UUID,
    config_version: int | None,
    max_tokens: int | None,
    history_token_limit: int | None,
) -> ChatbotModelSettings:
    """Turn a saved config (+ request overrides) into chat-model settings.

    Raises HTTPException(404) when the config/version isn't in the project and
    ChatbotInputError when the config isn't a usable Anthropic text config.
    """
    version_row = _get_config_version(
        session=session,
        project_id=project_id,
        config_id=config_id,
        config_version=config_version,
    )

    try:
        blob = ConfigBlob.model_validate(version_row.config_blob)
    except ValidationError as e:
        logger.warning(
            f"[resolve_chatbot_model_settings] [KAAPI] Stored config blob is invalid | "
            f"config_id: {config_id}, version: {version_row.version}",
            exc_info=True,
        )
        raise ChatbotInputError(
            f"[KAAPI] Stored configuration blob is invalid: {e}"
        ) from e

    completion = blob.completion
    if (
        not isinstance(completion, KaapiTextCompletionConfig)
        or completion.provider != Provider.ANTHROPIC
    ):
        logger.warning(
            f"[resolve_chatbot_model_settings] [KAAPI] Unsupported config for chatbot | "
            f"config_id: {config_id}, version: {version_row.version}, "
            f"provider: {completion.provider}, type: {completion.type}"
        )
        raise ChatbotInputError(
            "[KAAPI] Chatbot only supports configs with completion provider "
            f"'{Provider.ANTHROPIC.value}' and type 'text'; got provider "
            f"'{completion.provider}', type '{completion.type}'."
        )

    params = completion.params
    if not params.model:
        logger.warning(
            f"[resolve_chatbot_model_settings] [KAAPI] Config has no model | "
            f"config_id: {config_id}, version: {version_row.version}"
        )
        raise ChatbotInputError(
            "[KAAPI] Config is missing completion.params.model; set a model on the config."
        )

    thinking = params.thinking if params.thinking is not None else default_thinking()
    model_settings = ChatbotModelSettings(
        model=params.model,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort=_to_anthropic_effort(params.effort),
        thinking=thinking,
        history_token_limit=(
            history_token_limit
            if history_token_limit is not None
            else DEFAULT_HISTORY_TOKEN_LIMIT
        ),
    )
    logger.info(
        f"[resolve_chatbot_model_settings] Resolved chatbot model settings | "
        f"config_id: {config_id}, version: {version_row.version}, "
        f"model: {model_settings.model}, effort: {model_settings.reasoning_effort}"
    )
    return model_settings
