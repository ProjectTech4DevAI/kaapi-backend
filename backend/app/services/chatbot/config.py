import logging
from typing import Any, get_args
from uuid import UUID

from fastapi import HTTPException
from openai import OpenAI
from pydantic import ValidationError
from sqlmodel import Session

from app.crud import CollectionCrud
from app.crud.config import ConfigVersionCrud
from app.models import ConfigVersion, ProviderType
from app.models.llm.constants import Provider
from app.models.llm.request import ConfigBlob, KaapiTextCompletionConfig
from app.services.chatbot.agents.rag.retriever import get_knowledge_base_client
from app.services.chatbot.exceptions import ChatbotInputError
from app.services.chatbot.utils import (
    DEFAULT_HISTORY_TOKEN_LIMIT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicEffort,
    ChatbotModelSettings,
    default_thinking,
)

logger = logging.getLogger(__name__)

_ANTHROPIC_EFFORT_LEVELS: tuple[AnthropicEffort, ...] = get_args(AnthropicEffort)
THINKING_TYPE_ENABLED = "enabled"


def _get_config_version(
    *, session: Session, project_id: int, config_id: UUID, config_version: int | None
) -> ConfigVersion:
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
    project_id: int,
    config_id: UUID,
    config_version: int | None,
    max_tokens: int | None,
    history_token_limit: int | None,
) -> ChatbotModelSettings:
    """Turn a saved config (+ request overrides) into chat-model settings."""
    version_row = _get_config_version(
        session=session,
        project_id=project_id,
        config_id=config_id,
        config_version=config_version,
    )

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
        instructions=params.instructions,
        knowledge_base_ids=tuple(params.knowledge_base_ids or ()),
        max_num_results=params.max_num_results,
    )
    return model_settings


def _validate_knowledge_base_ids(
    *, session: Session, project_id: int, knowledge_base_ids: tuple[str, ...]
) -> None:
    if not knowledge_base_ids:
        error_message = (
            "[KAAPI] The rag agent needs a config with non-empty "
            "completion.params.knowledge_base_ids; add the collection's knowledge base id."
        )
        logger.warning(
            f"[_validate_knowledge_base_ids] {error_message} | project_id: {project_id}"
        )
        raise ChatbotInputError(error_message)

    known_ids: set[str] = set()
    for collection in CollectionCrud(session, project_id).read_all():
        if collection.provider == ProviderType.openai:
            known_ids.add(collection.knowledge_base_id)

    unknown_ids: list[str] = []
    for knowledge_base_id in knowledge_base_ids:
        if knowledge_base_id not in known_ids:
            unknown_ids.append(knowledge_base_id)
    if unknown_ids:
        error_message = (
            "[KAAPI] Config knowledge_base_ids not found among this project's OpenAI "
            f"collections: {', '.join(unknown_ids)}; use the knowledge_base_id of an "
            "existing collection."
        )
        logger.warning(
            f"[_validate_knowledge_base_ids] {error_message} | project_id: {project_id}"
        )
        raise ChatbotInputError(error_message)


def prepare_rag_search_client(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    model_settings: ChatbotModelSettings,
) -> OpenAI:
    """Validate the rag agent's knowledge bases and return the client that searches them."""
    _validate_knowledge_base_ids(
        session=session,
        project_id=project_id,
        knowledge_base_ids=model_settings.knowledge_base_ids,
    )
    return get_knowledge_base_client(
        session=session, organization_id=organization_id, project_id=project_id
    )
