import logging
from typing import Any, Literal

from uuid import UUID
from sqlmodel import Session, select
from app.core.util import now
import base64
import json
from app.models.llm import LlmCall, LLMCallRequest, ConfigBlob
from app.models.llm.request import (
    TextInput,
    AudioInput,
    QueryInput,
    ImageInput,
    PDFInput,
)

logger = logging.getLogger(__name__)


def serialize_input(query_input: QueryInput | str) -> str:
    """Serialize query input for database storage.

    For text: stores the actual content value
    For audio: stores metadata (type, mime_type, size)
    """
    # Handle string input (should be normalized by QueryParams validator, but be defensive)
    if isinstance(query_input, str):
        return query_input
    elif isinstance(query_input, TextInput):
        return query_input.content.value
    elif isinstance(query_input, AudioInput):
        return json.dumps(
            {
                "type": "audio",
                "format": query_input.content.format,
                "mime_type": query_input.content.mime_type,
                # approximate byte size from b64encoded value
                "size_bytes": len(query_input.content.value) * 3 // 4,
            }
        )
    else:
        return str(query_input)


def create_llm_call(
    session: Session,
    *,
    request: LLMCallRequest,
    job_id: UUID,
    project_id: int,
    organization_id: int,
    resolved_config: ConfigBlob,
    original_provider: str,
) -> LlmCall:
    """
    Create a new LLM call record in the database.

    Args:
        session: Database session
        request: The LLM call request containing query and config
        job_id: Reference to the parent job
        project_id: Project this LLM call belongs to
        organization_id: Organization this LLM call belongs to
        resolved_config: The resolved configuration blob (either from stored config or ad-hoc)

    Returns:
        LlmCall: The created LLM call record
    """
    # Determine input/output types based on completion config type
    completion_config = resolved_config.completion
    completion_type = completion_config.type or (
        completion_config.params.get("type", "text")
        if isinstance(completion_config.params, dict)
        else getattr(completion_config.params, "type", "text")
    )

    input_type: Literal["text", "audio", "image", "pdf", "multimodal"]
    output_type: Literal["text", "audio", "image"] | None

    query_input = request.query.input

    if completion_type == "stt":
        input_type = "audio"
        output_type = "text"
    elif completion_type == "tts":
        input_type = "text"
        output_type = "audio"
    elif isinstance(query_input, ImageInput):
        input_type = "image"
        output_type = "text"
    elif isinstance(query_input, PDFInput):
        input_type = "pdf"
        output_type = "text"
    elif isinstance(query_input, list):
        input_type = "multimodal"
        output_type = "text"
    else:
        input_type = "text"
        output_type = "text"

    model = (
        completion_config.params.get("model", "")
        if isinstance(completion_config.params, dict)
        else getattr(completion_config.params, "model", "")
    )

    # Build config dict for storage
    config_dict: dict[str, Any]
    if request.config.is_stored_config:
        config_dict = {
            "config_id": str(request.config.id),
            "config_version": request.config.version,
        }
    else:
        config_dict = {
            "config_blob": resolved_config.model_dump(),
        }

    # Extract conversation info if present
    conversation_id = None
    auto_create = None
    if request.query.conversation:
        conversation_id = request.query.conversation.id
        auto_create = request.query.conversation.auto_create

    db_llm_call = LlmCall(
        job_id=job_id,
        project_id=project_id,
        organization_id=organization_id,
        input=serialize_input(request.query.input),
        input_type=input_type,
        output_type=output_type,
        provider=original_provider,
        model=model,
        conversation_id=conversation_id,
        auto_create=auto_create,
        config=config_dict,
    )

    session.add(db_llm_call)
    session.commit()
    session.refresh(db_llm_call)

    logger.info(
        f"[create_llm_call] Created LLM call id={db_llm_call.id}, "
        f"job_id={job_id}, provider={original_provider}, model={model}"
    )

    return db_llm_call


def update_llm_call_response(
    session: Session,
    *,
    llm_call_id: UUID,
    provider_response_id: str | None = None,
    content: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    conversation_id: str | None = None,
) -> LlmCall:
    """
    Update an LLM call record with response data.

    Args:
        session: Database session
        llm_call_id: The LLM call record ID to update
        provider_response_id: Original response ID from the provider
        content: Response content dict
        usage: Token usage dict
        conversation_id: Conversation ID if created/updated

    Returns:
        LlmCall: The updated LLM call record

    Raises:
        ValueError: If the LLM call record is not found
    """
    db_llm_call = session.get(LlmCall, llm_call_id)
    if not db_llm_call:
        raise ValueError(f"LLM call not found with id={llm_call_id}")

    if provider_response_id is not None:
        db_llm_call.provider_response_id = provider_response_id

    if content is not None:
        # For audio outputs (AudioOutput model): calculate size metadata from base64 content
        # AudioOutput serializes as: {"type": "audio", "content": {"format": "base64", "value": "...", "mime_type": "..."}}
        if content.get("type") == "audio":
            audio_value = content.get("content", {}).get("value")
            if audio_value:
                try:
                    audio_data = base64.b64decode(audio_value)
                    content["audio_size_bytes"] = len(audio_data)
                except Exception as e:
                    logger.warning(
                        f"[update_llm_call_response] Failed to calculate audio size: {e}"
                    )

        db_llm_call.content = content

    if usage is not None:
        db_llm_call.usage = usage
    if conversation_id is not None:
        db_llm_call.conversation_id = conversation_id

    db_llm_call.updated_at = now()

    session.add(db_llm_call)
    session.commit()
    session.refresh(db_llm_call)

    logger.info(f"[update_llm_call_response] Updated LLM call id={llm_call_id}")

    return db_llm_call


def get_llm_call_by_id(
    session: Session,
    llm_call_id: UUID,
    project_id: int | None = None,
) -> LlmCall | None:
    statement = select(LlmCall).where(
        LlmCall.id == llm_call_id,
        LlmCall.deleted_at.is_(None),
    )

    if project_id is not None:
        statement = statement.where(LlmCall.project_id == project_id)

    return session.exec(statement).first()


def get_llm_calls_by_job_id(
    session: Session,
    job_id: UUID,
) -> list[LlmCall]:
    statement = (
        select(LlmCall)
        .where(
            LlmCall.job_id == job_id,
            LlmCall.deleted_at.is_(None),
        )
        .order_by(LlmCall.created_at.desc())
    )

    return list(session.exec(statement).all())
