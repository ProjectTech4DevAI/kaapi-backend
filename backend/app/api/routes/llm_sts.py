"""Speech-to-Speech (STS) API endpoint with RAG."""

import logging
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import Message
from app.models.llm.constants import (
    DEFAULT_RAG_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
)
from app.models.llm.request import (
    ChainBlock,
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
    LLMChainRequest,
    QueryParams,
    RAGBlockSpec,
    SpeechToSpeechRequest,
    STTBlockSpec,
    STTLLMParams,
    TextLLMParams,
    TTSBlockSpec,
    TTSLLMParams,
)
from app.services.llm.chain.utils import (
    DEFAULT_RAG_INSTRUCTIONS,
    SUPPORTED_LANGUAGE_CODES,
)
from app.services.llm.jobs import start_chain_job
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])


DEFAULT_RAG_TEMPERATURE = 0.1
DEFAULT_TTS_FORMAT = "ogg"  # mappers translate to opus (WhatsApp compatible)

BlockType = Literal["stt", "text", "tts"]


DETECTED_LANGUAGE_MARKER = "{{detected}}"


def _resolve_languages(request: SpeechToSpeechRequest) -> tuple[str, str]:
    """Pick effective input/output language codes.

    When the caller pins `output_language`, that wins.
    When `input_language` is a concrete BCP-47 code and output is omitted, the
    output mirrors the input.
    When the caller wants STT-driven detection (input is "auto" or "unknown")
    and hasn't pinned the output, we set the output to `{{detected}}` so
    `_substitute_detected_language_marker` (jobs.py) swaps in the language STT
    actually detected — or falls back to en-IN if STT couldn't detect one.
    """
    input_lang = request.input_language or "auto"
    if request.output_language:
        return input_lang, request.output_language
    if input_lang in ("auto", "unknown"):
        return input_lang, DETECTED_LANGUAGE_MARKER
    return input_lang, input_lang


def _merge_stt(user: STTLLMParams | None, input_lang: str) -> STTLLMParams:
    base = STTLLMParams(model=DEFAULT_STT_MODEL, input_language=input_lang)
    if user is None:
        return base
    overrides = user.model_dump(exclude_unset=True)
    overrides["input_language"] = input_lang  # route owns this
    return base.model_copy(update=overrides)


def _merge_rag(
    user: TextLLMParams | None, knowledge_base_ids: list[str]
) -> TextLLMParams:
    base = TextLLMParams(
        model=DEFAULT_RAG_MODEL,
        temperature=DEFAULT_RAG_TEMPERATURE,
        instructions=DEFAULT_RAG_INSTRUCTIONS,
    )
    merged = (
        base
        if user is None
        else base.model_copy(update=user.model_dump(exclude_unset=True))
    )
    return merged.model_copy(update={"knowledge_base_ids": knowledge_base_ids})


def _merge_tts(user: TTSLLMParams | None, output_lang: str) -> TTSLLMParams:
    base = TTSLLMParams(
        model=DEFAULT_TTS_MODEL,
        voice=DEFAULT_TTS_VOICE,
        language=output_lang,
        response_format=DEFAULT_TTS_FORMAT,
    )
    if user is None:
        return base
    overrides = user.model_dump(exclude_unset=True)
    overrides["language"] = output_lang  # route owns this
    return base.model_copy(update=overrides)


def _inline_call_config(
    type_: BlockType,
    params: STTLLMParams | TextLLMParams | TTSLLMParams,
    provider: str | None,
) -> LLMCallConfig:
    return LLMCallConfig(
        blob=ConfigBlob(
            completion=KaapiCompletionConfig(
                provider=provider,
                type=type_,
                params=params.model_dump(exclude_none=True),
            )
        )
    )


def _stored_call_config(config_id: UUID, config_version: int) -> LLMCallConfig:
    return LLMCallConfig(id=config_id, version=config_version)


# ---------- Per-block resolution ----------


def _resolve_stt_block(
    spec: STTBlockSpec | None, input_lang: str, provider: str | None
) -> ChainBlock:
    if spec and spec.is_stored_ref:
        config = _stored_call_config(spec.config_id, spec.config_version)
    else:
        merged = _merge_stt(spec.params if spec else None, input_lang)
        config = _inline_call_config("stt", merged, provider)
    return ChainBlock(config=config, intermediate_callback=True)


def _resolve_rag_block(
    spec: RAGBlockSpec | None,
    knowledge_base_ids: list[str],
    provider: str | None,
) -> ChainBlock:
    if spec and spec.is_stored_ref:
        config = _stored_call_config(spec.config_id, spec.config_version)
    else:
        merged = _merge_rag(spec.params if spec else None, knowledge_base_ids)
        config = _inline_call_config("text", merged, provider or "openai")
    return ChainBlock(config=config, intermediate_callback=True)


def _resolve_tts_block(
    spec: TTSBlockSpec | None, output_lang: str, provider: str | None
) -> ChainBlock:
    if spec and spec.is_stored_ref:
        config = _stored_call_config(spec.config_id, spec.config_version)
    else:
        merged = _merge_tts(spec.params if spec else None, output_lang)
        config = _inline_call_config("tts", merged, provider)
    return ChainBlock(config=config, intermediate_callback=False)


# ---------- Metadata ----------


def _model_for_metadata(
    spec: STTBlockSpec | RAGBlockSpec | TTSBlockSpec | None, default: str
) -> str:
    """Best-effort model label for logs/metadata."""
    if spec is None:
        return default
    if spec.is_stored_ref:
        return f"stored:{spec.config_id}@v{spec.config_version}"
    if spec.params and getattr(spec.params, "model", None):
        return spec.params.model
    return default


def _build_metadata(
    request: SpeechToSpeechRequest, input_lang: str, output_lang: str
) -> dict[str, Any]:
    metadata = dict(request.request_metadata or {})
    metadata.update(
        {
            "speech_to_speech": True,
            "input_language": input_lang,
            "output_language": output_lang,
            "stt_model": _model_for_metadata(request.stt, DEFAULT_STT_MODEL),
            "llm_model": _model_for_metadata(request.rag, DEFAULT_RAG_MODEL),
            "tts_model": _model_for_metadata(request.tts, DEFAULT_TTS_MODEL),
        }
    )
    return metadata


# ---------- Endpoint ----------


@router.post(
    "/llm/chain/sts",
    description=load_description("llm/speech_to_speech.md"),
    response_model=APIResponse[Message],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def speech_to_speech(
    current_user: AuthContextDep,
    session: SessionDep,
    request: SpeechToSpeechRequest,
) -> APIResponse[Message]:
    """Run the STT → RAG → TTS chain for a single voice input."""
    project_id = current_user.project_.id
    organization_id = current_user.organization_.id

    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    if (
        request.input_language
        and request.input_language not in SUPPORTED_LANGUAGE_CODES
    ):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported input language code: {request.input_language}. Supported: {', '.join(SUPPORTED_LANGUAGE_CODES)}",
        )

    if request.output_language and (
        request.output_language not in SUPPORTED_LANGUAGE_CODES
        or request.output_language in ("auto", "unknown")
    ):
        tts_supported = SUPPORTED_LANGUAGE_CODES - {"auto", "unknown"}
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported output language code: {request.output_language}. Supported: {', '.join(tts_supported)}",
        )

    input_lang, output_lang = _resolve_languages(request)

    blocks = [
        _resolve_stt_block(request.stt, input_lang, request.stt_provider),
        _resolve_rag_block(
            request.rag, request.knowledge_base_ids, request.rag_provider
        ),
        _resolve_tts_block(request.tts, output_lang, request.tts_provider),
    ]

    logger.info(
        f"[speech_to_speech] Starting STS chain | "
        f"project_id={project_id}, "
        f"input_lang={input_lang}, output_lang={output_lang}"
    )

    chain_request = LLMChainRequest(
        query=QueryParams(input=request.query),
        blocks=blocks,
        callback_url=request.callback_url,
        request_metadata=_build_metadata(request, input_lang, output_lang),
    )

    start_chain_job(
        db=session,
        request=chain_request,
        project_id=project_id,
        organization_id=organization_id,
    )

    return APIResponse.success_response(
        data=Message(
            message=(
                "Speech-to-speech processing initiated. "
                "You will receive intermediate callbacks for STT and LLM outputs, "
                "followed by the final callback with audio and text."
            )
        )
    )
