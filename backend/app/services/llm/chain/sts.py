"""Builds the STT -> LLM -> TTS chain request for the speech-to-speech endpoint."""

from fastapi import HTTPException
from sqlmodel import Session

from app.crud.config import ConfigVersionCrud
from app.models.llm.constants import (
    DEFAULT_RAG_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TEXT_MODELS,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    CompletionType,
    Provider,
)
from app.models.llm.request import (
    ChainBlock,
    ConfigBlob,
    LLMCallConfig,
    LLMChainRequest,
    STTLLMParams,
    TextLLMParams,
    TTSLLMParams,
    build_kaapi_completion_config,
)
from app.models.llm.sts import (
    LLMStepConfig,
    SpeechToSpeechRequest,
    STTStepConfig,
    TTSStepConfig,
)
from app.services.llm.chain.utils import DEFAULT_LLM_INSTRUCTIONS
from app.services.llm.jobs import resolve_config_blob

DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_PROVIDER = Provider.OPENAI
DEFAULT_TTS_FORMAT = "ogg"  # mappers translate to opus (WhatsApp compatible)

# Substituted in jobs.py with the language STT detected (en-IN if none).
DETECTED_LANGUAGE_MARKER = "{{detected}}"
UNRESOLVED_LANGUAGES = ("auto", "unknown")
CONVERSATION_PROVIDERS = (Provider.OPENAI, "openai-native")


def _call_config(
    step: STTStepConfig | LLMStepConfig | TTSStepConfig | None,
    completion_type: CompletionType,
    provider: str | None,
    params: STTLLMParams | TextLLMParams | TTSLLMParams,
) -> LLMCallConfig:
    if step is not None and step.is_stored_ref:
        return LLMCallConfig(id=step.id, version=step.version)
    return LLMCallConfig(
        blob=ConfigBlob(
            completion=build_kaapi_completion_config(
                provider=provider, type=completion_type, params=params
            )
        )
    )


def _validate_stored_step_type(
    session: Session,
    project_id: int,
    step: STTStepConfig | LLMStepConfig | TTSStepConfig | None,
    expected_type: CompletionType,
    step_name: str,
) -> ConfigBlob | None:
    # A stored config carries its own type; catch a mismatch here instead of mid-chain.
    if step is None or not step.is_stored_ref:
        return None
    config_crud = ConfigVersionCrud(
        session=session, project_id=project_id, config_id=step.id
    )
    blob, error = resolve_config_blob(
        config_crud, LLMCallConfig(id=step.id, version=step.version)
    )
    if error:
        raise HTTPException(status_code=422, detail=f"config.{step_name}: {error}")
    if blob.completion.type != expected_type:
        raise HTTPException(
            status_code=422,
            detail=(
                f"config.{step_name}: stored config has type "
                f"'{blob.completion.type}', expected '{expected_type.value}'"
            ),
        )
    return blob


def _uses_gcp_with_kb(blob: ConfigBlob) -> bool:
    # Same rule as LLMStepConfig: Vertex has no file-search stores, so KB ids would be dropped.
    completion = blob.completion
    return completion.provider == Provider.GOOGLE_GCP and bool(
        getattr(completion.params, "knowledge_base_ids", None)
    )


def _stt_params(step: STTStepConfig | None) -> STTLLMParams:
    if step is None:
        return STTLLMParams(model=DEFAULT_STT_MODEL, input_language="auto")
    return step.params or STTLLMParams()


def _llm_params(step: LLMStepConfig | None, provider: str) -> TextLLMParams:
    if step is None:
        return TextLLMParams(
            model=DEFAULT_RAG_MODEL,
            temperature=DEFAULT_LLM_TEMPERATURE,
            instructions=DEFAULT_LLM_INSTRUCTIONS,
        )
    params = step.params or TextLLMParams()
    if params.model is not None:
        return params
    # Blob validation rejects a model-less text config; AI Studio shares Google's default.
    default_model = (
        DEFAULT_TEXT_MODELS.get(provider) or DEFAULT_TEXT_MODELS[Provider.GOOGLE]
    )
    return params.model_copy(update={"model": default_model})


def _tts_params(
    step: TTSStepConfig | None, stt_step: STTStepConfig | None
) -> TTSLLMParams:
    if step is None:
        params = TTSLLMParams(
            model=DEFAULT_TTS_MODEL,
            voice=DEFAULT_TTS_VOICE,
            response_format=DEFAULT_TTS_FORMAT,
        )
    else:
        params = step.params or TTSLLMParams()
    if params.language is not None:
        return params

    language = DETECTED_LANGUAGE_MARKER
    if stt_step is None or not stt_step.is_stored_ref:
        input_language = _stt_params(stt_step).input_language
        if input_language is not None and input_language not in UNRESOLVED_LANGUAGES:
            language = input_language
    # Set after validation: the TTS language normaliser would mangle the marker.
    return params.model_copy(update={"language": language})


def build_sts_chain_request(
    session: Session, project_id: int, request: SpeechToSpeechRequest
) -> LLMChainRequest:
    """Validate stored step configs and build the 3-block chain request."""
    config = request.config
    stt = config.stt if config else None
    llm = config.llm if config else None
    tts = config.tts if config else None

    _validate_stored_step_type(session, project_id, stt, CompletionType.STT, "stt")
    stored_llm_blob = _validate_stored_step_type(
        session, project_id, llm, CompletionType.TEXT, "llm"
    )
    _validate_stored_step_type(session, project_id, tts, CompletionType.TTS, "tts")

    conversation = llm.conversation if llm else None
    llm_provider = (llm.provider if llm else None) or DEFAULT_LLM_PROVIDER
    if (
        conversation is not None
        and stored_llm_blob is not None
        and stored_llm_blob.completion.provider not in CONVERSATION_PROVIDERS
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "config.llm: conversation is not supported with provider "
                f"'{stored_llm_blob.completion.provider}'"
            ),
        )
    if stored_llm_blob is not None and _uses_gcp_with_kb(stored_llm_blob):
        raise HTTPException(
            status_code=422,
            detail=(
                "config.llm: knowledge_base_ids is not supported with provider "
                f"'{stored_llm_blob.completion.provider}'"
            ),
        )

    stt_block = ChainBlock(
        config=_call_config(
            stt, CompletionType.STT, stt.provider if stt else None, _stt_params(stt)
        ),
        intermediate_callback=True,
        include_provider_raw_response=request.include_provider_raw_response,
    )
    llm_block = ChainBlock(
        config=_call_config(
            llm,
            CompletionType.TEXT,
            llm_provider,
            _llm_params(llm, llm_provider),
        ),
        intermediate_callback=True,
        include_provider_raw_response=request.include_provider_raw_response,
        conversation=conversation,
    )
    tts_block = ChainBlock(
        config=_call_config(
            tts,
            CompletionType.TTS,
            tts.provider if tts else None,
            _tts_params(tts, stt),
        ),
        intermediate_callback=False,
        include_provider_raw_response=request.include_provider_raw_response,
    )

    return LLMChainRequest(
        query=request.query,
        blocks=[stt_block, llm_block, tts_block],
        callback_url=request.callback_url,
        request_metadata=request.request_metadata,
    )
