"""Speech-to-Speech (STS) API endpoint with RAG."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import Message
from app.models.llm.request import (
    LLMChainRequest,
    QueryParams,
    SpeechToSpeechRequest,
)
from app.services.llm.chain.utils import (
    SUPPORTED_LANGUAGE_CODES,
    build_rag_block,
    build_stt_block,
    build_tts_block,
)
from app.services.llm.jobs import start_chain_job
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])


@router.post(
    "/llm/sts",
    description=load_description("llm/speech_to_speech.md"),
    response_model=APIResponse[Message],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def speech_to_speech(
    _current_user: AuthContextDep,
    _session: SessionDep,
    request: SpeechToSpeechRequest,
):
    """
    Speech-to-speech (STS) endpoint with RAG.

    Executes a 3-block chain:
    1. STT (Speech-to-Text) - Transcribes audio to text (auto-detects language for Sarvam)
    2. RAG (Retrieval-Augmented Generation) - Processes text with knowledge base
    3. TTS (Text-to-Speech) - Converts response back to audio

    Input: Voice note (WhatsApp compatible)
    Output 1: Voice note
    Output 2: text (via intermediate callback)

    """
    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    # Validate callback URL
    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    # Validate BCP-47 language codes
    if (
        request.input_language
        and request.input_language not in SUPPORTED_LANGUAGE_CODES
    ):
        return APIResponse.failure_response(
            error=f"Unsupported input language code: {request.input_language}. Supported: {', '.join(sorted(SUPPORTED_LANGUAGE_CODES))}",
            metadata={"status_code": 400},
        )

    if (
        request.output_language
        and request.output_language not in SUPPORTED_LANGUAGE_CODES
    ):
        return APIResponse.failure_response(
            error=f"Unsupported output language code: {request.output_language}. Supported: {', '.join(sorted(SUPPORTED_LANGUAGE_CODES))}",
            metadata={"status_code": 400},
        )

    # Determine language codes (already BCP-47, no conversion needed)
    input_lang_code = request.input_language or "auto"

    # If output_language not set, default to input_language
    # If input is "auto", use "{{detected}}" marker to signal TTS to use detected language
    if request.output_language:
        output_lang_code = request.output_language
    elif input_lang_code == "auto":
        output_lang_code = "{{detected}}"  # Marker to use detected language from STT
    else:
        output_lang_code = input_lang_code

    logger.info(
        f"[speech_to_speech] Starting STS chain | "
        f"project_id={project_id}, "
        f"input_lang={input_lang_code}, "
        f"output_lang={output_lang_code}, "
        f"stt_model={request.stt_model.value}, "
        f"llm_model={request.llm_model.value}, "
        f"tts_model={request.tts_model.value}"
    )

    # Build 3-block chain: STT → RAG → TTS
    blocks = [
        build_stt_block(request.stt_model, input_lang_code),
        build_rag_block(request.llm_model, request.knowledge_base_ids),
        build_tts_block(request.tts_model, output_lang_code),
    ]

    metadata = request.request_metadata or {}
    metadata.update(
        {
            "speech_to_speech": True,
            "input_language": input_lang_code,
            "output_language": output_lang_code,
            "stt_model": request.stt_model.value,
            "llm_model": request.llm_model.value,
            "tts_model": request.tts_model.value,
        }
    )

    # Create chain request
    chain_request = LLMChainRequest(
        query=QueryParams(input=request.query),
        blocks=blocks,
        callback_url=request.callback_url,
        request_metadata=metadata,
    )

    # Start async chain job
    start_chain_job(
        db=_session,
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
