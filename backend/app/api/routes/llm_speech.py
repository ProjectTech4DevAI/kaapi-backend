"""Speech-to-Speech (STS) API endpoint with RAG."""

import logging

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import Message
from app.models.llm.request import (
    LLMChainRequest,
    QueryParams,
    SpeechToSpeechRequest,
)
from app.services.llm.chain.utils import (
    LANGUAGE_CODES,
    build_rag_block,
    build_stt_block,
    build_tts_block,
    get_language_code,
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
    Output: Voice note + text (via callback)

    Edge cases:
    - Empty STT output: Chain fails with clear error
    - Audio > 16MB: TTS provider will fail (caught and reported)
    - Invalid audio format: STT provider will fail (caught and reported)
    """
    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    # Validate callback URL
    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    # Validate and determine languages
    if request.input_language and request.input_language != "auto":
        if request.input_language not in LANGUAGE_CODES:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400,
                detail=f"Unsupported input language: {request.input_language}. Supported: {', '.join(LANGUAGE_CODES.keys())}",
            )

    if request.output_language and request.output_language not in LANGUAGE_CODES:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output language: {request.output_language}. Supported: {', '.join(LANGUAGE_CODES.keys())}",
        )

    input_lang_code = get_language_code(request.input_language)
    output_lang_code = get_language_code(
        request.output_language, default=request.input_language or "auto"
    )

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

    # Add metadata to track STS-specific info
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
        query=QueryParams(input=request.audio),
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
