from sqlmodel import Session

from app.crud import JobCrud
from app.crud.llm import create_llm_call, update_llm_call_response
from app.models import JobType, Job
from app.models.llm.response import LLMCallResponse
from app.models.llm.request import (
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
    QueryParams,
)
from app.tests.utils.utils import get_project
from app.models.llm import LLMCallRequest


def create_llm_job(db: Session) -> Job:
    """Create a persisted LLM_API job for use in tests."""
    project = get_project(db, "Dalgo")
    return JobCrud(db).create(
        job_type=JobType.LLM_API, trace_id="test-llm-trace", project_id=project.id
    )


def create_llm_call_with_response(
    db: Session,
    job_id,
    project_id: int,
    organization_id: int,
) -> LLMCallResponse:
    """
    Create a persisted LlmCall with a completed response for use in tests.

    Uses a standard OpenAI text-completion config and fixed response values
    so tests can assert against predictable data.
    """
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            params={
                "model": "gpt-4o",
                "instructions": "You are helpful.",
                "temperature": 0.7,
            },
            type="text",
        )
    )

    llm_call = create_llm_call(
        db,
        request=LLMCallRequest(
            query=QueryParams(input="What is the capital of France?"),
            config=LLMCallConfig(blob=config_blob),
        ),
        job_id=job_id,
        project_id=project_id,
        organization_id=organization_id,
        resolved_config=config_blob,
        original_provider="openai",
    )

    update_llm_call_response(
        db,
        llm_call_id=llm_call.id,
        provider_response_id="resp_abc123",
        content={"type": "text", "content": {"format": "text", "value": "Paris"}},
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": None,
        },
    )

    return llm_call


def create_llm_call_with_audio_uri_response(
    db: Session,
    job_id,
    project_id: int,
    organization_id: int,
    s3_path: str = "s3://bucket/audio/output.wav",
) -> LLMCallResponse:
    """
    Create a persisted LlmCall with audio content stored as an S3 URI.

    Simulates the TTS path where audio is uploaded to S3 and stored with
    format='uri' (internal format, must be swapped to presigned URL on read).
    """
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            params={
                "model": "gpt-4o",
                "instructions": "You are helpful.",
                "temperature": 0.7,
            },
            type="text",
        )
    )

    llm_call = create_llm_call(
        db,
        request=LLMCallRequest(
            query=QueryParams(input="Say hello"),
            config=LLMCallConfig(blob=config_blob),
        ),
        job_id=job_id,
        project_id=project_id,
        organization_id=organization_id,
        resolved_config=config_blob,
        original_provider="openai",
    )

    update_llm_call_response(
        db,
        llm_call_id=llm_call.id,
        provider_response_id="resp_tts_xyz",
        content={
            "type": "audio",
            "content": {
                "format": "uri",
                "value": s3_path,
                "mime_type": "audio/wav",
            },
        },
        usage={
            "input_tokens": 5,
            "output_tokens": 0,
            "total_tokens": 5,
            "reasoning_tokens": None,
        },
    )

    return llm_call
