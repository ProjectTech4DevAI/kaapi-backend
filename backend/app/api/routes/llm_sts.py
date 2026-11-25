"""Speech-to-Speech (STS) API endpoint: STT -> LLM -> TTS chain."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.docs.llm.examples import SPEECH_TO_SPEECH_EXAMPLES
from app.api.openapi_docs import request_examples
from app.api.permissions import Permission, require_permission
from app.crud.jobs import JobCrud
from app.models import LLMJobImmediatePublic
from app.models.llm.sts import SpeechToSpeechRequest
from app.services.llm.chain.sts import build_sts_chain_request
from app.services.llm.jobs import start_chain_job
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])


@router.post(
    "/llm/chain/sts",
    description=load_description("llm/speech_to_speech.md"),
    response_model=APIResponse[LLMJobImmediatePublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    openapi_extra=request_examples(SPEECH_TO_SPEECH_EXAMPLES),
)
def speech_to_speech(
    current_user: AuthContextDep,
    session: SessionDep,
    request: SpeechToSpeechRequest,
) -> APIResponse[LLMJobImmediatePublic]:
    """Run the STT -> LLM -> TTS chain for a single voice input."""
    project_id = current_user.project_.id
    organization_id = current_user.organization_.id

    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    chain_request = build_sts_chain_request(session, project_id, request)

    job_id = start_chain_job(
        db=session,
        request=chain_request,
        project_id=project_id,
        organization_id=organization_id,
    )
    logger.info(
        f"[speech_to_speech] Started STS chain | project_id={project_id}, job_id={job_id}"
    )

    job = JobCrud(session=session).get(job_id=job_id, project_id=project_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    message = "Your response is being generated and will be delivered via callback."
    if not request.callback_url:
        message = "Your response is being generated"

    return APIResponse.success_response(
        data=LLMJobImmediatePublic(
            job_id=job.id,
            status=job.status.value,
            message=message,
            job_inserted_at=job.inserted_at,
            job_updated_at=job.updated_at,
        )
    )
