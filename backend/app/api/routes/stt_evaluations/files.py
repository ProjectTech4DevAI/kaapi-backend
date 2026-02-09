"""Audio file upload API routes for STT evaluation."""

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models.stt_evaluation import AudioUploadResponse
from app.services.stt_evaluations.audio import upload_audio_file
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/files",
    response_model=APIResponse[AudioUploadResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Upload audio file",
    description=load_description("stt_evaluation/upload_audio.md"),
)
def upload_audio(
    _session: SessionDep,
    auth_context: AuthContextDep,
    file: UploadFile = File(..., description="Audio file to upload"),
) -> APIResponse[AudioUploadResponse]:
    """Upload an audio file for STT evaluation."""
    logger.info(
        f"[upload_audio] Uploading audio file | "
        f"project_id: {auth_context.project_.id}, filename: {file.filename}"
    )

    result = upload_audio_file(
        session=_session,
        file=file,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=result)
