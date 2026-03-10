"""Audio file upload API routes for STT evaluation."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query

from app.api.deps import AuthContextDep, SessionDep
from app.core.cloud import get_cloud_storage
from app.api.permissions import Permission, require_permission
from app.models.file import AudioUploadResponse, FilePublic
from app.crud.file import get_file_by_id, list_files
from app.services.stt_evaluations.audio import upload_audio_file
from app.services.stt_evaluations.helpers import build_file_schema, build_file_schemas
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


@router.get(
    "/files",
    response_model=APIResponse[list[FilePublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="List audio files",
    description=load_description("stt_evaluation/list_audios.md"),
)
def list_audio(
    _session: SessionDep,
    auth_context: AuthContextDep,
    include_signed_url: bool = Query(
        False, description="Include a signed URL to access the audio file"
    ),
) -> APIResponse[list[FilePublic]]:
    """Get audio files per project if provided"""

    logger.info(
        f"[list_audio] Listing audio files | "
        f"project_id: {auth_context.project_.id}, "
        f"include_signed_url: {include_signed_url}"
    )

    storage = None
    if include_signed_url:
        storage = get_cloud_storage(
            session=_session, project_id=auth_context.project_.id
        )

    files = list_files(
        session=_session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    result = build_file_schemas(
        files=files, include_signed_url=include_signed_url, storage=storage
    )

    return APIResponse.success_response(data=result)


@router.get(
    "/files/{file_id}",
    response_model=APIResponse[FilePublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get audio file by ID",
    description=load_description("stt_evaluation/get_audio.md"),
)
def get_audio(
    _session: SessionDep,
    auth_context: AuthContextDep,
    file_id: int,
    include_signed_url: bool = Query(
        False, description="Include a signed URL to access the audio file"
    ),
) -> APIResponse[FilePublic]:
    """Get a single audio file by ID with optional signed URL."""
    logger.info(
        f"[get_audio] Getting audio file | "
        f"project_id: {auth_context.project_.id}, file_id: {file_id}, "
        f"include_signed_url: {include_signed_url}"
    )

    file = get_file_by_id(
        session=_session,
        file_id=file_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not file:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")

    storage = None
    if include_signed_url:
        storage = get_cloud_storage(
            session=_session, project_id=auth_context.project_.id
        )

    result = build_file_schema(
        file=file, include_signed_url=include_signed_url, storage=storage
    )

    return APIResponse.success_response(data=result)
