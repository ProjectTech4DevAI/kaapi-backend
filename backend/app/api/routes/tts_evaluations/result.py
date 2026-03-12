"""TTS result feedback API routes."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.cloud import get_cloud_storage
from app.crud.tts_evaluations import (
    get_tts_result_by_id,
    update_tts_human_feedback,
)
from app.models.job import JobStatus
from app.models.tts_evaluation import (
    TTSFeedbackUpdate,
    TTSResultPublic,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


@router.patch(
    "/results/{result_id}",
    response_model=APIResponse[TTSResultPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Update human feedback",
    description=load_description("tts_evaluation/update_feedback.md"),
)
def update_result_feedback(
    session: SessionDep,
    auth_context: AuthContextDep,
    result_id: int,
    feedback: TTSFeedbackUpdate = Body(...),
) -> APIResponse[TTSResultPublic]:
    """Update human feedback on a TTS result."""
    logger.info(
        f"[update_result_feedback] Updating feedback | "
        f"result_id: {result_id}, is_correct: {feedback.is_correct}"
    )

    # Verify result exists and belongs to this project
    existing = get_tts_result_by_id(
        session=session,
        result_id=result_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not existing:
        raise HTTPException(status_code=404, detail="Result not found")

    if existing.status != JobStatus.SUCCESS.value:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot provide feedback on result with status '{existing.status}'. "
                "Only completed results accept feedback."
            ),
        )

    # Build kwargs only for explicitly provided fields so that
    # sending {"is_correct": null} clears the value, while omitting
    # the field leaves it unchanged.
    update_kwargs: dict = {}
    if "is_correct" in feedback.model_fields_set:
        update_kwargs["is_correct"] = feedback.is_correct
    if "comment" in feedback.model_fields_set:
        update_kwargs["comment"] = feedback.comment

    result = update_tts_human_feedback(
        session=session,
        result_id=result_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        **update_kwargs,
    )

    return APIResponse.success_response(data=TTSResultPublic.from_model(result))


@router.get(
    "/results/{result_id}",
    response_model=APIResponse[TTSResultPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get TTS result",
    description=load_description("tts_evaluation/get_result.md"),
)
def get_result(
    session: SessionDep,
    auth_context: AuthContextDep,
    result_id: int,
    include_signed_url: bool = Query(
        False, description="Include signed URL for generated audio file"
    ),
) -> APIResponse[TTSResultPublic]:
    """Get a TTS result by ID."""
    result = get_tts_result_by_id(
        session=session,
        result_id=result_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    signed_url = None
    if include_signed_url:
        storage = get_cloud_storage(
            session=session, project_id=auth_context.project_.id
        )
        signed_url = storage.get_signed_url(result.object_store_url)

    return APIResponse.success_response(
        data=TTSResultPublic.from_model(result, signed_url=signed_url)
    )
