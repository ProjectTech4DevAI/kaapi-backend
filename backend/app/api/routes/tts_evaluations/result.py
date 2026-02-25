"""TTS result feedback API routes."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.tts_evaluations import (
    get_tts_result_by_id,
    update_tts_human_feedback,
)
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

    # Update feedback
    result = update_tts_human_feedback(
        session=session,
        result_id=result_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        is_correct=feedback.is_correct,
        comment=feedback.comment,
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

    return APIResponse.success_response(data=TTSResultPublic.from_model(result))
