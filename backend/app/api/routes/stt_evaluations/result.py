"""STT result feedback API routes."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.stt_evaluations import (
    get_stt_result_by_id,
    update_human_feedback,
)
from app.models.stt_evaluation import (
    STTFeedbackUpdate,
    STTResultPublic,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


@router.patch(
    "/results/{result_id}",
    response_model=APIResponse[STTResultPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Update human feedback",
    description=load_description("stt_evaluation/update_feedback.md"),
)
def update_result_feedback(
    _session: SessionDep,
    auth_context: AuthContextDep,
    result_id: int,
    feedback: STTFeedbackUpdate = Body(...),
) -> APIResponse[STTResultPublic]:
    """Update human feedback on an STT result."""
    logger.info(
        f"[update_result_feedback] Updating feedback | "
        f"result_id: {result_id}, is_correct: {feedback.is_correct}"
    )

    result = update_human_feedback(
        session=_session,
        result_id=result_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        is_correct=feedback.is_correct,
        comment=feedback.comment,
    )

    return APIResponse.success_response(
        data=STTResultPublic(
            id=result.id,
            transcription=result.transcription,
            provider=result.provider,
            status=result.status,
            score=result.score,
            is_correct=result.is_correct,
            comment=result.comment,
            error_message=result.error_message,
            stt_sample_id=result.stt_sample_id,
            evaluation_run_id=result.evaluation_run_id,
            organization_id=result.organization_id,
            project_id=result.project_id,
            inserted_at=result.inserted_at,
            updated_at=result.updated_at,
        )
    )


@router.get(
    "/results/{result_id}",
    response_model=APIResponse[STTResultPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get STT result",
    description=load_description("stt_evaluation/get_result.md"),
)
def get_result(
    _session: SessionDep,
    auth_context: AuthContextDep,
    result_id: int,
) -> APIResponse[STTResultPublic]:
    """Get an STT result by ID."""
    result = get_stt_result_by_id(
        session=_session,
        result_id=result_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    return APIResponse.success_response(
        data=STTResultPublic(
            id=result.id,
            transcription=result.transcription,
            provider=result.provider,
            status=result.status,
            score=result.score,
            is_correct=result.is_correct,
            comment=result.comment,
            error_message=result.error_message,
            stt_sample_id=result.stt_sample_id,
            evaluation_run_id=result.evaluation_run_id,
            organization_id=result.organization_id,
            project_id=result.project_id,
            inserted_at=result.inserted_at,
            updated_at=result.updated_at,
        )
    )
