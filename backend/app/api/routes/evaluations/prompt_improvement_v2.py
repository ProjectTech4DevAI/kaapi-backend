"""v2 prompt-improvement trigger — consumes three-metric judge results.

Mirrors the v1 improve-prompt route but requires a judged (v2) run and returns a
typed prompt recommendation. The v1 route is left unchanged.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models.evaluation import ImprovePromptRequest
from app.models.llm.response import LLMJobImmediatePublic
from app.services.evaluations import start_prompt_improvement_job
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["Evaluation v2"])


@router.post(
    "/{evaluation_id}/improve-prompt",
    description=load_description("evaluation/improve_prompt_v2.md"),
    response_model=APIResponse[LLMJobImmediatePublic],
    status_code=202,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def improve_evaluation_prompt_v2(
    evaluation_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
    request: ImprovePromptRequest,
) -> APIResponse[LLMJobImmediatePublic]:
    """Enqueue a v2 prompt-recommendation job for a judged evaluation run."""
    try:
        validate_callback_url(str(request.callback_url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid_callback_url: {exc}")

    job = start_prompt_improvement_job(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        callback_url=str(request.callback_url),
        require_judge_run=True,
    )

    return APIResponse.success_response(
        data=LLMJobImmediatePublic(
            job_id=job.id,
            status=job.status.value,
            message="Prompt recommendation is running; the result will be delivered to your callback_url.",
            job_inserted_at=job.inserted_at,
            job_updated_at=job.updated_at,
        )
    )
