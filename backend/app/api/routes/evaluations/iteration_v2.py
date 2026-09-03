"""v2 evaluation iteration loop trigger — chains eval -> improve-prompt -> eval."""

import logging

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.rate_monitor import monitor_rate
from app.models.evaluation_iteration import (
    EvaluationIterationCreateRequest,
    EvaluationIterationRunImmediatePublic,
)
from app.services.evaluations.iteration import validate_and_start_evaluation_iteration
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["Evaluation v2"])


@router.post(
    "/iterations",
    description=load_description("evaluation/create_evaluation_iteration_v2.md"),
    response_model=APIResponse[EvaluationIterationRunImmediatePublic],
    status_code=202,
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("evaluations")),
    ],
)
def create_evaluation_iteration_v2(
    session: SessionDep,
    auth_context: AuthContextDep,
    request: EvaluationIterationCreateRequest,
) -> APIResponse[EvaluationIterationRunImmediatePublic]:
    """Kick off a self-driving eval -> improve-prompt -> eval loop."""
    try:
        validate_callback_url(str(request.callback_url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid_callback_url: {exc}")

    iteration_run = validate_and_start_evaluation_iteration(
        session=session,
        dataset_id=request.dataset_id,
        experiment_name=request.experiment_name,
        config_id=request.config_id,
        config_version=request.config_version,
        max_rounds=request.max_rounds,
        callback_url=str(request.callback_url),
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        trace_id=correlation_id.get() or "N/A",
    )

    return APIResponse.success_response(
        data=EvaluationIterationRunImmediatePublic(
            iteration_run_id=iteration_run.id,
            status=iteration_run.status,
            message=(
                "Evaluation iteration loop is running; the round-by-round report "
                "will be delivered to your callback_url."
            ),
            inserted_at=iteration_run.inserted_at,
            updated_at=iteration_run.updated_at,
        )
    )
