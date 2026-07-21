"""v2 evaluation run trigger — replica of the v1 trigger plus native judging."""

import logging
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Body, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.rate_monitor import monitor_rate
from app.models.evaluation import EvaluationRunPublic
from app.services.evaluations.judge import validate_and_start_judged_evaluation
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["Evaluation v2"])


@router.post(
    "",
    description=load_description("evaluation/create_evaluation_v2.md"),
    response_model=APIResponse[EvaluationRunPublic],
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("evaluations")),
    ],
)
def evaluate_v2(
    session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int = Body(..., description="ID of the evaluation dataset"),
    experiment_name: str = Body(
        ..., description="Name for this evaluation experiment/run"
    ),
    config_id: UUID = Body(..., description="Stored config ID"),
    config_version: int = Body(..., ge=1, description="Stored config version"),
) -> APIResponse[EvaluationRunPublic]:
    """Start a v2 evaluation run.

    Always fast and judged; there is no `run_mode` — batch judging is deferred.
    """

    eval_run = validate_and_start_judged_evaluation(
        session=session,
        dataset_id=dataset_id,
        run_name=experiment_name,
        config_id=config_id,
        config_version=config_version,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        trace_id=correlation_id.get() or "N/A",
    )
    return APIResponse.success_response(data=eval_run)
