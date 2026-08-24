"""v2 evaluation run trigger — replica of the v1 trigger plus native judging."""

import logging
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import HttpUrl

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.rate_monitor import monitor_rate
from app.models.evaluation import EvaluationRunPublic
from app.services.evaluations.fast import validate_and_start_fast_evaluation
from app.utils import APIResponse, load_description, validate_callback_url

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
    duplication_factor: int
    | None = Body(
        None,
        ge=1,
        description=(
            "Optional per-run override of the dataset's stored duplication_factor. "
            "Only supported for runtime-duplicated (v2) datasets; rejected with 422 "
            "otherwise. Omit to use the dataset's stored factor."
        ),
    ),
    callback_url: HttpUrl
    | None = Body(
        None,
        description=(
            "Optional HTTPS webhook POSTed the run's result once it reaches a "
            "terminal state (completed or failed)."
        ),
    ),
) -> APIResponse[EvaluationRunPublic]:
    """Start a v2 evaluation run.

    Always fast and judged; there is no `run_mode` — batch judging is deferred.
    Judging always runs (`is_judge_run=True`); there is no per-run judge config.
    """
    if callback_url is not None:
        try:
            validate_callback_url(str(callback_url))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"invalid_callback_url: {exc}")

    eval_run = validate_and_start_fast_evaluation(
        session=session,
        dataset_id=dataset_id,
        run_name=experiment_name,
        config_id=config_id,
        config_version=config_version,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        trace_id=correlation_id.get() or "N/A",
        is_judge_run=True,
        callback_url=str(callback_url) if callback_url else None,
        duplication_factor=duplication_factor,
    )
    return APIResponse.success_response(data=eval_run)
