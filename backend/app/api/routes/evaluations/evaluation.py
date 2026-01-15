"""Evaluation run API routes."""

import logging

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
)

from app.api.deps import AuthContextDep, SessionDep
from app.crud.evaluations import list_evaluation_runs as list_evaluation_runs_crud
from app.models.evaluation import EvaluationRunPublic
from app.api.permissions import Permission, require_permission
from app.services.evaluations import (
    get_evaluation_with_scores,
    start_evaluation,
)
from app.utils import (
    APIResponse,
    load_description,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    description=load_description("evaluation/create_evaluation.md"),
    response_model=APIResponse[EvaluationRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def evaluate(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int = Body(..., description="ID of the evaluation dataset"),
    experiment_name: str = Body(
        ..., description="Name for this evaluation experiment/run"
    ),
    config: dict = Body(default_factory=dict, description="Evaluation configuration"),
    assistant_id: str
    | None = Body(
        None, description="Optional assistant ID to fetch configuration from"
    ),
) -> APIResponse[EvaluationRunPublic]:
    """Start an evaluation run."""
    eval_run = start_evaluation(
        session=_session,
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        config=config,
        assistant_id=assistant_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if eval_run.status == "failed":
        return APIResponse.failure_response(
            error=eval_run.error_message or "Evaluation failed to start",
            data=eval_run,
        )

    return APIResponse.success_response(data=eval_run)


@router.get(
    "/",
    description=load_description("evaluation/list_evaluations.md"),
    response_model=APIResponse[list[EvaluationRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_evaluation_runs(
    _session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = 50,
    offset: int = 0,
) -> APIResponse[list[EvaluationRunPublic]]:
    """List evaluation runs."""
    logger.info(
        f"[list_evaluation_runs] Listing evaluation runs | "
        f"org_id={auth_context.organization_.id} | "
        f"project_id={auth_context.project_.id} | limit={limit} | offset={offset}"
    )

    return APIResponse.success_response(
        data=list_evaluation_runs_crud(
            session=_session,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            limit=limit,
            offset=offset,
        )
    )


@router.get(
    "/{evaluation_id}",
    description=load_description("evaluation/get_evaluation.md"),
    response_model=APIResponse[EvaluationRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_evaluation_run_status(
    evaluation_id: int,
    _session: SessionDep,
    auth_context: AuthContextDep,
    get_trace_info: bool = Query(
        False,
        description=(
            "If true, fetch and include Langfuse trace scores with Q&A context. "
            "On first request, data is fetched from Langfuse and cached. "
            "Subsequent requests return cached data."
        ),
    ),
    resync_score: bool = Query(
        False,
        description=(
            "If true, clear cached scores and re-fetch from Langfuse. "
            "Useful when new evaluators have been added or scores have been updated. "
            "Requires get_trace_info=true."
        ),
    ),
) -> APIResponse[EvaluationRunPublic]:
    """Get evaluation run status with optional trace info."""
    if resync_score and not get_trace_info:
        raise HTTPException(
            status_code=400,
            detail="resync_score=true requires get_trace_info=true",
        )

    eval_run, error = get_evaluation_with_scores(
        session=_session,
        evaluation_id=evaluation_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        get_trace_info=get_trace_info,
        resync_score=resync_score,
    )

    if not eval_run:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Evaluation run {evaluation_id} not found or not accessible "
                "to this organization"
            ),
        )

    if error:
        return APIResponse.failure_response(error=error, data=eval_run)
    return APIResponse.success_response(data=eval_run)
