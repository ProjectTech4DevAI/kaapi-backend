"""Evaluation run API routes."""

import logging
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
)

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.rate_monitor import monitor_rate
from app.crud.evaluations import list_evaluation_runs as list_evaluation_runs_crud
from app.crud.evaluations.core import group_traces_by_question_id
from app.models.config.version import ConfigVersionPublic
from app.models.evaluation import EvaluationRunPublic, RunModeEnum
from app.models.llm.request import LLMCallConfig
from app.services.evaluations import (
    get_evaluation_with_scores,
    improve_prompt,
    validate_and_start_batch_evaluation,
    validate_and_start_fast_evaluation,
)
from app.utils import (
    APIResponse,
    load_description,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["Evaluation"])


@router.post(
    "",
    description=load_description("evaluation/create_evaluation.md"),
    response_model=APIResponse[EvaluationRunPublic],
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("evaluations")),
    ],
)
def evaluate(
    session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int = Body(..., description="ID of the evaluation dataset"),
    experiment_name: str = Body(
        ..., description="Name for this evaluation experiment/run"
    ),
    config_id: UUID = Body(..., description="Stored config ID"),
    config_version: int = Body(..., ge=1, description="Stored config version"),
    run_mode: RunModeEnum = Body(
        default=RunModeEnum.FAST,
        description="Execution mode: 'batch' or 'fast'. Omit to default to 'fast'.",
    ),
    judge_config: LLMCallConfig
    | None = Body(
        default=None,
        description=(
            "Optional per-run tailoring for the native correctness judge (fast "
            "mode only): a saved config reference (id + version) OR an ad-hoc "
            "blob (completion params + optional prompt_template), exactly one of "
            "the two. Omit to use the built-in prompt + fallback model. Never "
            "toggles judging — judging is always on for fast runs."
        ),
    ),
) -> APIResponse[EvaluationRunPublic]:
    """Start an evaluation run."""
    logger.info(
        f"[evaluate] Starting evaluation | run_mode={run_mode.value} | "
        f"experiment_name={experiment_name} | dataset_id={dataset_id} | "
        f"org_id={auth_context.organization_.id} | "
        f"project_id={auth_context.project_.id}"
    )

    if run_mode == RunModeEnum.FAST:
        eval_run = validate_and_start_fast_evaluation(
            session=session,
            dataset_id=dataset_id,
            run_name=experiment_name,
            config_id=config_id,
            config_version=config_version,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            judge_config=judge_config,
            trace_id=correlation_id.get() or "N/A",
        )
        return APIResponse.success_response(data=eval_run)

    eval_run = validate_and_start_batch_evaluation(
        session=session,
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        config_id=config_id,
        config_version=config_version,
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
    "",
    description=load_description("evaluation/list_evaluations.md"),
    response_model=APIResponse[list[EvaluationRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_evaluation_runs(
    session: SessionDep,
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
            session=session,
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
    session: SessionDep,
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
    export_format: str = Query(
        "row",
        description=(
            "Controls the Traces structure."
            "'grouped' collates repeated questions horizontally using Parent Question ID."
        ),
        enum=["row", "grouped"],
    ),
) -> APIResponse[EvaluationRunPublic]:
    """Get evaluation run status with optional trace info."""
    if resync_score and not get_trace_info:
        raise HTTPException(
            status_code=400,
            detail="resync_score=true requires get_trace_info=true",
        )
    if export_format == "grouped" and not get_trace_info:
        raise HTTPException(
            status_code=400, detail="export_format=grouped requires get_trace_info=true"
        )

    eval_run, error = get_evaluation_with_scores(
        session=session,
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
    # Formatter = grouped
    if export_format == "grouped" and eval_run.score and "traces" in eval_run.score:
        try:
            grouped_traces = group_traces_by_question_id(eval_run.score["traces"])
            eval_run.score["traces"] = grouped_traces
        except ValueError as e:
            return APIResponse.failure_response(error=str(e), data=eval_run)

    if error:
        return APIResponse.failure_response(error=error, data=eval_run)
    return APIResponse.success_response(data=eval_run)


@router.post(
    "/{evaluation_id}/improve-prompt",
    description=load_description("evaluation/improve_prompt.md"),
    response_model=APIResponse[ConfigVersionPublic],
    status_code=201,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def improve_evaluation_prompt(
    evaluation_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ConfigVersionPublic]:
    """Generate an AI-improved prompt iteration from a completed evaluation run."""
    logger.info(
        f"[improve_evaluation_prompt] Starting | evaluation_id={evaluation_id} "
        f"org_id={auth_context.organization_.id} project_id={auth_context.project_.id}"
    )

    new_version = improve_prompt(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=new_version)
