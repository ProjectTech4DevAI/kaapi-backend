"""Fast evaluation orchestration service.

This is the only place that decides whether a /evaluations request enters the
fast-eval path. It also hosts the worker-side entry point invoked by the
`run_evaluation_fast` Celery task.

See `Fast Evaluation SRD.md` for the full design.
"""

import logging
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.celery.utils import start_fast_evaluation as enqueue_fast_evaluation
from app.core.config import settings
from app.core.db import engine
from app.crud.evaluations import (
    get_dataset_by_id,
    resolve_evaluation_config,
    run_fast_evaluation,
)
from app.crud.evaluations.core import update_evaluation_run
from app.models.evaluation import EvaluationRun, EvaluationRunUpdate, RunModeEnum
from app.models.llm.request import TextLLMParams
from app.services.evaluations.evaluation import create_evaluation_run_or_409
from app.services.llm.providers import LLMProvider
from app.utils import get_langfuse_client, get_openai_client

logger = logging.getLogger(__name__)


# Error codes surfaced in HTTPException.detail so the UI can localize/branch.
ERR_CONFIG_TYPE_UNSUPPORTED = "config_type_unsupported"
ERR_DATASET_TOO_LARGE_FOR_FAST = "dataset_too_large_for_fast"


def is_dataset_fast_eligible(*, original_items_count: int) -> bool:
    """A dataset is eligible for fast mode when its unique-row count is within cap."""
    return original_items_count <= settings.EVAL_FAST_MAX_UNIQUE_ROWS


def validate_and_start_fast_evaluation(
    *,
    session: Session,
    dataset_id: int,
    run_name: str,
    config_id: UUID,
    config_version: int,
    organization_id: int,
    project_id: int,
    trace_id: str = "N/A",
) -> EvaluationRun:
    """Validate + create + dispatch a fast evaluation run.

    Validation (in order):
    1. Dataset exists and has a Langfuse id.
    2. Config resolves to a text-type OpenAI config.
    3. Dataset's original_items_count <= EVAL_FAST_MAX_UNIQUE_ROWS.
    4. (organization_id, project_id, run_name) is unique — enforced by the DB
       constraint; a collision is translated to 409 by the shared helper.

    On success the function creates the EvaluationRun row with
    `run_mode="fast"`, `status="processing"`, and enqueues the orchestrator
    task. The caller (route) returns the row immediately.
    """
    logger.info(
        f"[validate_and_start_fast_evaluation] Starting fast eval | "
        f"run_name={run_name} | dataset_id={dataset_id} | "
        f"org_id={organization_id} | project_id={project_id}"
    )

    # 1. Dataset must exist + have a Langfuse id (same as batch path).
    dataset = get_dataset_by_id(
        session=session,
        dataset_id=dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset {dataset_id} not found or not accessible to this "
                "organization/project"
            ),
        )
    if not dataset.langfuse_dataset_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset {dataset_id} has no Langfuse dataset id; cannot run "
                "evaluation."
            ),
        )

    # 2. Config must resolve and be a text OpenAI config.
    config_blob, error = resolve_evaluation_config(
        session=session,
        config_id=config_id,
        config_version=config_version,
        project_id=project_id,
    )
    if error or config_blob is None:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to resolve config: {error}",
        )
    if config_blob.completion.provider != LLMProvider.OPENAI:
        raise HTTPException(
            status_code=422,
            detail="Only 'openai' provider is supported for evaluation configs",
        )
    if config_blob.completion.type != "text":
        raise HTTPException(
            status_code=422,
            detail=ERR_CONFIG_TYPE_UNSUPPORTED,
        )

    # 3. Dataset must be small enough for fast eval.
    original_items_count = (dataset.dataset_metadata or {}).get("original_items_count")
    if original_items_count is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{ERR_DATASET_TOO_LARGE_FOR_FAST}: dataset {dataset_id} is "
                "missing 'original_items_count' metadata; cannot verify it has at "
                f"most {settings.EVAL_FAST_MAX_UNIQUE_ROWS} unique rows for fast mode."
            ),
        )
    if not is_dataset_fast_eligible(original_items_count=original_items_count):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{ERR_DATASET_TOO_LARGE_FOR_FAST}: dataset has "
                f"{original_items_count} unique rows; fast mode requires at most "
                f"{settings.EVAL_FAST_MAX_UNIQUE_ROWS}."
            ),
        )

    # 4. Create the run; the shared helper translates a duplicate run_name into 409.
    eval_run = create_evaluation_run_or_409(
        session=session,
        run_name=run_name,
        dataset_name=dataset.name,
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        organization_id=organization_id,
        project_id=project_id,
        run_mode=RunModeEnum.FAST,
        log_context="validate_and_start_fast_evaluation",
    )

    # Flip to processing before dispatching the task so the GET endpoint
    # reflects the correct state immediately.
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(status="processing"),
    )

    # Dispatch the orchestrator. If enqueue fails, mark the run as failed so it
    # doesn't linger in `processing` forever.
    try:
        enqueue_fast_evaluation(eval_run_id=eval_run.id, trace_id=trace_id)
    except Exception as exc:
        logger.error(
            f"[validate_and_start_fast_evaluation] Failed to enqueue task | "
            f"eval_run_id={eval_run.id} | error={exc}",
            exc_info=True,
        )
        update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(
                status="failed",
                error_message=f"Failed to enqueue fast eval task: {exc}",
            ),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue fast evaluation task",
        )

    return eval_run


def execute_fast_evaluation(*, eval_run_id: int) -> None:
    """Worker-side entry point: run the full fast-eval pipeline.

    Called from the `run_evaluation_fast` Celery task. Opens its own DB
    session so the task is self-contained, then resolves config + clients and
    delegates to `run_fast_evaluation` (CRUD).

    On terminal failure the run is marked `failed` with a descriptive
    error_message and the exception is re-raised so Celery records the failure
    (no automatic retry for this task — stage-level idempotency is the retry
    surface).
    """
    logger.info(f"[execute_fast_evaluation] Starting | eval_run_id={eval_run_id}")

    with Session(engine) as session:
        eval_run = session.get(EvaluationRun, eval_run_id)
        if eval_run is None:
            logger.error(
                f"[execute_fast_evaluation] EvaluationRun not found | "
                f"eval_run_id={eval_run_id}"
            )
            raise ValueError(f"EvaluationRun {eval_run_id} not found")

        if eval_run.run_mode != RunModeEnum.FAST.value:
            logger.error(
                f"[execute_fast_evaluation] Wrong run_mode for fast task | "
                f"eval_run_id={eval_run_id} | run_mode={eval_run.run_mode}"
            )
            raise ValueError(
                f"EvaluationRun {eval_run_id} has run_mode={eval_run.run_mode}, "
                f"expected 'fast'"
            )

        if eval_run.status == "completed":
            logger.info(
                f"[execute_fast_evaluation] Run already completed, skipping | "
                f"eval_run_id={eval_run_id}"
            )
            return

        try:
            config_blob, error = resolve_evaluation_config(
                session=session,
                config_id=eval_run.config_id,
                config_version=eval_run.config_version,
                project_id=eval_run.project_id,
            )
            if error or config_blob is None:
                raise ValueError(f"Failed to resolve config: {error}")

            text_params = TextLLMParams.model_validate(config_blob.completion.params)

            openai_client = get_openai_client(
                session=session,
                org_id=eval_run.organization_id,
                project_id=eval_run.project_id,
            )
            langfuse_client = get_langfuse_client(
                session=session,
                org_id=eval_run.organization_id,
                project_id=eval_run.project_id,
            )

            run_fast_evaluation(
                session=session,
                openai_client=openai_client,
                langfuse=langfuse_client,
                eval_run=eval_run,
                config=text_params,
            )

        except Exception as exc:
            logger.error(
                f"[execute_fast_evaluation] Run failed | "
                f"eval_run_id={eval_run_id} | error={exc}",
                exc_info=True,
            )
            # Re-fetch the row in case our session was rolled back.
            session.rollback()
            failed_run = session.get(EvaluationRun, eval_run_id)
            if failed_run is not None:
                update_evaluation_run(
                    session=session,
                    eval_run=failed_run,
                    update=EvaluationRunUpdate(
                        status="failed",
                        error_message=f"Fast eval failed: {exc}",
                    ),
                )
            raise
