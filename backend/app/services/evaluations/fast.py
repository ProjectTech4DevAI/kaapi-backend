"""Fast evaluation orchestration service.

This is the only place that decides whether a /evaluations request enters the
fast-eval path. It also hosts the worker-side entry points invoked by the
`run_evaluation_fast_chunk` and `run_evaluation_fast_aggregate` Celery tasks.

See `Fast Evaluation SRD.md` for the full design.
"""

import logging
import math
from uuid import UUID

from fastapi import HTTPException
from langfuse import Langfuse
from openai import OpenAI
from sqlmodel import Session

from app.celery.utils import start_fast_evaluation_chunk
from app.core.config import settings
from app.core.db import engine
from app.crud.evaluations import (
    get_dataset_by_id,
    resolve_evaluation_config,
    run_fast_evaluation,
)
from app.crud.evaluations.batch import fetch_dataset_items
from app.crud.evaluations.core import update_evaluation_run
from app.crud.evaluations.fast import mark_response_chunk_failed, run_response_chunk
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

    # Fetch the dataset items now to size the fan-out: ceil(total / chunk_size)
    # parallel chunk tasks drain the responses stage across workers. Any failure
    # here marks the run failed so it never lingers in `processing`.
    try:
        langfuse_client = get_langfuse_client(
            session=session,
            org_id=organization_id,
            project_id=project_id,
        )
        dataset_items = fetch_dataset_items(
            langfuse=langfuse_client, dataset_name=dataset.name
        )
        total_items = len(dataset_items)
        if total_items == 0:
            raise ValueError(f"Dataset '{dataset.name}' returned no items")
        n_chunks = math.ceil(total_items / settings.EVAL_FAST_CHUNK_SIZE)

        # total_items isn't on EvaluationRunUpdate; set it directly, then flip to
        # processing so the GET endpoint reflects state before dispatch.
        eval_run.total_items = total_items
        eval_run = update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(status="processing"),
        )

        for chunk_index in range(n_chunks):
            start_fast_evaluation_chunk(
                eval_run_id=eval_run.id,
                chunk_index=chunk_index,
                trace_id=trace_id,
            )
        logger.info(
            f"[validate_and_start_fast_evaluation] Dispatched chunk tasks | "
            f"eval_run_id={eval_run.id} | total_items={total_items} | "
            f"n_chunks={n_chunks}"
        )
    except Exception as exc:
        logger.error(
            f"[validate_and_start_fast_evaluation] Failed to start run | "
            f"eval_run_id={eval_run.id} | error={exc}",
            exc_info=True,
        )
        update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(
                status="failed",
                error_message=f"Failed to start fast eval: {exc}",
            ),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to start fast evaluation",
        )

    return eval_run


def _get_fast_run(*, session: Session, eval_run_id: int) -> EvaluationRun:
    """Load a fast-mode EvaluationRun, raising if missing or wrong run_mode."""
    eval_run = session.get(EvaluationRun, eval_run_id)
    if eval_run is None:
        raise ValueError(f"EvaluationRun {eval_run_id} not found")
    if eval_run.run_mode != RunModeEnum.FAST:
        raise ValueError(
            f"EvaluationRun {eval_run_id} has run_mode={eval_run.run_mode.value}, "
            f"expected 'fast'"
        )
    return eval_run


def _resolve_config_and_clients(
    *, session: Session, eval_run: EvaluationRun
) -> tuple[TextLLMParams, OpenAI, Langfuse]:
    """Resolve the run's text config and build its OpenAI + Langfuse clients."""
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
    return text_params, openai_client, langfuse_client


def execute_fast_evaluation_chunk(*, eval_run_id: int, chunk_index: int) -> None:
    """Worker-side entry point for one responses chunk.

    Called from `run_evaluation_fast_chunk`. Slices the dataset deterministically
    (sorted by item id so every chunk task partitions the same way) and delegates
    to `run_response_chunk`. On failure it marks ONLY this chunk's batch_job
    failed — the run's fate is decided at aggregation / by the cron healer — then
    re-raises so Celery records the failure.
    """
    logger.info(
        f"[execute_fast_evaluation_chunk] Starting | "
        f"eval_run_id={eval_run_id} | chunk_index={chunk_index}"
    )

    with Session(engine) as session:
        eval_run = _get_fast_run(session=session, eval_run_id=eval_run_id)
        if eval_run.status == "completed":
            logger.info(
                f"[execute_fast_evaluation_chunk] Run already completed, skipping | "
                f"eval_run_id={eval_run_id} | chunk_index={chunk_index}"
            )
            return

        try:
            text_params, openai_client, langfuse_client = _resolve_config_and_clients(
                session=session, eval_run=eval_run
            )
            dataset_items = fetch_dataset_items(
                langfuse=langfuse_client, dataset_name=eval_run.dataset_name
            )
            # Deterministic partition: identical order across every chunk task so
            # slices never overlap or miss items.
            dataset_items.sort(key=lambda item: item["id"])
            start = chunk_index * settings.EVAL_FAST_CHUNK_SIZE
            items_slice = dataset_items[start : start + settings.EVAL_FAST_CHUNK_SIZE]

            log_prefix = (
                f"[org={eval_run.organization_id}]"
                f"[project={eval_run.project_id}]"
                f"[eval={eval_run.id}]"
            )
            run_response_chunk(
                session=session,
                openai_client=openai_client,
                eval_run=eval_run,
                config=text_params,
                dataset_items_slice=items_slice,
                chunk_index=chunk_index,
                log_prefix=log_prefix,
            )

        except Exception as exc:
            logger.error(
                f"[execute_fast_evaluation_chunk] Chunk failed | "
                f"eval_run_id={eval_run_id} | chunk_index={chunk_index} | error={exc}",
                exc_info=True,
            )
            session.rollback()
            failed_run = session.get(EvaluationRun, eval_run_id)
            if failed_run is not None:
                mark_response_chunk_failed(
                    session=session,
                    eval_run=failed_run,
                    chunk_index=chunk_index,
                    error_message=f"Fast eval chunk failed: {exc}",
                )
            raise


def execute_fast_evaluation_aggregate(*, eval_run_id: int) -> None:
    """Worker-side entry point for the fan-in aggregate.

    Called from `run_evaluation_fast_aggregate`. Merges the response chunks and
    runs embeddings + scoring + completion via `run_fast_evaluation`. This task
    owns the run's completed/failed transition (and the notification side effect
    via `update_evaluation_run`), so on terminal failure it marks the whole run
    failed and re-raises.
    """
    logger.info(
        f"[execute_fast_evaluation_aggregate] Starting | eval_run_id={eval_run_id}"
    )

    with Session(engine) as session:
        eval_run = _get_fast_run(session=session, eval_run_id=eval_run_id)
        if eval_run.status == "completed":
            logger.info(
                f"[execute_fast_evaluation_aggregate] Run already completed, "
                f"skipping | eval_run_id={eval_run_id}"
            )
            return

        try:
            _, openai_client, langfuse_client = _resolve_config_and_clients(
                session=session, eval_run=eval_run
            )
            run_fast_evaluation(
                session=session,
                openai_client=openai_client,
                langfuse=langfuse_client,
                eval_run=eval_run,
            )

        except Exception as exc:
            logger.error(
                f"[execute_fast_evaluation_aggregate] Run failed | "
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
