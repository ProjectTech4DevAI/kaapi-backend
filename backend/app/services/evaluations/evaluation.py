"""Evaluation run orchestration service."""

import logging

from fastapi import HTTPException
from sqlmodel import Session

from app.crud.assistants import get_assistant_by_id
from app.crud.evaluations import (
    create_evaluation_run,
    fetch_trace_scores_from_langfuse,
    get_dataset_by_id,
    get_evaluation_run_by_id,
    save_score,
    start_evaluation_batch,
)
from app.models.evaluation import EvaluationRun
from app.utils import get_langfuse_client, get_openai_client

logger = logging.getLogger(__name__)


def build_evaluation_config(
    session: Session,
    config: dict,
    assistant_id: str | None,
    project_id: int,
) -> dict:
    """
    Build evaluation configuration from assistant or provided config.

    If assistant_id is provided, fetch assistant and merge with config.
    Config values take precedence over assistant values.

    Args:
        session: Database session
        config: Provided configuration dict
        assistant_id: Optional assistant ID to fetch configuration from
        project_id: Project ID for assistant lookup

    Returns:
        Complete evaluation configuration dict

    Raises:
        HTTPException: If assistant not found or model missing
    """
    if assistant_id:
        assistant = get_assistant_by_id(
            session=session,
            assistant_id=assistant_id,
            project_id=project_id,
        )

        if not assistant:
            raise HTTPException(
                status_code=404, detail=f"Assistant {assistant_id} not found"
            )

        logger.info(
            f"[build_evaluation_config] Found assistant in DB | id={assistant.id} | "
            f"model={assistant.model} | instructions="
            f"{assistant.instructions[:50] if assistant.instructions else 'None'}..."
        )

        # Build config from assistant (use provided config values to override if present)
        merged_config = {
            "model": config.get("model", assistant.model),
            "instructions": config.get("instructions", assistant.instructions),
            "temperature": config.get("temperature", assistant.temperature),
        }

        # Add tools if vector stores are available
        vector_store_ids = config.get(
            "vector_store_ids", assistant.vector_store_ids or []
        )
        if vector_store_ids and len(vector_store_ids) > 0:
            merged_config["tools"] = [
                {
                    "type": "file_search",
                    "vector_store_ids": vector_store_ids,
                }
            ]

        logger.info("[build_evaluation_config] Using config from assistant")
        return merged_config

    # Using provided config directly
    logger.info("[build_evaluation_config] Using provided config directly")

    # Validate that config has minimum required fields
    if not config.get("model"):
        raise HTTPException(
            status_code=400,
            detail="Config must include 'model' when assistant_id is not provided",
        )

    return config


def start_evaluation(
    session: Session,
    dataset_id: int,
    experiment_name: str,
    config: dict,
    assistant_id: str | None,
    organization_id: int,
    project_id: int,
) -> EvaluationRun:
    """
    Start an evaluation run.

    Steps:
    1. Validate dataset exists and has Langfuse ID
    2. Build config (from assistant or direct)
    3. Create evaluation run record
    4. Start batch processing

    Args:
        session: Database session
        dataset_id: ID of the evaluation dataset
        experiment_name: Name for this evaluation experiment/run
        config: Evaluation configuration
        assistant_id: Optional assistant ID to fetch configuration from
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        EvaluationRun instance

    Raises:
        HTTPException: If dataset not found or evaluation fails to start
    """
    logger.info(
        f"[start_evaluation] Starting evaluation | experiment_name={experiment_name} | "
        f"dataset_id={dataset_id} | "
        f"org_id={organization_id} | "
        f"assistant_id={assistant_id} | "
        f"config_keys={list(config.keys())}"
    )

    dataset = get_dataset_by_id(
        session=session,
        dataset_id=dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found or not accessible to this "
            f"organization/project",
        )

    logger.info(
        f"[start_evaluation] Found dataset | id={dataset.id} | name={dataset.name} | "
        f"object_store_url={'present' if dataset.object_store_url else 'None'} | "
        f"langfuse_id={dataset.langfuse_dataset_id}"
    )

    if not dataset.langfuse_dataset_id:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} does not have a Langfuse dataset ID. "
            "Please ensure Langfuse credentials were configured when the dataset was created.",
        )

    eval_config = build_evaluation_config(
        session=session,
        config=config,
        assistant_id=assistant_id,
        project_id=project_id,
    )

    openai_client = get_openai_client(
        session=session,
        org_id=organization_id,
        project_id=project_id,
    )
    langfuse = get_langfuse_client(
        session=session,
        org_id=organization_id,
        project_id=project_id,
    )

    eval_run = create_evaluation_run(
        session=session,
        run_name=experiment_name,
        dataset_name=dataset.name,
        dataset_id=dataset_id,
        config=eval_config,
        organization_id=organization_id,
        project_id=project_id,
    )

    try:
        eval_run = start_evaluation_batch(
            langfuse=langfuse,
            openai_client=openai_client,
            session=session,
            eval_run=eval_run,
            config=eval_config,
        )

        logger.info(
            f"[start_evaluation] Evaluation started successfully | "
            f"batch_job_id={eval_run.batch_job_id} | total_items={eval_run.total_items}"
        )

        return eval_run

    except Exception as e:
        logger.error(
            f"[start_evaluation] Failed to start evaluation | run_id={eval_run.id} | {e}",
            exc_info=True,
        )
        # Error is already handled in start_evaluation_batch
        session.refresh(eval_run)
        return eval_run


def get_evaluation_with_scores(
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
    get_trace_info: bool,
    resync_score: bool,
) -> tuple[EvaluationRun | None, str | None]:
    """
    Get evaluation run, optionally with trace scores from Langfuse.

    Handles caching logic for trace scores - scores are fetched on first request
    and cached in the database.

    Args:
        session: Database session
        evaluation_id: ID of the evaluation run
        organization_id: Organization ID
        project_id: Project ID
        get_trace_info: If true, fetch trace scores
        resync_score: If true, clear cached scores and re-fetch

    Returns:
        Tuple of (EvaluationRun or None, error_message or None)
    """
    logger.info(
        f"[get_evaluation_with_scores] Fetching status for evaluation run | "
        f"evaluation_id={evaluation_id} | "
        f"org_id={organization_id} | "
        f"project_id={project_id} | "
        f"get_trace_info={get_trace_info} | "
        f"resync_score={resync_score}"
    )

    eval_run = get_evaluation_run_by_id(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    if not eval_run:
        return None, None

    # Only fetch trace info for completed evaluations
    if eval_run.status != "completed":
        if get_trace_info:
            return eval_run, (
                f"Trace info is only available for completed evaluations. "
                f"Current status: {eval_run.status}"
            )
        return eval_run, None

    # Check if we already have cached summary_scores
    has_summary_scores = (
        eval_run.score is not None and "summary_scores" in eval_run.score
    )

    # If not requesting trace info, return existing score (with summary_scores)
    if not get_trace_info:
        return eval_run, None

    # Check if we already have cached traces
    has_cached_traces = eval_run.score is not None and "traces" in eval_run.score
    if not resync_score and has_cached_traces:
        return eval_run, None

    langfuse = get_langfuse_client(
        session=session,
        org_id=organization_id,
        project_id=project_id,
    )

    # Capture data needed for Langfuse fetch and DB update
    dataset_name = eval_run.dataset_name
    run_name = eval_run.run_name
    eval_run_id = eval_run.id
    existing_summary_scores = (
        eval_run.score.get("summary_scores", []) if has_summary_scores else []
    )

    try:
        langfuse_score = fetch_trace_scores_from_langfuse(
            langfuse=langfuse,
            dataset_name=dataset_name,
            run_name=run_name,
        )
    except ValueError as e:
        logger.warning(
            f"[get_evaluation_with_scores] Run not found in Langfuse | "
            f"evaluation_id={evaluation_id} | error={e}"
        )
        return eval_run, str(e)
    except Exception as e:
        logger.error(
            f"[get_evaluation_with_scores] Failed to fetch trace info | "
            f"evaluation_id={evaluation_id} | error={e}",
            exc_info=True,
        )
        return eval_run, f"Failed to fetch trace info from Langfuse: {str(e)}"

    # Merge summary_scores: existing scores + new scores from Langfuse
    # Create a map of existing scores by name
    existing_scores_map = {s["name"]: s for s in existing_summary_scores}
    langfuse_summary_scores = langfuse_score.get("summary_scores", [])

    # Merge: Langfuse scores take precedence (more up-to-date)
    for langfuse_summary in langfuse_summary_scores:
        existing_scores_map[langfuse_summary["name"]] = langfuse_summary

    merged_summary_scores = list(existing_scores_map.values())

    # Build final score with merged summary_scores and traces
    score = {
        "summary_scores": merged_summary_scores,
        "traces": langfuse_score.get("traces", []),
    }

    eval_run = save_score(
        eval_run_id=eval_run_id,
        organization_id=organization_id,
        project_id=project_id,
        score=score,
    )

    return eval_run, None
