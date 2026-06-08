"""Evaluation run orchestration service."""

import logging
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.crud.evaluations import (
    create_evaluation_run,
    fetch_trace_scores_from_langfuse,
    get_dataset_by_id,
    get_evaluation_run_by_id,
    resolve_evaluation_config,
    save_score,
    start_evaluation_batch,
)
from app.crud.evaluations.score import CategoryMetrics
from app.models.evaluation import EvaluationRun
from app.models.llm.request import TextLLMParams, STTLLMParams, TTSLLMParams
from app.services.llm.providers import LLMProvider
from app.utils import get_langfuse_client, get_openai_client
from app.core.cloud.storage import get_cloud_storage
from app.core.storage_utils import load_json_from_object_store

logger = logging.getLogger(__name__)


def _compute_category_metrics(
    traces: list[dict[str, Any]],
) -> list[CategoryMetrics]:
    """Aggregate cosine + correctness scores per category across the trace list.

    Returns one entry per distinct category with `total_evals` and the mean
    cosine / correctness scores (or `None` for either if no trace in that
    category had the score). Categories sorted alphabetically; "Other"
    always appears last so the default bucket doesn't dominate the UI.

    Backwards-compatible: traces produced before this feature don't carry a
    `category` field and land in "Other".
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for trace in traces:
        raw_category = trace.get("category") or ""
        category = raw_category.title() if raw_category else "Other"
        cosine_vals: list[float] = []
        correctness_vals: list[float] = []
        # Match by substring so we tolerate provider-specific score naming
        # (e.g. "cosine_similarity", "correctness", "correctness_score").
        for score in trace.get("scores") or []:
            name = (score.get("name") or "").lower()
            value = score.get("value")
            if not isinstance(value, (int, float)):
                continue
            if "cosine" in name:
                cosine_vals.append(float(value))
            elif "correctness" in name:
                correctness_vals.append(float(value))
        slot = buckets.setdefault(category, {"cosine": [], "correctness": [], "count": 0})  # type: ignore[arg-type]
        slot["cosine"].extend(cosine_vals)
        slot["correctness"].extend(correctness_vals)
        slot["count"] = int(slot.get("count", 0)) + 1  # type: ignore[arg-type]

    def _mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    metrics: list[CategoryMetrics] = []
    for category, slot in buckets.items():
        metrics.append(
            CategoryMetrics(
                category=category,
                total_evals=int(slot["count"]),  # type: ignore[arg-type]
                avg_cosine=_mean(slot["cosine"]),  # type: ignore[arg-type]
                avg_correctness=_mean(slot["correctness"]),  # type: ignore[arg-type]
            )
        )

    metrics.sort(key=lambda m: (m["category"] == "Other", m["category"]))
    return metrics


def _attach_category_metrics(score: dict[str, Any] | None) -> dict[str, Any] | None:
    """Idempotently add `category_metrics` to a score dict in place.

    No-op when `score` is None or has no `traces`. Safe to call multiple
    times — recomputes from the current `traces` so cached scores that
    predate this feature get backfilled on read.
    """
    if not score or not isinstance(score, dict):
        return score
    traces = score.get("traces")
    if not isinstance(traces, list):
        return score
    score["category_metrics"] = _compute_category_metrics(traces)
    return score


def start_evaluation(
    session: Session,
    dataset_id: int,
    experiment_name: str,
    config_id: UUID,
    config_version: int,
    organization_id: int,
    project_id: int,
) -> EvaluationRun:
    """
    Start an evaluation run.

    Steps:
    1. Validate dataset exists and has Langfuse ID
    2. Resolve config from stored config management
    3. Create evaluation run record
    4. Start batch processing

    Args:
        session: Database session
        dataset_id: ID of the evaluation dataset
        experiment_name: Name for this evaluation experiment/run
        config_id: UUID of the stored config
        config_version: Version number of the config
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        EvaluationRun instance

    Raises:
        HTTPException: If dataset not found, config invalid, or evaluation fails to start
    """
    logger.info(
        f"[start_evaluation] Starting evaluation | experiment_name={experiment_name} | "
        f"dataset_id={dataset_id} | "
        f"org_id={organization_id} | "
        f"config_id={config_id} | "
        f"config_version={config_version}"
    )

    # Step 1: Fetch dataset from database
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

    # Step 2: Resolve config from stored config management
    config, error = resolve_evaluation_config(
        session=session,
        config_id=config_id,
        config_version=config_version,
        project_id=project_id,
    )
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to resolve config from stored config: {error}",
        )
    elif config.completion.provider != LLMProvider.OPENAI:
        raise HTTPException(
            status_code=422,
            detail="Only 'openai' provider is supported for evaluation configs",
        )

    logger.info(
        "[start_evaluation] Successfully resolved config from config management"
    )

    # Get API clients
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

    # Step 3: Create EvaluationRun record with config references
    eval_run = create_evaluation_run(
        session=session,
        run_name=experiment_name,
        dataset_name=dataset.name,
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        organization_id=organization_id,
        project_id=project_id,
    )

    # Step 4: Start the batch evaluation
    try:
        # Convert params dict to appropriate model instance based on type
        param_models = {
            "text": TextLLMParams,
            "stt": STTLLMParams,
            "tts": TTSLLMParams,
        }
        model_class = param_models[config.completion.type]
        validated_params = model_class.model_validate(config.completion.params)

        eval_run = start_evaluation_batch(
            langfuse=langfuse,
            openai_client=openai_client,
            session=session,
            eval_run=eval_run,
            config=validated_params,
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

    # Caching strategy: On the first request, trace scores are fetched from Langfuse
    # and a copy is stored in S3. Subsequent requests serve from
    # S3 instead of Langfuse, which is significantly faster. Use resync_score=true
    # to bypass the cache and re-fetch from Langfuse and store to S3 again.
    has_cached_traces_s3 = eval_run.score_trace_url is not None
    has_cached_traces_db = eval_run.score is not None and "traces" in eval_run.score
    if not resync_score:
        if has_cached_traces_s3:
            try:
                storage = get_cloud_storage(session=session, project_id=project_id)
                traces = load_json_from_object_store(
                    storage=storage, url=eval_run.score_trace_url
                )
                if traces is not None:
                    eval_run.score = _attach_category_metrics(
                        {
                            "summary_scores": (eval_run.score or {}).get(
                                "summary_scores", []
                            ),
                            "traces": traces,
                        }
                    )
                    logger.info(
                        f"[get_evaluation_with_scores] Loaded traces from S3 | "
                        f"evaluation_id={evaluation_id} | "
                        f"traces_count={len(traces)}"
                    )
                    return eval_run, None
            except Exception as e:
                logger.warning(
                    f"[get_evaluation_with_scores] Error loading traces from S3: {e} | "
                    f"evaluation_id={evaluation_id}",
                    exc_info=True,
                )

        if has_cached_traces_db:
            # Backfill category_metrics on read for runs whose cached score
            # was written before this feature shipped.
            eval_run.score = _attach_category_metrics(eval_run.score)
            logger.info(
                f"[get_evaluation_with_scores] Returning traces from DB | "
                f"evaluation_id={evaluation_id}"
            )
            return eval_run, None

    # Resync requested — fetch fresh scores from Langfuse
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

    # Build final score and persist to S3/DB for future cached reads
    score = _attach_category_metrics(
        {
            "summary_scores": merged_summary_scores,
            "traces": langfuse_score.get("traces", []),
        }
    )

    eval_run = save_score(
        eval_run_id=eval_run_id,
        organization_id=organization_id,
        project_id=project_id,
        score=score,
    )

    if eval_run:
        eval_run.score = score

    return eval_run, None
