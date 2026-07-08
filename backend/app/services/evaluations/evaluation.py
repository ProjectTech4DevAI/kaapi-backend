"""Evaluation run orchestration service."""

import logging
from typing import Any
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.celery.utils import start_evaluation_batch_submission
from app.core.cloud.storage import get_cloud_storage
from app.core.storage_utils import load_json_from_object_store
from app.crud.evaluations import (
    EvaluationScore,
    create_evaluation_run,
    fetch_trace_scores_from_langfuse,
    get_dataset_by_id,
    get_evaluation_run_by_id,
    merge_scores_step_forward,
    resolve_evaluation_config,
    save_score,
    sort_traces_by_question_id,
)
from app.crud.evaluations.core import update_evaluation_run
from app.crud.evaluations.merge import apply_cosine_breakdown
from app.crud.evaluations.score import CategoryMetrics, TraceData
from app.models.evaluation import EvaluationRun, EvaluationRunUpdate, RunModeEnum
from app.models.llm.constants import CompletionType
from app.services.llm.providers import LLMProvider
from app.utils import get_tracing_client

logger = logging.getLogger(__name__)

# Error code surfaced in HTTPException.detail when a run_name collides with the
# (organization_id, project_id, run_name) unique constraint. Shared by the batch
# and fast paths so both return an identical 409 contract.
ERR_RUN_NAME_ALREADY_EXISTS = "run_name_already_exists"

# Name of the (organization_id, project_id, run_name) unique constraint on
# evaluation_run. Used to distinguish a run_name collision (-> 409) from any
# other IntegrityError (e.g. FK violations), which must not be masked.
_RUN_NAME_UNIQUE_CONSTRAINT = "uq_evaluation_run_org_project_run_name"

# Providers whose configs can run through the async batch evaluation path.
# Mirrors app.services.assessment.service._SUPPORTED_BATCH_PROVIDERS — the
# native variants forward raw params but still submit via the same batch API.
_SUPPORTED_BATCH_PROVIDERS = {
    LLMProvider.OPENAI,
    LLMProvider.OPENAI_NATIVE,
    LLMProvider.GOOGLE_AISTUDIO,
    LLMProvider.GOOGLE_AISTUDIO_NATIVE,
}


def _is_run_name_conflict(error: IntegrityError) -> bool:
    """True only when the IntegrityError is the run_name uniqueness violation."""
    constraint_name = getattr(
        getattr(error.orig, "diag", None), "constraint_name", None
    )
    if constraint_name:
        return constraint_name == _RUN_NAME_UNIQUE_CONSTRAINT
    # Fall back to matching the constraint name in the raw error text for
    # drivers/dialects that don't expose structured diagnostics.
    return _RUN_NAME_UNIQUE_CONSTRAINT in str(error.orig or error)


def create_evaluation_run_or_409(
    *,
    session: Session,
    run_name: str,
    dataset_name: str,
    dataset_id: int,
    config_id: UUID,
    config_version: int,
    organization_id: int,
    project_id: int,
    run_mode: RunModeEnum = RunModeEnum.BATCH,
    log_context: str,
) -> EvaluationRun:
    """Create an EvaluationRun, translating a duplicate-run_name collision into 409.

    The (organization_id, project_id, run_name) unique constraint guards against
    double-click / client-retry races; on collision we roll back and raise a 409
    instead of leaking the IntegrityError.
    """
    try:
        return create_evaluation_run(
            session=session,
            run_name=run_name,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            config_id=config_id,
            config_version=config_version,
            organization_id=organization_id,
            project_id=project_id,
            run_mode=run_mode,
        )
    except IntegrityError as exc:
        session.rollback()
        # Only a run_name collision becomes a 409; any other IntegrityError
        # (e.g. a dataset/config FK violation) is a real failure and must not be
        # masked as a duplicate-name error.
        if not _is_run_name_conflict(exc):
            logger.error(
                f"[{log_context}] IntegrityError creating run | run_name={run_name} | "
                f"org_id={organization_id} | project_id={project_id} | error={exc}",
                exc_info=True,
            )
            raise
        logger.warning(
            f"[{log_context}] Duplicate run_name | run_name={run_name} | "
            f"org_id={organization_id} | project_id={project_id}"
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"{ERR_RUN_NAME_ALREADY_EXISTS}: a run with name '{run_name}' "
                "already exists for this organization and project. Pick a new "
                "run_name or fetch the existing run via GET /evaluations."
            ),
        )


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

    Datasets uploaded without a `category` CSV column have no `category` key on
    any trace — in that case we return `[]` so the response omits the category
    dimension entirely rather than reporting a synthetic "Other" bucket.
    """
    if not any("category" in trace for trace in traces):
        return []

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


def validate_and_start_batch_evaluation(
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
    3. Check the config provider is supported for batch evaluation
    4. Check the config type is 'text'
    5. Create evaluation run record
    6. Start batch processing

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
        f"[validate_and_start_batch_evaluation] Starting evaluation | experiment_name={experiment_name} | "
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
        f"[validate_and_start_batch_evaluation] Found dataset | id={dataset.id} | name={dataset.name} | "
        f"object_store_url={'present' if dataset.object_store_url else 'None'} | "
        f"langfuse_id={dataset.langfuse_dataset_id}"
    )

    if not dataset.langfuse_dataset_id and not dataset.object_store_url:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} has no Langfuse nor object-store "
            "backing; cannot run evaluation.",
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
    elif config.completion.provider not in _SUPPORTED_BATCH_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Provider '{config.completion.provider}' is not supported for "
                f"evaluation configs. Supported providers: "
                f"{sorted(_SUPPORTED_BATCH_PROVIDERS)}"
            ),
        )

    if config.completion.type != CompletionType.TEXT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Config type '{config.completion.type}' is not supported for "
                f"evaluation configs. Only '{CompletionType.TEXT.value}' type is supported."
            ),
        )

    logger.info(
        "[validate_and_start_batch_evaluation] Successfully resolved config from config management"
    )

    # Step 3: Create EvaluationRun record with config references
    eval_run = create_evaluation_run_or_409(
        session=session,
        run_name=experiment_name,
        dataset_name=dataset.name,
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        organization_id=organization_id,
        project_id=project_id,
        log_context="validate_and_start_batch_evaluation",
    )

    # Step 4: Queue the batch submission for asynchronous processing
    trace_id = correlation_id.get() or "N/A"
    try:
        celery_task_id = start_evaluation_batch_submission(
            project_id=project_id,
            job_id=str(eval_run.id),
            trace_id=trace_id,
            organization_id=organization_id,
            config_id=str(config_id),
            config_version=config_version,
        )
        logger.info(
            f"[validate_and_start_batch_evaluation] Batch submission queued | "
            f"run_id={eval_run.id} | celery_task_id={celery_task_id}"
        )
        return eval_run
    except Exception as e:
        logger.error(
            f"[validate_and_start_batch_evaluation] Failed to queue batch submission | run_id={eval_run.id} | {e}",
            exc_info=True,
        )
        eval_run = update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(
                status="failed",
                error_message=f"Failed to queue batch submission: {e}",
            ),
        )
        return eval_run


def _load_cached_traces(
    session: Session,
    project_id: int,
    eval_run: EvaluationRun,
) -> tuple[list[TraceData], bool]:
    """
    Load previously cached traces for an evaluation run.

    Traces are cached in S3 (pointed to by ``score_trace_url``) with a DB fallback
    (``score["traces"]``). This returns whatever is cached so a resync can merge
    against it instead of overwriting it.

    Returns:
        Tuple of (traces, load_failed). ``load_failed`` is True only when a cache
        pointer exists but could not be read — the caller must then avoid
        overwriting the cache, otherwise it would lose data. A genuinely empty
        cache (first sync) returns ([], False).
    """
    if eval_run.score_trace_url:
        try:
            storage = get_cloud_storage(session=session, project_id=project_id)
            traces = load_json_from_object_store(
                storage=storage, url=eval_run.score_trace_url
            )
            if traces is not None:
                return traces, False
            logger.warning(
                f"[_load_cached_traces] Cached traces URL returned no data | "
                f"evaluation_id={eval_run.id} | url={eval_run.score_trace_url}"
            )
            return [], True
        except Exception as e:
            logger.warning(
                f"[_load_cached_traces] Error loading traces from S3: {e} | "
                f"evaluation_id={eval_run.id}",
                exc_info=True,
            )
            return [], True

    if eval_run.score is not None and "traces" in eval_run.score:
        return eval_run.score.get("traces", []) or [], False

    return [], False


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

    # If not requesting trace info, return existing score (with summary_scores)
    if not get_trace_info:
        return eval_run, None

    # Caching strategy: trace scores are fetched from Langfuse once, then cached
    # (traces in S3, summary in the DB). Normal reads serve from that cache, which is
    # much faster than hitting Langfuse. resync_score=true bypasses the cache.
    #
    # A resync re-fetches from Langfuse and MERGES the result with the cached traces
    # step-forward (union by trace_id), so the cached pair count can only grow. This
    # prevents a transient Langfuse fetch failure from making the count go *down*
    # (e.g. 29/30 -> 27/30): a partial sync at worst contributes nothing new, and a
    # later resync can backfill the missing traces (15/30 -> 24/30 -> 30/30).
    cached_traces, cache_load_failed = _load_cached_traces(
        session=session, project_id=project_id, eval_run=eval_run
    )

    if not resync_score and cached_traces:
        # Serve from cache, but backfill any trace missing a cosine score from
        # per_item_scores and recompute the summary, so older caches self-heal
        # without a resync. The step-forward merge (empty fresh side) preserves
        # summary-only scores and sorts traces.
        cached_score, _ = merge_scores_step_forward(
            existing_score={
                "summary_scores": (eval_run.score or {}).get("summary_scores", []),
                "traces": cached_traces,
            },
            fresh_score={"summary_scores": [], "traces": []},
            per_item_scores=eval_run.per_item_scores,
        )
        apply_cosine_breakdown(
            cached_score["summary_scores"],
            total_items=eval_run.total_items,
            unscoreable=eval_run.unscoreable,
        )
        eval_run.score = _attach_category_metrics(cached_score)
        logger.info(
            f"[get_evaluation_with_scores] Served traces from cache | "
            f"evaluation_id={evaluation_id} | traces_count={len(cached_traces)}"
        )
        return eval_run, None

    # On resync, if a cache pointer exists but we could not read it, do NOT overwrite
    # it with a fresh-only fetch (that could regress the cached pair count). Surface
    # the error and leave the existing cache untouched.
    if resync_score and cache_load_failed:
        logger.warning(
            f"[get_evaluation_with_scores] Skipping resync to avoid data loss | "
            f"evaluation_id={evaluation_id} | reason=could_not_read_cached_traces"
        )
        return eval_run, (
            "Could not read existing cached traces; skipping resync to avoid "
            "losing data. Please try again."
        )

    # Fetch fresh scores from Langfuse (first sync, or resync).
    langfuse = get_tracing_client(
        session=session,
        org_id=organization_id,
        project_id=project_id,
    )

    # Opt-out: resync needs Langfuse (400); a normal read serves durable cosine.
    if langfuse is None:
        if resync_score:
            raise HTTPException(
                status_code=400,
                detail="Tracing is disabled for this project; cannot resync "
                "scores from Langfuse.",
            )
        cosine_score, _ = merge_scores_step_forward(
            existing_score={
                "summary_scores": (eval_run.score or {}).get("summary_scores", []),
                "traces": cached_traces or [],
            },
            fresh_score={"summary_scores": [], "traces": []},
            per_item_scores=eval_run.per_item_scores,
        )
        apply_cosine_breakdown(
            cosine_score["summary_scores"],
            total_items=eval_run.total_items,
            unscoreable=eval_run.unscoreable,
        )
        eval_run.score = _attach_category_metrics(cosine_score)
        return eval_run, None

    dataset_name = eval_run.dataset_name
    run_name = eval_run.run_name
    eval_run_id = eval_run.id

    try:
        langfuse_score = fetch_trace_scores_from_langfuse(
            langfuse=langfuse,
            dataset_name=dataset_name,
            run_name=run_name,
            project_id=project_id,
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

    # Step-forward merge: combine the freshly fetched score with the cached one so the
    # result never shrinks, then recompute summaries from the merged union.
    existing_score: EvaluationScore = {
        "summary_scores": (eval_run.score or {}).get("summary_scores", []),
        "traces": cached_traces,
    }
    # Pass per_item_scores so any trace Langfuse is still missing a cosine for is
    # backfilled from our source of truth, recovering computed-but-unwritten scores.
    merged_score, merge_stats = merge_scores_step_forward(
        existing_score=existing_score,
        fresh_score=langfuse_score,
        per_item_scores=eval_run.per_item_scores,
    )

    # Re-attach the run-level denominator and unscoreable breakdown, which
    # `compute_summary_scores` (trace-only) cannot know.
    apply_cosine_breakdown(
        merged_score["summary_scores"],
        total_items=eval_run.total_items,
        unscoreable=eval_run.unscoreable,
    )

    # Recompute `category_metrics` from the merged trace set so the per-category
    # rollup stays in sync with the new traces, not just the cached ones.
    # `_attach_category_metrics` mutates in place and is idempotent.
    _attach_category_metrics(merged_score)

    logger.info(
        f"[get_evaluation_with_scores] Merged traces step-forward | "
        f"evaluation_id={evaluation_id} | cached={len(cached_traces)} | "
        f"fetched={len(langfuse_score.get('traces', []))} | "
        f"merged={len(merged_score['traces'])} | reused={merge_stats['reused']} | "
        f"updated={merge_stats['updated']} | added={merge_stats['added']}"
    )

    eval_run = save_score(
        eval_run_id=eval_run_id,
        organization_id=organization_id,
        project_id=project_id,
        score=merged_score,
    )

    if eval_run:
        eval_run.score = merged_score

    return eval_run, None
