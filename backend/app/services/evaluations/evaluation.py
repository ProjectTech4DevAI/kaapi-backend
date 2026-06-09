"""Evaluation run orchestration service."""

import logging
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

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
    start_evaluation_batch,
)
from app.crud.evaluations.score import TraceData
from app.models.evaluation import EvaluationRun, RunModeEnum
from app.models.llm.request import STTLLMParams, TextLLMParams, TTSLLMParams
from app.services.llm.providers import LLMProvider
from app.utils import get_langfuse_client, get_openai_client

logger = logging.getLogger(__name__)

# Error code surfaced in HTTPException.detail when a run_name collides with the
# (organization_id, project_id, run_name) unique constraint. Shared by the batch
# and fast paths so both return an identical 409 contract.
ERR_RUN_NAME_ALREADY_EXISTS = "run_name_already_exists"

# Name of the (organization_id, project_id, run_name) unique constraint on
# evaluation_run. Used to distinguish a run_name collision (-> 409) from any
# other IntegrityError (e.g. FK violations), which must not be masked.
_RUN_NAME_UNIQUE_CONSTRAINT = "uq_evaluation_run_org_project_run_name"


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
    run_mode: RunModeEnum | str = RunModeEnum.BATCH,
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
    eval_run = create_evaluation_run_or_409(
        session=session,
        run_name=experiment_name,
        dataset_name=dataset.name,
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        organization_id=organization_id,
        project_id=project_id,
        log_context="start_evaluation",
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
        eval_run.score = {
            "summary_scores": (eval_run.score or {}).get("summary_scores", []),
            "traces": cached_traces,
        }
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
    langfuse = get_langfuse_client(
        session=session,
        org_id=organization_id,
        project_id=project_id,
    )

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
    merged_score, merge_stats = merge_scores_step_forward(
        existing_score=existing_score,
        fresh_score=langfuse_score,
    )

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
