import logging
from typing import Any
from uuid import UUID

from langfuse import Langfuse
from sqlmodel import Session, select

from app.core.cloud.storage import get_cloud_storage
from app.core.db import engine
from app.core.storage_utils import upload_jsonl_to_object_store
from app.core.util import now
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations.langfuse import fetch_trace_scores_from_langfuse
from app.crud.evaluations.score import DEFAULT_CATEGORY, EvaluationScore
from app.models import EvaluationRun, EvaluationRunUpdate
from app.models.evaluation import RunModeEnum
from app.models.config.config import ConfigTag
from app.models.llm.request import ConfigBlob, LLMCallConfig
from app.models.stt_evaluation import EvaluationType
from app.services.llm.jobs import resolve_config_blob

logger = logging.getLogger(__name__)


def resolve_evaluation_config(
    session: Session,
    config_id: UUID,
    config_version: int,
    project_id: int,
    tag: ConfigTag = ConfigTag.DEFAULT,
) -> tuple[ConfigBlob | None, str | None]:
    """
    Resolve config blob from stored config management.

    Args:
        session: Database session
        config_id: UUID of the stored config
        config_version: Version number of the config
        project_id: Project ID for access control

    Returns:
        Tuple of (ConfigBlob or None, error_message or None)
    """
    config_version_crud = ConfigVersionCrud(
        session=session,
        config_id=config_id,
        project_id=project_id,
        tag=tag,
    )

    return resolve_config_blob(
        config_crud=config_version_crud,
        config=LLMCallConfig(id=config_id, version=config_version),
    )


def create_evaluation_run(
    session: Session,
    run_name: str,
    dataset_name: str,
    dataset_id: int,
    config_id: UUID,
    config_version: int,
    organization_id: int,
    project_id: int,
    run_mode: RunModeEnum = RunModeEnum.BATCH,
) -> EvaluationRun:
    """
    Create a new evaluation run record in the database.

    Args:
        session: Database session
        run_name: Name of the evaluation run/experiment
        dataset_name: Name of the dataset being used
        dataset_id: ID of the dataset
        config_id: UUID of the stored config
        config_version: Version number of the config
        organization_id: Organization ID
        project_id: Project ID
        run_mode: Execution mode (RunModeEnum.BATCH default, or RunModeEnum.FAST)

    Returns:
        The created EvaluationRun instance
    """
    eval_run = EvaluationRun(
        run_name=run_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        type=EvaluationType.TEXT.value,
        config_id=config_id,
        config_version=config_version,
        status="pending",
        run_mode=run_mode,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    session.add(eval_run)
    try:
        session.commit()
        session.refresh(eval_run)
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to create EvaluationRun: {e}", exc_info=True)
        raise

    logger.info(
        f"[create_evaluation_run] Created EvaluationRun record: id={eval_run.id}, "
        f"run_name={run_name}, config_id={config_id}, config_version={config_version}"
    )
    return eval_run


def list_evaluation_runs(
    session: Session,
    organization_id: int,
    project_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[EvaluationRun]:
    """
    List all evaluation runs for an organization and project.

    Args:
        session: Database session
        organization_id: Organization ID to filter by
        project_id: Project ID to filter by
        limit: Maximum number of runs to return (default 50)
        offset: Number of runs to skip (for pagination)

    Returns:
        List of EvaluationRun objects, ordered by most recent first
    """
    statement = (
        select(EvaluationRun)
        .where(EvaluationRun.organization_id == organization_id)
        .where(EvaluationRun.project_id == project_id)
        .where(EvaluationRun.type == EvaluationType.TEXT.value)
        .order_by(EvaluationRun.inserted_at.desc())
        .limit(limit)
        .offset(offset)
    )

    runs = session.exec(statement).all()

    logger.info(
        f"Found {len(runs)} evaluation runs for org_id={organization_id}, "
        f"project_id={project_id}"
    )

    return runs


def get_evaluation_run_by_id(
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
) -> EvaluationRun | None:
    """
    Get a specific evaluation run by ID.

    Args:
        session: Database session
        evaluation_id: ID of the evaluation run
        organization_id: Organization ID (for access control)
        project_id: Project ID (for access control)

    Returns:
        EvaluationRun if found and accessible, None otherwise
    """
    statement = (
        select(EvaluationRun)
        .where(EvaluationRun.id == evaluation_id)
        .where(EvaluationRun.organization_id == organization_id)
        .where(EvaluationRun.project_id == project_id)
        .where(EvaluationRun.type == EvaluationType.TEXT.value)
    )

    eval_run = session.exec(statement).first()

    if eval_run:
        logger.info(
            f"Found evaluation run {evaluation_id}: status={eval_run.status}, "
            f"batch_job_id={eval_run.batch_job_id}"
        )
    else:
        logger.warning(
            f"Evaluation run {evaluation_id} not found or not accessible "
            f"for org_id={organization_id}, project_id={project_id}"
        )

    return eval_run


TERMINAL_EVAL_STATUSES = {"completed", "failed"}


def update_evaluation_run(
    session: Session,
    eval_run: EvaluationRun,
    update: EvaluationRunUpdate,
) -> EvaluationRun:
    """
    Apply a partial update to an evaluation run and persist it.

    Only fields explicitly set on `update` are applied (`exclude_unset=True`
    semantics), so callers don't accidentally clear unrelated columns.
    `updated_at` is always bumped.

    When the update transitions the run into a terminal status (`completed`
    or `failed`) for the first time, enqueues a Celery task to send
    completion notifications. The downstream task re-checks the
    `notification` table for existing rows so a duplicate enqueue is a
    no-op rather than a duplicate send.
    """
    update_fields = update.model_dump(exclude_unset=True)
    incoming_status = update_fields.get("status")
    previous_status = eval_run.status
    should_notify = (
        incoming_status in TERMINAL_EVAL_STATUSES
        and previous_status not in TERMINAL_EVAL_STATUSES
    )

    for field_name, new_value in update_fields.items():
        setattr(eval_run, field_name, new_value)

    eval_run.updated_at = now()

    # Persist to database
    session.add(eval_run)
    try:
        session.commit()
        session.refresh(eval_run)
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to update EvaluationRun: {e}", exc_info=True)
        raise

    if should_notify:
        _enqueue_eval_completion_notification(eval_run)

    return eval_run


def _enqueue_eval_completion_notification(eval_run: EvaluationRun) -> None:
    """Enqueue the completion email task, swallowing broker errors.

    Imported lazily to avoid a circular import: the celery task module
    pulls in CRUD helpers via `app.crud.user_project`, which would loop
    back through `app.crud` package init.
    """
    try:
        from app.celery.tasks.notifications import send_eval_completion_notification

        send_eval_completion_notification.delay(eval_run.id)
        logger.info(
            f"[update_evaluation_run] Enqueued completion notification | "
            f"evaluation_id={eval_run.id} | status={eval_run.status}"
        )
    except Exception as e:
        logger.error(
            f"[update_evaluation_run] Failed to enqueue completion notification | "
            f"evaluation_id={eval_run.id} | error={e}",
            exc_info=True,
        )


def get_or_fetch_score(
    session: Session,
    eval_run: EvaluationRun,
    langfuse: Langfuse,
    force_refetch: bool = False,
) -> EvaluationScore:
    """
    Get cached score with trace info or fetch from Langfuse and update.

    This function implements a cache-on-first-request pattern:
    - If score already has 'traces' key, return it
    - Otherwise, fetch from Langfuse, merge with existing summary_scores, and return
    - If force_refetch is True, always fetch fresh data from Langfuse

    Args:
        session: Database session
        eval_run: EvaluationRun instance
        langfuse: Configured Langfuse client
        force_refetch: If True, skip cache and fetch fresh from Langfuse

    Returns:
        Score data with per-trace scores and summary statistics

    Raises:
        ValueError: If the run is not found in Langfuse
        Exception: If Langfuse API calls fail
    """
    # Check if score already exists with traces
    has_traces = eval_run.score is not None and "traces" in eval_run.score
    if not force_refetch and has_traces:
        logger.info(
            f"[get_or_fetch_score] Returning existing score | evaluation_id={eval_run.id}"
        )
        return eval_run.score

    logger.info(
        f"[get_or_fetch_score] Fetching score from Langfuse | "
        f"evaluation_id={eval_run.id} | dataset={eval_run.dataset_name} | "
        f"run={eval_run.run_name} | force_refetch={force_refetch}"
    )

    # Get existing summary_scores if any (e.g., cosine_similarity from cron job)
    existing_summary_scores = []
    if eval_run.score and "summary_scores" in eval_run.score:
        existing_summary_scores = eval_run.score.get("summary_scores", [])

    # Fetch from Langfuse
    langfuse_score = fetch_trace_scores_from_langfuse(
        langfuse=langfuse,
        dataset_name=eval_run.dataset_name,
        run_name=eval_run.run_name,
        project_id=eval_run.project_id,
    )

    # Merge summary_scores: existing scores + new scores from Langfuse
    existing_scores_map = {s["name"]: s for s in existing_summary_scores}
    for langfuse_summary in langfuse_score.get("summary_scores", []):
        existing_scores_map[langfuse_summary["name"]] = langfuse_summary

    merged_summary_scores = list(existing_scores_map.values())

    # Build final score with merged summary_scores and traces
    score: EvaluationScore = {
        "summary_scores": merged_summary_scores,
        "traces": langfuse_score.get("traces", []),
    }

    # Update score column using existing helper
    update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(score=score),
    )

    total_traces = len(score.get("traces", []))
    logger.info(
        f"[get_or_fetch_score] Updated score | "
        f"evaluation_id={eval_run.id} | total_traces={total_traces}"
    )

    return score


def _upload_score_traces(
    *,
    session: Session,
    eval_run_id: int,
    project_id: int,
    traces: list[dict[str, Any]],
) -> str | None:
    """Upload per-trace records to S3 for an evaluation run.

    Shared by ``persist_score_traces`` (Q&A skeleton at the response stage) and
    ``save_score`` (full scored unit at completion). Returns the object-store
    URL on success, ``""`` when there are no traces to upload, or ``None`` when
    an upload was attempted but failed — so the caller can fall back to DB
    storage or skip persistence. Never raises.
    """
    if not traces:
        return ""
    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        score_trace_url = upload_jsonl_to_object_store(
            storage=storage,
            results=traces,
            filename=f"traces_{eval_run_id}.json",
            subdirectory=f"evaluations/score/{eval_run_id}",
            format="json",
        )
        if score_trace_url:
            logger.info(
                f"[_upload_score_traces] uploaded traces to S3 | "
                f"evaluation_id={eval_run_id} | url={score_trace_url} | "
                f"traces_count={len(traces)}"
            )
        else:
            logger.warning(
                f"[_upload_score_traces] failed to upload traces to S3 | "
                f"evaluation_id={eval_run_id}"
            )
        return score_trace_url
    except Exception as e:
        logger.error(
            f"[_upload_score_traces] Error uploading traces to S3: {e} | "
            f"evaluation_id={eval_run_id}",
            exc_info=True,
        )
        return None


def persist_score_traces(
    eval_run_id: int,
    organization_id: int,
    project_id: int,
    traces: list[dict[str, Any]],
) -> EvaluationRun | None:
    """Persist per-trace records to S3 and record the pointer in
    ``score_trace_url``, WITHOUT touching the ``score`` column.

    Used at the response stage to durably store the Q&A skeleton before cosine
    is computed: the embedding stage runs in a later poll cycle and reloads the
    skeleton from ``score_trace_url`` to build the full trace unit. Leaving
    ``score`` untouched keeps the run score-less (UI shows "processing", not
    "No scores available") until it is marked completed. If the S3 upload fails
    the pointer is left empty; cosine stays recoverable from the durable
    ``per_item_scores`` on read.

    Creates its own DB session so it can run after the request's main session is
    released.

    Returns:
        Updated EvaluationRun instance, or None if not found.
    """
    with Session(engine) as session:
        eval_run = get_evaluation_run_by_id(
            session=session,
            evaluation_id=eval_run_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if not eval_run:
            return None

        score_trace_url = _upload_score_traces(
            session=session,
            eval_run_id=eval_run_id,
            project_id=project_id,
            traces=traces,
        )
        update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(score_trace_url=score_trace_url or None),
        )

        logger.info(
            f"[persist_score_traces] Persisted trace pointer | "
            f"evaluation_id={eval_run_id} | traces={len(traces)}"
        )

        return eval_run


def save_score(
    eval_run_id: int,
    organization_id: int,
    project_id: int,
    score: EvaluationScore,
) -> EvaluationRun | None:
    """
    Save score to evaluation run with its own session.

    Persists the per-trace records to S3 (via ``_upload_score_traces``) and
    writes the ``score`` column: just the summary when traces live in S3, or the
    full unit as a DB fallback when the S3 upload fails. Creates its own database
    session, allowing it to be called after releasing the request's main session.

    To persist only the trace skeleton without writing ``score`` (the response
    stage, before cosine is computed), use ``persist_score_traces`` instead.

    Args:
        eval_run_id: ID of the evaluation run to update
        organization_id: Organization ID for access control
        project_id: Project ID for access control
        score: Score data to save

    Returns:
        Updated EvaluationRun instance, or None if not found
    """

    with Session(engine) as session:
        eval_run = get_evaluation_run_by_id(
            session=session,
            evaluation_id=eval_run_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if not eval_run:
            return None

        traces = score.get("traces", [])
        summary_score = score.get("summary_scores", [])

        score_trace_url = _upload_score_traces(
            session=session,
            eval_run_id=eval_run_id,
            project_id=project_id,
            traces=traces,
        )

        # IF TRACES DATA IS STORED IN S3 URL THEN HERE WE ARE JUST STORING THE SUMMARY SCORE
        # TODO: Evaluate whether this behaviour is needed or completely discard the storing data in db
        if score_trace_url:
            db_score = {"summary_scores": summary_score}
        else:
            # fallback to store data in db if failed to store in s3
            db_score = score

        update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(
                score=db_score,
                score_trace_url=score_trace_url or None,
            ),
        )

        logger.info(
            f"[save_score] Saved score | evaluation_id={eval_run_id} | "
            f"traces={len(score.get('traces', []))}"
        )

        return eval_run


def group_traces_by_question_id(
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Group evaluation traces by question_id for horizontal comparison.

    Returns:
        List of grouped traces sorted by question_id:
        [
            {
                "question_id": 1,
                "question": "What is Python?",
                "ground_truth_answer": "...",
                "category": "health",
                "llm_answers": ["Answer 1", "Answer 2"],
                "trace_ids": ["trace-1", "trace-2"],
                "scores": [[...], [...]]
            }
        ]
    """

    # whether question_id exists in the traces
    if traces and (
        traces[0].get("question_id") is None or traces[0].get("question_id") == ""
    ):
        raise ValueError("Grouped export format is not available for this evaluation.")

    groups: dict[int, list[dict[str, Any]]] = {}

    for trace in traces:
        question_id = trace.get("question_id")
        if question_id not in groups:
            groups[question_id] = []
        groups[question_id].append(trace)

    result: list[dict[str, Any]] = []
    for question_id in sorted(groups.keys()):
        group_traces = groups[question_id]
        first = group_traces[0]
        result.append(
            {
                "question_id": question_id,
                "question": first.get("question", ""),
                "ground_truth_answer": first.get("ground_truth_answer", ""),
                "category": first.get("category") or DEFAULT_CATEGORY,
                "llm_answers": [t.get("llm_answer", "") for t in group_traces],
                "trace_ids": [t.get("trace_id", "") for t in group_traces],
                "scores": [t.get("scores", []) for t in group_traces],
            }
        )

    logger.info(f"[group_traces_by_question_id] Created {len(result)} groups")
    return result


def resolve_model_from_config(
    session: Session,
    eval_run: EvaluationRun,
) -> str:
    """
    Resolve the model name from the evaluation run's config.

    Args:
        session: Database session
        eval_run: EvaluationRun instance

    Returns:
        Model name from config

    Raises:
        ValueError: If config is missing, invalid, or has no model
    """
    if not eval_run.config_id or not eval_run.config_version:
        raise ValueError(
            f"Evaluation run {eval_run.id} has no config reference "
            f"(config_id={eval_run.config_id}, config_version={eval_run.config_version})"
        )

    config, error = resolve_evaluation_config(
        session=session,
        config_id=eval_run.config_id,
        config_version=eval_run.config_version,
        project_id=eval_run.project_id,
    )

    if error or config is None:
        raise ValueError(
            f"Config resolution failed for evaluation {eval_run.id} "
            f"(config_id={eval_run.config_id}, version={eval_run.config_version}): {error}"
        )

    # params is a dict, not a Pydantic model, so use dict access
    model = config.completion.params.get("model")
    if not model:
        raise ValueError(
            f"Config for evaluation {eval_run.id} does not contain a 'model' parameter"
        )
    return model
