import logging

from langfuse import Langfuse
from sqlmodel import Session, select

from app.core.util import now
from app.crud.evaluations.langfuse import fetch_trace_scores_from_langfuse
from app.crud.evaluations.score import EvaluationScore
from app.models import EvaluationRun

logger = logging.getLogger(__name__)


def create_evaluation_run(
    session: Session,
    run_name: str,
    dataset_name: str,
    dataset_id: int,
    config: dict,
    organization_id: int,
    project_id: int,
) -> EvaluationRun:
    """
    Create a new evaluation run record in the database.

    Args:
        session: Database session
        run_name: Name of the evaluation run/experiment
        dataset_name: Name of the dataset being used
        dataset_id: ID of the dataset
        config: Configuration dict for the evaluation
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        The created EvaluationRun instance
    """
    eval_run = EvaluationRun(
        run_name=run_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        config=config,
        status="pending",
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

    logger.info(f"Created EvaluationRun record: id={eval_run.id}, run_name={run_name}")

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


def update_evaluation_run(
    session: Session,
    eval_run: EvaluationRun,
    status: str | None = None,
    error_message: str | None = None,
    object_store_url: str | None = None,
    score: dict | None = None,
    embedding_batch_job_id: int | None = None,
) -> EvaluationRun:
    """
    Update an evaluation run with new values and persist to database.

    This helper function ensures consistency when updating evaluation runs
    by always updating the timestamp and properly committing changes.

    Args:
        session: Database session
        eval_run: EvaluationRun instance to update
        status: New status value (optional)
        error_message: New error message (optional)
        object_store_url: New object store URL (optional)
        score: New score dict (optional)
        embedding_batch_job_id: New embedding batch job ID (optional)

    Returns:
        Updated and refreshed EvaluationRun instance
    """
    # Update provided fields
    if status is not None:
        eval_run.status = status
    if error_message is not None:
        eval_run.error_message = error_message
    if object_store_url is not None:
        eval_run.object_store_url = object_store_url
    if score is not None:
        eval_run.score = score
    if embedding_batch_job_id is not None:
        eval_run.embedding_batch_job_id = embedding_batch_job_id

    # Always update timestamp
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

    return eval_run


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
    update_evaluation_run(session=session, eval_run=eval_run, score=score)

    total_traces = len(score.get("traces", []))
    logger.info(
        f"[get_or_fetch_score] Updated score | "
        f"evaluation_id={eval_run.id} | total_traces={total_traces}"
    )

    return score


def save_score(
    eval_run_id: int,
    organization_id: int,
    project_id: int,
    score: EvaluationScore,
) -> EvaluationRun | None:
    """
    Save score to evaluation run with its own session.

    This function creates its own database session to persist the score,
    allowing it to be called after releasing the request's main session.

    Args:
        eval_run_id: ID of the evaluation run to update
        organization_id: Organization ID for access control
        project_id: Project ID for access control
        score: Score data to save

    Returns:
        Updated EvaluationRun instance, or None if not found
    """
    from app.core.db import engine

    with Session(engine) as session:
        eval_run = get_evaluation_run_by_id(
            session=session,
            evaluation_id=eval_run_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if eval_run:
            update_evaluation_run(session=session, eval_run=eval_run, score=score)
            logger.info(
                f"[save_score] Saved score | evaluation_id={eval_run_id} | "
                f"traces={len(score.get('traces', []))}"
            )
        return eval_run
