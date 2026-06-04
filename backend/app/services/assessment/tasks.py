"""Orchestrator: submit the run's current PENDING stage as a batch, then exit."""

import logging

from asgi_correlation_id import correlation_id
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session

from app.celery.tasks.job_execution import run_assessment_pipeline
from app.core.db import engine
from app.crud.assessment import (
    get_assessment_dataset_by_id,
    recompute_assessment_status,
    update_assessment_run_status,
)
from app.crud.assessment.batch import _load_dataset_rows, submit_assessment_batch
from app.crud.assessment.processing import parse_assessment_output
from app.crud.evaluations.core import resolve_evaluation_config
from app.crud.job import get_batch_job
from app.models.assessment import (
    Assessment,
    AssessmentAttachment,
    AssessmentRun,
    Stage,
    StageStatus,
)
from app.models.config.config import ConfigTag
from app.services.assessment.prefilter import resolve_prefilter_settings
from app.services.assessment.stages import (
    GATE_STAGES,
    STAGE_PARSERS,
    advance_or_finalize,
    build_pipeline,
    build_prefilter_requests,
    load_raw_batch_results,
    next_stage,
    ordered_stages,
    submit_prefilter_batch,
)

logger = logging.getLogger(__name__)

_PREFILTER_STAGES = {
    Stage.PRE_FILTER_TOPIC_RELEVANCE,
    Stage.PRE_FILTER_DUPLICATE_DETECTION,
}


def _mark_run_failed(run_id: int, error_message: str) -> None:
    """Fail a run from a fresh session so a killed task leaves no dangling run."""
    try:
        with Session(engine) as session:
            run = session.get(AssessmentRun, run_id)
            if (
                run is None
                or run.stage == Stage.COMPLETED
                or run.stage_status == StageStatus.FAILED
            ):
                return
            run.stage_status = StageStatus.FAILED
            update_assessment_run_status(
                session=session, run=run, status="failed", error_message=error_message
            )
            recompute_assessment_status(
                session=session, assessment_id=run.assessment_id
            )
            logger.info("[_mark_run_failed] run_id=%s marked failed", run_id)
    except Exception:
        logger.error(
            "[_mark_run_failed] could not mark run_id=%s failed", run_id, exc_info=True
        )


def execute_assessment_pipeline(
    run_id: int, organization_id: int, project_id: int
) -> None:
    """Guarded entrypoint: submit the run's current stage, never leave it dangling."""
    try:
        _orchestrate(run_id, organization_id, project_id)
    except SoftTimeLimitExceeded:
        logger.error("[execute_assessment_pipeline] soft time limit run_id=%s", run_id)
        _mark_run_failed(run_id, "Assessment run exceeded the time limit.")
        raise
    except Exception:
        logger.error(
            "[execute_assessment_pipeline] unexpected failure run_id=%s",
            run_id,
            exc_info=True,
        )
        _mark_run_failed(run_id, "Assessment run failed unexpectedly.")
        raise


def _dispatch(run_id: int, organization_id: int, project_id: int) -> None:
    run_assessment_pipeline.delay(
        run_id=run_id,
        organization_id=organization_id,
        project_id=project_id,
        trace_id=correlation_id.get() or "",
    )


def _resolve_run_context(
    session: Session, run: AssessmentRun, organization_id: int, project_id: int
):
    """Load the assessment, dataset, and resolved config; ``error`` set on failure."""
    assessment = session.get(Assessment, run.assessment_id)
    if assessment is None:
        return None, None, None, "Parent assessment not found."
    dataset = get_assessment_dataset_by_id(
        session=session,
        dataset_id=assessment.dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )
    config_blob, error = resolve_evaluation_config(
        session=session,
        config_id=run.config_id,
        config_version=run.config_version,
        project_id=project_id,
        tag=ConfigTag.ASSESSMENT,
    )
    if error or config_blob is None:
        return assessment, dataset, None, f"Config resolution failed: {error}"
    return assessment, dataset, config_blob, None


def _accepted_indices(
    session: Session, run: AssessmentRun, total_rows: int, project_id: int
) -> list[int]:
    """Row indices that passed every gate stage before the current one.

    Prefers the accepted set persisted by the gate stage on ``run.pipeline``
    (set in ``_record_gate_stats``), avoiding a re-download + re-parse of the
    gate batch at the memory-heavy prefilter -> assessment transition. Falls back
    to recomputing from the gate batches only if nothing was persisted.
    """
    stored = (run.pipeline or {}).get("accepted_indices")
    if stored is not None:
        return [i for i in sorted(stored) if 0 <= i < total_rows]

    accepted = set(range(total_rows))
    for stage in ordered_stages(run.pipeline):
        if stage == run.stage:
            break
        if stage not in GATE_STAGES:
            continue
        batch_id = (run.stage_batches or {}).get(stage)
        if batch_id is None:
            continue
        batch_job = get_batch_job(session=session, batch_job_id=batch_id)
        if not batch_job:
            continue
        raw = load_raw_batch_results(session, batch_job, project_id)
        outputs = parse_assessment_output(raw, batch_job.provider)
        parsed = STAGE_PARSERS[stage](outputs)
        accepted &= {idx for idx, r in parsed.items() if r.get("verdict")}
    return sorted(accepted)


def _orchestrate(run_id: int, organization_id: int, project_id: int) -> None:
    with Session(engine) as session:
        run = session.get(AssessmentRun, run_id)
        if run is None:
            logger.error("[execute_assessment_pipeline] run_id=%s not found", run_id)
            return
        if run.stage == Stage.COMPLETED or run.stage_status == StageStatus.FAILED:
            return

        if not run.pipeline:
            run.pipeline = build_pipeline(run.input or {})
            flag_modified(run, "pipeline")
        if run.stage is None:
            run.stage = next_stage(run.pipeline)
            run.stage_status = StageStatus.PENDING
            run.status = "processing"
        if run.stage_status != StageStatus.PENDING:
            session.add(run)
            session.commit()
            return
        session.add(run)
        session.commit()
        session.refresh(run)

        _submit_stage(session, run, organization_id, project_id)


def _submit_stage(
    session: Session, run: AssessmentRun, organization_id: int, project_id: int
) -> None:
    assessment, dataset, config_blob, error = _resolve_run_context(
        session, run, organization_id, project_id
    )
    if error:
        run.stage_status = StageStatus.FAILED
        update_assessment_run_status(
            session=session, run=run, status="failed", error_message=error
        )
        recompute_assessment_status(session=session, assessment_id=run.assessment_id)
        return

    all_rows = _load_dataset_rows(session, dataset)
    if not all_rows:
        run.stage_status = StageStatus.FAILED
        update_assessment_run_status(
            session=session,
            run=run,
            status="failed",
            error_message="Dataset has no rows.",
        )
        recompute_assessment_status(session=session, assessment_id=run.assessment_id)
        return

    accepted = _accepted_indices(session, run, len(all_rows), project_id)
    rows_with_idx = [(i, all_rows[i]) for i in accepted]
    stage = run.stage

    if not rows_with_idx:
        # Nothing left for this stage (all rows rejected upstream) — advance.
        _persist_advance(session, run, organization_id, project_id)
        return

    if stage in _PREFILTER_STAGES:
        cfg = resolve_prefilter_settings(run.input.get("prefilter_config") or {})
        attachments = [
            AssessmentAttachment(**a) for a in (run.input.get("attachments") or [])
        ]
        selected = cfg.get("tr_attachment_columns")
        if selected is not None:
            attachments = [a for a in attachments if a.column in set(selected)]
        jsonl = build_prefilter_requests(stage, rows_with_idx, cfg, attachments)
        batch_job = submit_prefilter_batch(
            session=session,
            organization_id=organization_id,
            project_id=project_id,
            jsonl_data=jsonl,
            display_name=f"assessment-{run.id}-{stage}",
        )
    elif stage == Stage.L2_ASSESSMENT:
        batch_job = submit_assessment_batch(
            session=session,
            run=run,
            assessment=assessment,
            dataset=dataset,
            config_blob=config_blob,
            assessment_input=run.input or {},
            organization_id=organization_id,
            project_id=project_id,
            preloaded_rows=[r for _, r in rows_with_idx],
            row_indices=[i for i, _ in rows_with_idx],
        )
        run.total_items = batch_job.total_items
    else:
        raise ValueError(f"Unknown stage: {stage}")

    stage_batches = dict(run.stage_batches or {})
    stage_batches[stage] = batch_job.id
    run.stage_batches = stage_batches
    flag_modified(run, "stage_batches")
    run.stage_status = StageStatus.PROCESSING
    run.status = "processing"
    session.add(run)
    session.commit()
    recompute_assessment_status(session=session, assessment_id=run.assessment_id)

    logger.info(
        "[execute_assessment_pipeline] run_id=%s | stage=%s submitted | batch=%s | rows=%s",
        run.id,
        stage,
        batch_job.id,
        len(rows_with_idx),
    )


def _persist_advance(
    session: Session, run: AssessmentRun, organization_id: int, project_id: int
) -> None:
    nxt = advance_or_finalize(run)
    session.add(run)
    session.commit()
    recompute_assessment_status(session=session, assessment_id=run.assessment_id)
    if not nxt:
        return
    # Commit precedes dispatch (the worker only acts on a committed PENDING run).
    # If the broker call fails the run would otherwise sit at PENDING forever — the
    # cron only re-polls PROCESSING runs — so mark it failed (resumable) instead.
    try:
        _dispatch(run.id, organization_id, project_id)
    except Exception:
        logger.error(
            "[_persist_advance] run_id=%s stage=%s enqueue failed — marking failed for resume",
            run.id,
            run.stage,
            exc_info=True,
        )
        run.stage_status = StageStatus.FAILED
        update_assessment_run_status(
            session=session,
            run=run,
            status="failed",
            error_message="Failed to enqueue the next pipeline stage. Resume the run to retry.",
        )
        recompute_assessment_status(session=session, assessment_id=run.assessment_id)
