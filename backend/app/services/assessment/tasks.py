"""Celery task logic for running a single assessment run (prefilter → L2 batch submit)."""

import logging

from sqlmodel import Session

from app.core.db import engine
from app.crud.assessment import (
    get_assessment_dataset_by_id,
    recompute_assessment_status,
    update_assessment_run_status,
)
from app.crud.assessment.batch import _load_dataset_rows, submit_assessment_batch
from app.crud.config import ConfigCrud
from app.crud.evaluations.core import resolve_evaluation_config
from app.models.assessment import (
    Assessment,
    AssessmentAttachment,
    AssessmentRun,
)
from app.models.config.config import ConfigTag
from app.services.assessment.prefilter import run_prefilter_pipeline

logger = logging.getLogger(__name__)


def execute_assessment_run(
    run_id: int,
    organization_id: int,
    project_id: int,
) -> None:
    """Run prefilter filtering then submit L2 batch for one AssessmentRun.

    Status transitions:
      pending → prefilter_processing → prefilter_failed (stop)
                              → l2_processing → (cron handles rest)
      pending → l2_processing  (when no prefilter_config)
    """
    with Session(engine) as session:
        run = session.get(AssessmentRun, run_id)
        if run is None:
            logger.error("[execute_assessment_run] run_id=%s not found", run_id)
            return

        assessment = session.get(Assessment, run.assessment_id)
        if assessment is None:
            logger.error(
                "[execute_assessment_run] parent assessment %s not found for run %s",
                run.assessment_id,
                run_id,
            )
            return

        assessment_input = run.input or {}
        dataset_id = assessment.dataset_id

        dataset = get_assessment_dataset_by_id(
            session=session,
            dataset_id=dataset_id,
            organization_id=organization_id,
            project_id=project_id,
        )

        config_crud = ConfigCrud(session=session, project_id=project_id)
        parent_config = config_crud.read_one(run.config_id)
        if parent_config is not None and parent_config.tag != ConfigTag.ASSESSMENT:
            logger.error(
                "[execute_assessment_run] config %s has wrong tag for run %s",
                run.config_id,
                run_id,
            )
            update_assessment_run_status(
                session=session,
                run=run,
                status="failed",
                error_message="Config tag is not ASSESSMENT.",
            )
            recompute_assessment_status(session=session, assessment_id=assessment.id)
            return

        config_blob, error = resolve_evaluation_config(
            session=session,
            config_id=run.config_id,
            config_version=run.config_version,
            project_id=project_id,
            tag=ConfigTag.ASSESSMENT,
        )
        if error or config_blob is None:
            logger.error(
                "[execute_assessment_run] config resolution failed run_id=%s: %s",
                run_id,
                error,
            )
            update_assessment_run_status(
                session=session,
                run=run,
                status="failed",
                error_message=f"Config resolution failed: {error}",
            )
            recompute_assessment_status(session=session, assessment_id=assessment.id)
            return

        all_rows = _load_dataset_rows(session=session, dataset=dataset)
        if not all_rows:
            logger.error(
                "[execute_assessment_run] dataset %s has no rows for run %s",
                dataset_id,
                run_id,
            )
            update_assessment_run_status(
                session=session,
                run=run,
                status="failed",
                error_message="Dataset has no rows.",
            )
            recompute_assessment_status(session=session, assessment_id=assessment.id)
            return

        # prefilter pipeline
        rows_for_l2 = all_rows
        row_indices_for_l2: list[int] | None = None
        prefilter_config = assessment_input.get("prefilter_config")
        if prefilter_config:
            update_assessment_run_status(
                session=session, run=run, status="prefilter_processing"
            )
            try:
                rows_for_l2, row_indices_for_l2, _ = run_prefilter_pipeline(
                    run=run,
                    rows=all_rows,
                    prefilter_config=prefilter_config,
                    session=session,
                    organization_id=organization_id,
                    project_id=project_id,
                    attachments=[
                        AssessmentAttachment(**a)
                        for a in assessment_input.get("attachments") or []
                    ],
                )
                logger.info(
                    "[execute_assessment_run] prefilter done | run_id=%s | rows_to_l2=%s / %s",
                    run_id,
                    len(rows_for_l2),
                    len(all_rows),
                )
            except Exception as prefilter_exc:
                logger.error(
                    "[execute_assessment_run] prefilter failed run_id=%s | %s",
                    run_id,
                    prefilter_exc,
                    exc_info=True,
                )
                update_assessment_run_status(
                    session=session,
                    run=run,
                    status="prefilter_failed",
                    error_message=f"prefilter pipeline failed: {prefilter_exc}",
                )
                recompute_assessment_status(
                    session=session, assessment_id=assessment.id
                )
                return  # L2 does not run when prefilter fails

        # L2 batch submit
        try:
            batch_job = submit_assessment_batch(
                session=session,
                run=run,
                assessment=assessment,
                dataset=dataset,
                config_blob=config_blob,
                assessment_input=assessment_input,
                organization_id=organization_id,
                project_id=project_id,
                preloaded_rows=rows_for_l2,
                row_indices=row_indices_for_l2,
            )
            update_assessment_run_status(
                session=session,
                run=run,
                status="l2_processing",
                batch_job_id=batch_job.id,
                total_items=batch_job.total_items,
            )
            logger.info(
                "[execute_assessment_run] L2 batch submitted | run_id=%s | batch_job_id=%s",
                run_id,
                batch_job.id,
            )
        except Exception as e:
            logger.error(
                "[execute_assessment_run] L2 batch submit failed run_id=%s: %s",
                run_id,
                e,
                exc_info=True,
            )
            update_assessment_run_status(
                session=session,
                run=run,
                status="failed",
                error_message="Batch submission failed. Please try again or contact support.",
            )

        recompute_assessment_status(session=session, assessment_id=assessment.id)
