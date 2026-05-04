"""Celery task function for STT automated metric computation."""

import logging
from typing import Any

from gevent import Timeout
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import update
from sqlmodel import Session, select

from app.core.db import engine
from app.core.util import now
from app.crud.stt_evaluations.run import update_stt_run
from app.models.job import JobStatus
from app.models.language import Language
from app.models.stt_evaluation import STTResult, STTSample
from app.services.stt_evaluations.metrics import (
    calculate_stt_metrics,
    compute_run_aggregate_scores,
)

logger = logging.getLogger(__name__)


def execute_metric_computation(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance: Any,
    organization_id: int,
    run_id: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute automated STT metrics (WER, CER, lenient WER, WIP) for a completed run.

    Fetches all successful results for the run, compares transcriptions against
    ground truth from STT samples, and updates scores on both individual results
    and the evaluation run.

    Args:
        project_id: Project ID
        job_id: Evaluation run ID (as string)
        task_id: Celery task ID
        task_instance: Celery task instance
        organization_id: Organization ID
        run_id: Evaluation run ID

    Returns:
        dict: Summary with scored/skipped/failed counts
    """
    logger.info(
        f"[execute_metric_computation] Starting | "
        f"run_id: {run_id}, project_id: {project_id}, "
        f"celery_task_id: {task_id}"
    )

    with Session(engine) as session:
        try:
            # Fetch all successful results for this run
            results_stmt = select(STTResult).where(
                STTResult.evaluation_run_id == run_id,
                STTResult.status == JobStatus.SUCCESS.value,
                STTResult.transcription.is_not(None),
            )
            results = list(session.exec(results_stmt).all())

            if not results:
                logger.info(
                    f"[execute_metric_computation] No successful results to score | "
                    f"run_id: {run_id}"
                )
                return {"success": True, "scored": 0, "skipped": 0, "failed": 0}

            # Batch-fetch all corresponding samples
            sample_ids = [r.stt_sample_id for r in results]
            samples_stmt = select(STTSample).where(STTSample.id.in_(sample_ids))
            samples = session.exec(samples_stmt).all()
            sample_map: dict[int, STTSample] = {s.id: s for s in samples}

            # Fetch language locales for all unique language_ids
            language_ids = {s.language_id for s in samples if s.language_id is not None}
            language_map: dict[int, str] = {}
            if language_ids:
                lang_stmt = select(Language).where(Language.id.in_(language_ids))
                languages = session.exec(lang_stmt).all()
                language_map = {lang.id: lang.locale for lang in languages}

            scored_count = 0
            skipped_count = 0
            failed_count = 0
            all_scores: list[dict[str, float]] = []
            score_updates: list[dict[str, Any]] = []
            timestamp = now()

            for result in results:
                sample = sample_map.get(result.stt_sample_id)

                # Skip if no sample or no ground truth
                if not sample or not sample.ground_truth:
                    skipped_count += 1
                    continue

                # Get language code for normalization
                language_code: str | None = None
                if sample.language_id is not None:
                    language_code = language_map.get(sample.language_id)

                try:
                    scores = calculate_stt_metrics(
                        hypothesis=result.transcription,
                        reference=sample.ground_truth,
                        language_code=language_code,
                    )
                    score_updates.append(
                        {
                            "id": result.id,
                            "score": scores,
                            "updated_at": timestamp,
                        }
                    )
                    all_scores.append(scores)
                    scored_count += 1
                except Exception as e:
                    logger.error(
                        f"[execute_metric_computation] Metric calculation failed | "
                        f"result_id: {result.id}, error: {e}",
                        exc_info=True,
                    )
                    failed_count += 1

            # Bulk update all result scores
            if score_updates:
                session.execute(update(STTResult), score_updates)
                session.commit()

            # Compute and store run-level aggregate scores
            if all_scores:
                aggregate = compute_run_aggregate_scores(all_scores)
                update_stt_run(
                    session=session,
                    run_id=run_id,
                    score=aggregate,
                )

            logger.info(
                f"[execute_metric_computation] Completed | "
                f"run_id: {run_id}, scored: {scored_count}, "
                f"skipped: {skipped_count}, failed: {failed_count}"
            )

            return {
                "success": True,
                "scored": scored_count,
                "skipped": skipped_count,
                "failed": failed_count,
            }

        except (Timeout, SoftTimeLimitExceeded) as err:
            timeout_err = TimeoutError("Task exceeded soft time limit")
            logger.error(
                f"[execute_metric_computation] STT metric computation timed out | run_id={run_id}"
            )
            update_stt_run(
                session=session,
                run_id=run_id,
                status="failed",
                error_message=str(timeout_err),
            )
            raise

        except Exception as e:
            logger.error(
                f"[execute_metric_computation] Failed | "
                f"run_id: {run_id}, error: {e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e)}
