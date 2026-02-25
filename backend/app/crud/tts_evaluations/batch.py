"""Batch submission functions for TTS evaluation processing."""

import logging
from typing import Any

from sqlmodel import Session

from app.core.batch import (
    GeminiBatchProvider,
    create_tts_batch_requests,
    start_batch_job,
)
from app.models.batch_job import BatchJobType
from app.crud.tts_evaluations.result import (
    get_pending_results_for_run,
    update_tts_result,
)
from app.crud.tts_evaluations.run import update_tts_run
from app.models import EvaluationRun
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.stt_evaluations.gemini import GeminiClient
from app.services.tts_evaluations.constants import (
    DEFAULT_STYLE_PROMPT,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE_NAME,
)

logger = logging.getLogger(__name__)


def start_tts_evaluation_batch(
    *,
    session: Session,
    run: EvaluationRun,
    results: list[TTSResult],
    org_id: int,
    project_id: int,
) -> dict[str, Any]:
    """Submit Gemini batch jobs for TTS evaluation.

    Submits one batch job per model. Each batch job is tracked via
    its config containing evaluation_run_id and tts_provider.

    Args:
        session: Database session
        run: The evaluation run record
        results: List of TTSResult records (contains sample_text)
        org_id: Organization ID
        project_id: Project ID

    Returns:
        dict: Result with batch job information per model

    Raises:
        Exception: If batch submission fails for all models
    """
    models = run.providers or [DEFAULT_TTS_MODEL]

    logger.info(
        f"[start_tts_evaluation_batch] Starting batch submission | "
        f"run_id: {run.id}, result_count: {len(results)}, "
        f"models: {models}"
    )

    # Initialize Gemini client
    gemini_client = GeminiClient.from_credentials(
        session=session,
        org_id=org_id,
        project_id=project_id,
    )

    # Collect unique sample texts and their result IDs per model
    # Group results by model to build per-model batch requests
    results_by_model: dict[str, list[TTSResult]] = {}
    for result in results:
        results_by_model.setdefault(result.provider, []).append(result)

    # Submit one batch job per model
    batch_jobs: dict[str, Any] = {}
    first_batch_job_id: int | None = None

    for model in models:
        model_results = results_by_model.get(model, [])
        if not model_results:
            continue

        texts = [r.sample_text for r in model_results]
        keys = [str(r.id) for r in model_results]

        # Create JSONL batch requests for TTS
        jsonl_data = create_tts_batch_requests(
            texts=texts,
            voice_name=DEFAULT_VOICE_NAME,
            style_prompt=DEFAULT_STYLE_PROMPT,
            keys=keys,
        )

        model_path = f"models/{model}"
        batch_provider = GeminiBatchProvider(
            client=gemini_client.client, model=model_path
        )

        try:
            batch_job = start_batch_job(
                session=session,
                provider=batch_provider,
                provider_name="google",
                job_type=BatchJobType.TTS_EVALUATION,
                organization_id=org_id,
                project_id=project_id,
                jsonl_data=jsonl_data,
                config={
                    "model": model,
                    "tts_provider": model,
                    "evaluation_run_id": run.id,
                    "voice_name": DEFAULT_VOICE_NAME,
                    "style_prompt": DEFAULT_STYLE_PROMPT,
                },
            )

            batch_jobs[model] = {
                "batch_job_id": batch_job.id,
                "provider_batch_id": batch_job.provider_batch_id,
            }

            if first_batch_job_id is None:
                first_batch_job_id = batch_job.id

            logger.info(
                f"[start_tts_evaluation_batch] Batch job created | "
                f"run_id: {run.id}, model: {model}, "
                f"batch_job_id: {batch_job.id}"
            )

        except Exception as e:
            logger.error(
                f"[start_tts_evaluation_batch] Failed to submit batch | "
                f"model: {model}, error: {str(e)}"
            )
            pending = get_pending_results_for_run(
                session=session, run_id=run.id, provider=model
            )
            for result in pending:
                update_tts_result(
                    session=session,
                    result_id=result.id,
                    status=JobStatus.FAILED.value,
                    error_message=f"Batch submission failed for {model}: {str(e)}",
                )
            session.commit()

    if not batch_jobs:
        raise Exception("Batch submission failed for all models")

    # Link first batch job to the evaluation run (for pending run detection)
    update_tts_run(
        session=session,
        run_id=run.id,
        status="processing",
        batch_job_id=first_batch_job_id,
    )

    logger.info(
        f"[start_tts_evaluation_batch] Batch submission complete | "
        f"run_id: {run.id}, models_submitted: {list(batch_jobs.keys())}, "
        f"result_count: {len(results)}"
    )

    return {
        "success": True,
        "run_id": run.id,
        "batch_jobs": batch_jobs,
        "result_count": len(results),
    }
