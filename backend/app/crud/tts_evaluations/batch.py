"""Batch submission functions for TTS evaluation processing."""

import logging
from typing import Any

from sqlmodel import Session

from app.core.batch import (
    GeminiBatchProvider,
    GeminiClient,
    create_tts_batch_requests,
    start_batch_job,
)
from app.crud.tts_evaluations.result import (
    get_pending_results_for_run,
    update_tts_result,
)
from app.crud.tts_evaluations.run import update_tts_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJobType
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.constants import (
    DEFAULT_STYLE_PROMPT,
    TTSExecutionModeEnum,
    get_tts_model_spec,
)

logger = logging.getLogger(__name__)


class TTSBatchSubmissionError(Exception):
    """Raised when no batch-mode model could be submitted to its provider."""


def start_tts_evaluation_batch(
    *,
    session: Session,
    run: EvaluationRun,
    results: list[TTSResult],
    models: list[str],
    org_id: int,
    project_id: int,
) -> dict[str, Any]:
    """Submit Gemini batch jobs for the batch-mode models of a TTS evaluation.

    Submits one batch job per model. Each batch job is tracked via
    its config containing evaluation_run_id and tts_provider.

    Args:
        session: Database session
        run: The evaluation run record
        results: TTSResult records (loaded); rows for non-batch models are ignored
        models: Batch-mode models to submit (sync models must not be passed)
        org_id: Organization ID
        project_id: Project ID

    Returns:
        dict: Result with batch job information per model

    Raises:
        ValueError: If a non-batch model is passed
        TTSBatchSubmissionError: If batch submission fails for every model
    """
    for model in models:
        if get_tts_model_spec(model).execution_mode != TTSExecutionModeEnum.BATCH:
            raise ValueError(f"Model '{model}' is not a batch-mode TTS model")

    # Capture plain values up front: start_batch_job commits, which expires every
    # ORM instance in the session, and touching them later would issue a reload
    # per row (and reopen a transaction right before the next network call).
    run_id = run.id
    texts_by_model: dict[str, list[str]] = {}
    keys_by_model: dict[str, list[str]] = {}
    for result in results:
        texts_by_model.setdefault(result.provider, []).append(result.sample_text)
        keys_by_model.setdefault(result.provider, []).append(str(result.id))

    logger.info(
        f"[start_tts_evaluation_batch] Starting batch submission | "
        f"run_id: {run_id}, result_count: {len(results)}, "
        f"models: {models}"
    )

    gemini_client = GeminiClient.from_credentials(
        session=session,
        org_id=org_id,
        project_id=project_id,
    )

    batch_jobs: dict[str, Any] = {}
    first_batch_job_id: int | None = None
    submitted_count = 0

    for model in models:
        texts = texts_by_model.get(model, [])
        if not texts:
            continue

        voice_name = get_tts_model_spec(model).default_voice
        jsonl_data = create_tts_batch_requests(
            texts=texts,
            voice_name=voice_name,
            style_prompt=DEFAULT_STYLE_PROMPT,
            keys=keys_by_model[model],
        )

        model_path = f"models/{model}"
        batch_provider = GeminiBatchProvider(
            client=gemini_client.client, model=model_path
        )

        try:
            batch_job = start_batch_job(
                session=session,
                provider=batch_provider,
                provider_name=get_tts_model_spec(model).provider.value,
                job_type=BatchJobType.TTS_EVALUATION,
                organization_id=org_id,
                project_id=project_id,
                jsonl_data=jsonl_data,
                config={
                    "model": model,
                    "tts_provider": model,
                    "evaluation_run_id": run_id,
                    "voice_name": voice_name,
                    "style_prompt": DEFAULT_STYLE_PROMPT,
                },
            )

            batch_jobs[model] = {
                "batch_job_id": batch_job.id,
                "provider_batch_id": batch_job.provider_batch_id,
            }
            submitted_count += len(texts)

            if first_batch_job_id is None:
                first_batch_job_id = batch_job.id

            logger.info(
                f"[start_tts_evaluation_batch] Batch job created | "
                f"run_id: {run_id}, model: {model}, "
                f"batch_job_id: {batch_job.id}"
            )

        except Exception as e:
            logger.error(
                f"[start_tts_evaluation_batch] Failed to submit batch | "
                f"run_id: {run_id}, model: {model}, error: {str(e)}",
                exc_info=True,
            )
            pending = get_pending_results_for_run(
                session=session, run_id=run_id, provider=model
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
        raise TTSBatchSubmissionError("Batch submission failed for all models")

    # Link first batch job to the evaluation run (for pending run detection)
    update_tts_run(
        session=session,
        run_id=run_id,
        status="processing",
        batch_job_id=first_batch_job_id,
    )

    logger.info(
        f"[start_tts_evaluation_batch] Batch submission complete | "
        f"run_id: {run_id}, models_submitted: {list(batch_jobs.keys())}, "
        f"result_count: {submitted_count}"
    )

    return {
        "success": True,
        "run_id": run_id,
        "batch_jobs": batch_jobs,
        "result_count": submitted_count,
    }
