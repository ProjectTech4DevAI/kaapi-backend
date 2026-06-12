"""Assessment batch result processing and polling.

processing but adapted for multi-provider (OpenAI + Google) support.
"""

import json
import logging
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session

from app.celery.tasks.job_execution import run_assessment_pipeline
from app.core.batch import BATCH_KEY, poll_batch_status, process_completed_batch
from app.core.batch.anthropic import (
    MessageBatchStatus,
    extract_text_from_anthropic_response,
)
from app.core.batch.base import BatchProvider
from app.core.batch.gemini import BatchJobState, extract_text_from_response_dict
from app.crud.assessment import (
    recompute_assessment_status,
    update_assessment_run_prefilter_stats,
    update_assessment_run_status,
)
from app.crud.job import get_batch_job
from app.models.assessment import Assessment, AssessmentRun, StageStatus
from app.services.assessment.stages import (
    GATE_STAGES,
    STAGE_PARSERS,
    _get_batch_provider,
    advance_or_finalize,
    load_raw_batch_results,
)
from app.services.llm.providers.registry import LLMProvider

logger = logging.getLogger(__name__)


def format_assessment_failure_message(exc: Exception) -> str:
    """Extract a DB-safe error message from assessment polling exceptions."""
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            message = detail.strip()
            if message:
                return message
        elif detail:
            try:
                return json.dumps(detail, ensure_ascii=False)
            except (TypeError, ValueError):
                pass

    message = str(exc).strip()
    return message or exc.__class__.__name__


def _sanitize_json_output(raw: str) -> str:
    """Escape control characters inside JSON string values that the model emitted literally.

    Strict structured-output mode should prevent this, but long Indic-language
    responses sometimes contain literal newlines / tabs inside string values,
    making the JSON unparseable.  This function walks the raw text once and
    replaces any bare control characters found while inside a JSON string with
    their JSON escape equivalents, producing valid JSON without touching the
    surrounding structure.
    """
    result: list[str] = []
    in_string = False
    escape_next = False

    for ch in raw:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\":
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)

    return "".join(result)


def parse_assessment_output(
    raw_results: list[dict[str, Any]],
    provider_name: str,
) -> list[dict[str, Any]]:
    """Parse batch results into assessment output format.

    Args:
        raw_results: Raw results from batch provider
        provider_name: Provider name ('openai', 'google', or 'anthropic')

    Returns:
        List of parsed results with row_id, output text, usage, etc.
    """
    results = []

    for result in raw_results:
        row_id = result.get(BATCH_KEY) or result.get("key", "unknown")

        if provider_name in (LLMProvider.OPENAI, LLMProvider.OPENAI_NATIVE):
            response = result.get("response", {})
            response_status = response.get("status_code")
            response_body = result.get("response", {}).get("body", {})
            error = result.get("error")

            if error:
                results.append(
                    {
                        "row_id": row_id,
                        "output": None,
                        "error": error.get("message", str(error)),
                        "usage": None,
                    }
                )
                continue

            if response_status and response_status >= 400:
                response_error = response_body.get("error", {})
                results.append(
                    {
                        "row_id": row_id,
                        "output": None,
                        "error": response_error.get(
                            "message", f"Request failed with status {response_status}"
                        ),
                        "usage": None,
                        "response_id": response_body.get("id"),
                    }
                )
                continue

            # Prefer the convenience field when present; otherwise concatenate all
            # output_text fragments so structured JSON isn't truncated mid-object.
            generated_text = response_body.get("output_text") or ""

            if not isinstance(generated_text, str) or not generated_text:
                output = response_body.get("output", "")
                text_chunks: list[str] = []

                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            for content in item.get("content", []):
                                if (
                                    isinstance(content, dict)
                                    and content.get("type") == "output_text"
                                ):
                                    text = content.get("text")
                                    if isinstance(text, str) and text:
                                        text_chunks.append(text)
                    generated_text = "".join(text_chunks)
                elif isinstance(output, str):
                    generated_text = output

            if generated_text:
                try:
                    generated_text = json.dumps(
                        json.loads(generated_text), ensure_ascii=False
                    )
                except (json.JSONDecodeError, TypeError):
                    # Model emitted literal control characters inside string values.
                    # Sanitize and retry once.
                    try:
                        sanitized = _sanitize_json_output(generated_text)
                        generated_text = json.dumps(
                            json.loads(sanitized), ensure_ascii=False
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass

            results.append(
                {
                    "row_id": row_id,
                    "output": generated_text,
                    "error": None if generated_text else "Empty response output",
                    "usage": response_body.get("usage"),
                    "response_id": response_body.get("id"),
                }
            )

        elif provider_name in (LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC_NATIVE):
            response = result.get("response")
            error = result.get("error")

            if error:
                results.append(
                    {
                        "row_id": row_id,
                        "output": None,
                        "error": str(error),
                        "usage": None,
                    }
                )
                continue

            if response:
                text = extract_text_from_anthropic_response(response)
                results.append(
                    {
                        "row_id": row_id,
                        "output": text if text else None,
                        "error": None if text else "Empty response output",
                        "usage": response.get("usage"),
                        "response_id": response.get("id"),
                    }
                )
            else:
                results.append(
                    {
                        "row_id": row_id,
                        "output": None,
                        "error": "Empty response",
                        "usage": None,
                    }
                )

        elif provider_name in (
            LLMProvider.GOOGLE_AISTUDIO,
            LLMProvider.GOOGLE_AISTUDIO_NATIVE,
        ):
            response = result.get("response")
            error = result.get("error")

            if error:
                results.append(
                    {
                        "row_id": row_id,
                        "output": None,
                        "error": str(error),
                        "usage": None,
                    }
                )
                continue

            if response:
                text = extract_text_from_response_dict(response)
                results.append(
                    {
                        "row_id": row_id,
                        "output": text if text else None,
                        "error": None if text else "Empty response output",
                        "usage": None,
                    }
                )
            else:
                results.append(
                    {
                        "row_id": row_id,
                        "output": None,
                        "error": "Empty response",
                        "usage": None,
                    }
                )

        else:
            logger.warning(
                "[parse_assessment_output] Unknown provider '%s' for row_id=%s — skipping",
                provider_name,
                row_id,
            )

    logger.info(
        "[parse_assessment_output] Parsed %s results | provider=%s",
        len(results),
        provider_name,
    )
    return results


_PROVIDER_SUCCESS = {
    "completed",
    BatchJobState.SUCCEEDED.value,
    MessageBatchStatus.ENDED.value,
}
_PROVIDER_FAILED = {
    "failed",
    "expired",
    "cancelled",
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}


def _poll_stage_outcome(session: Session, provider: BatchProvider, batch_job) -> str:
    """Poll one stage's batch; on success download+persist. Returns the outcome."""
    status_result = poll_batch_status(
        session=session, provider=provider, batch_job=batch_job
    )
    session.refresh(batch_job)
    status = batch_job.provider_status

    if status in _PROVIDER_SUCCESS:
        if batch_job.provider_output_file_id:
            process_completed_batch(
                session=session, provider=provider, batch_job=batch_job
            )
            return "completed"
        counts = status_result.get("request_counts") or {}
        if counts.get("completed", 0) == 0 and (
            counts.get("failed", 0) > 0 or status_result.get("error_file_id")
        ):
            return "failed"
        return "no_change"  # output genuinely not ready yet — retry next cycle
    if status in _PROVIDER_FAILED:
        return "failed"
    return "no_change"


def _record_gate_stats(
    session: Session, run: AssessmentRun, stage: str, batch_job, project_id: int
) -> None:
    """For a go/no-go stage, persist passed/rejected counts and accepted row indices.

    The accepted indices are stored on ``run.pipeline`` so the next stage's batch
    build reads them directly instead of re-downloading and re-parsing this batch.
    """
    try:
        raw = load_raw_batch_results(session, batch_job, project_id)
        outputs = parse_assessment_output(raw, batch_job.provider)
        parsed = STAGE_PARSERS[stage](outputs)
        total = len(parsed)
        passed = sum(1 for r in parsed.values() if r.get("verdict"))
        update_assessment_run_prefilter_stats(
            session=session,
            run=run,
            prefilter_total_rows=total,
            prefilter_total_passed=passed,
            prefilter_total_rejected=total - passed,
        )

        # Persist the cumulative accepted set (intersect with prior gates).
        accepted = {idx for idx, r in parsed.items() if r.get("verdict")}
        prev = (run.pipeline or {}).get("accepted_indices")
        if prev is not None:
            accepted &= set(prev)
        pipeline = dict(run.pipeline or {})
        pipeline["accepted_indices"] = sorted(accepted)
        run.pipeline = pipeline
        flag_modified(run, "pipeline")
    except Exception as exc:
        logger.warning(
            "[_record_gate_stats] run_id=%s stage=%s — %s", run.id, stage, exc
        )


def _fail_run_stage(
    session: Session, run: AssessmentRun, message: str
) -> dict[str, Any]:
    # Keep run.stage at the failed stage so a resume knows where to restart;
    # stage_status == FAILED is the failure marker.
    run.stage_status = StageStatus.FAILED
    update_assessment_run_status(
        session=session, run=run, status="failed", error_message=message
    )
    recompute_assessment_status(session=session, assessment_id=run.assessment_id)
    return {"run_id": run.id, "current_status": "failed", "action": "failed"}


async def process_run_batches(run: AssessmentRun, session: Session) -> dict[str, Any]:
    """Poll the run's current-stage batch; on completion advance to the next stage."""
    parent = session.get(Assessment, run.assessment_id)
    if not parent:
        raise ValueError(f"Parent assessment {run.assessment_id} not found")

    stage = run.stage
    if not stage or run.stage_status != StageStatus.PROCESSING:
        return {"run_id": run.id, "current_status": run.status, "action": "no_change"}

    batch_id = (run.stage_batches or {}).get(stage)
    batch_job = (
        get_batch_job(session=session, batch_job_id=batch_id) if batch_id else None
    )
    if not batch_job:
        return _fail_run_stage(session, run, f"Stage {stage} batch not found")

    # Transient errors here (DNS, network, provider hiccup) must NOT fail the run —
    # the batch is still running. Skip this cycle; the cron retries next tick.
    try:
        provider = _get_batch_provider(
            session=session,
            provider_name=batch_job.provider,
            organization_id=parent.organization_id,
            project_id=parent.project_id,
        )
        outcome = _poll_stage_outcome(session, provider, batch_job)
    except Exception as exc:
        logger.warning(
            "[process_run_batches] run_id=%s stage=%s poll error, will retry: %s",
            run.id,
            stage,
            exc,
        )
        return {"run_id": run.id, "current_status": run.status, "action": "no_change"}

    if outcome == "no_change":
        return {"run_id": run.id, "current_status": run.status, "action": "no_change"}
    if outcome == "failed":
        return _fail_run_stage(
            session, run, batch_job.error_message or f"Stage {stage} failed"
        )

    run.stage_status = StageStatus.COMPLETED
    if stage in GATE_STAGES:
        _record_gate_stats(session, run, stage, batch_job, parent.project_id)

    nxt = advance_or_finalize(run)
    session.add(run)
    session.commit()
    recompute_assessment_status(session=session, assessment_id=run.assessment_id)

    if nxt:
        try:
            run_assessment_pipeline.delay(
                run_id=run.id,
                organization_id=parent.organization_id,
                project_id=parent.project_id,
                trace_id="",
            )
        except Exception as exc:
            logger.error(
                "[process_run_batches] run_id=%s stage=%s enqueue failed — marking failed for resume: %s",
                run.id,
                run.stage,
                exc,
                exc_info=True,
            )
            return _fail_run_stage(
                session,
                run,
                "Failed to enqueue the next pipeline stage. Resume the run to retry.",
            )

    return {
        "run_id": run.id,
        "assessment_id": run.assessment_id,
        "experiment_name": parent.experiment_name,
        "current_status": run.status,
        "action": "processed",
    }
