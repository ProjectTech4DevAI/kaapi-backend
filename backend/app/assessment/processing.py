"""Assessment batch result processing and polling.

Handles downloading completed batch results, parsing them, and updating
the assessment run status. Follows the same pattern as text evaluation
processing but adapted for multi-provider (OpenAI + Google) support.
"""

import json
import logging
from typing import Any

from sqlmodel import Session

from app.assessment.crud import (
    recompute_assessment_status,
    update_assessment_run_status,
)
from app.assessment.models import AssessmentRun
from app.core.batch import (
    BATCH_KEY,
    GeminiBatchProvider,
    OpenAIBatchProvider,
    download_batch_results,
    poll_batch_status,
    upload_batch_results_to_object_store,
)
from app.core.batch.base import BatchProvider
from app.core.batch.client import GeminiClient
from app.core.batch.gemini import BatchJobState, extract_text_from_response_dict
from app.crud.job import get_batch_job
from app.services.llm.providers.registry import LLMProvider
from app.utils import get_openai_client

logger = logging.getLogger(__name__)


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


def _get_batch_provider(
    session: Session,
    provider_name: str,
    organization_id: int,
    project_id: int,
) -> BatchProvider:
    """Get the appropriate batch provider instance."""
    if provider_name in (LLMProvider.OPENAI, LLMProvider.OPENAI_NATIVE):
        openai_client = get_openai_client(
            session=session,
            org_id=organization_id,
            project_id=project_id,
        )
        return OpenAIBatchProvider(client=openai_client)

    if provider_name in (LLMProvider.GOOGLE, LLMProvider.GOOGLE_NATIVE):
        gemini_client = GeminiClient.from_credentials(
            session=session,
            org_id=organization_id,
            project_id=project_id,
        )
        return GeminiBatchProvider(client=gemini_client.client)

    raise ValueError(f"Unsupported provider for assessment polling: {provider_name}")


def parse_assessment_output(
    raw_results: list[dict[str, Any]],
    provider_name: str,
) -> list[dict[str, Any]]:
    """Parse batch results into assessment output format.

    Args:
        raw_results: Raw results from batch provider
        provider_name: Provider name ('openai' or 'google')

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

        elif provider_name in (LLMProvider.GOOGLE, LLMProvider.GOOGLE_NATIVE):
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

    logger.info(
        f"[parse_assessment_output] Parsed {len(results)} results | "
        f"provider={provider_name}"
    )
    return results


async def check_and_process_assessment(
    run: AssessmentRun,
    session: Session,
) -> dict[str, Any]:
    """Check assessment batch status and process if completed.

    Args:
        run: AssessmentRun to check
        session: Database session

    Returns:
        Dict with status information
    """
    log_prefix = f"[check_and_process_assessment][assessment_run={run.id}]"
    previous_status = run.status

    try:
        if not run.batch_job_id:
            raise ValueError(f"Assessment run {run.id} has no batch_job_id")

        batch_job = get_batch_job(session=session, batch_job_id=run.batch_job_id)
        if not batch_job:
            raise ValueError(f"BatchJob {run.batch_job_id} not found")

        # Get provider and poll status
        provider = _get_batch_provider(
            session=session,
            provider_name=batch_job.provider,
            organization_id=run.organization_id,
            project_id=run.project_id,
        )
        status_result = poll_batch_status(
            session=session,
            provider=provider,
            batch_job=batch_job,
        )
        session.refresh(batch_job)

        provider_status = batch_job.provider_status

        if (
            provider_status == "completed"
            or provider_status == BatchJobState.SUCCEEDED.value
        ):
            if not batch_job.provider_output_file_id:
                request_counts = status_result.get("request_counts") or {}
                error_file_id = status_result.get("error_file_id")
                failed_count = request_counts.get("failed", 0)
                completed_count = request_counts.get("completed", 0)
                total_count = request_counts.get("total", 0)

                if error_file_id and failed_count > 0 and completed_count == 0:
                    error_msg = (
                        f"Batch completed with {failed_count} failed request(s)"
                        f" and no successful outputs"
                    )
                    if total_count:
                        error_msg += f" out of {total_count}"
                    error_msg += f" (error_file_id: {error_file_id})"

                    update_assessment_run_status(
                        session=session,
                        run=run,
                        status="failed",
                        error_message=error_msg,
                    )
                    if run.assessment_id is not None:
                        recompute_assessment_status(
                            session=session, assessment_id=run.assessment_id
                        )

                    return {
                        "run_id": run.id,
                        "assessment_id": run.assessment_id,
                        "run_name": run.run_name,
                        "previous_status": previous_status,
                        "current_status": "failed",
                        "provider_status": provider_status,
                        "action": "failed",
                        "error": error_msg,
                    }

                logger.info(
                    f"{log_prefix} Batch completed but output file is not ready yet | "
                    f"batch_job_id={batch_job.id} | provider_status={provider_status}"
                )
                return {
                    "run_id": run.id,
                    "assessment_id": run.assessment_id,
                    "run_name": run.run_name,
                    "previous_status": previous_status,
                    "current_status": run.status,
                    "provider_status": provider_status,
                    "action": "no_change",
                }

            # Download and process results
            raw_results = download_batch_results(provider=provider, batch_job=batch_job)

            # Upload raw results to object store
            object_store_url = None
            try:
                object_store_url = upload_batch_results_to_object_store(
                    session=session, batch_job=batch_job, results=raw_results
                )
            except Exception as e:
                logger.warning(f"{log_prefix} Object store upload failed: {e}")

            # Parse results
            parsed = parse_assessment_output(raw_results, batch_job.provider)
            error_count = sum(1 for r in parsed if r.get("error"))
            success_count = sum(1 for r in parsed if not r.get("error"))

            # Update run status
            error_msg = f"{error_count} item(s) failed" if error_count > 0 else None
            run_status = (
                "failed"
                if parsed and success_count == 0 and error_count > 0
                else "completed"
            )

            if not parsed:
                run_status = "failed"
                error_msg = "Batch completed but no valid results were produced"

            update_assessment_run_status(
                session=session,
                run=run,
                status=run_status,
                error_message=error_msg,
                object_store_url=object_store_url,
            )
            if run.assessment_id is not None:
                recompute_assessment_status(
                    session=session, assessment_id=run.assessment_id
                )

            return {
                "run_id": run.id,
                "assessment_id": run.assessment_id,
                "run_name": run.run_name,
                "previous_status": previous_status,
                "current_status": run_status,
                "provider_status": provider_status,
                "action": "processed" if run_status == "completed" else "failed",
                "total_results": len(parsed),
                "errors": error_count,
            }

        elif provider_status in (
            "failed",
            "expired",
            "cancelled",
            BatchJobState.FAILED.value,
            BatchJobState.CANCELLED.value,
            BatchJobState.EXPIRED.value,
        ):
            error_msg = batch_job.error_message or f"Batch {provider_status}"
            update_assessment_run_status(
                session=session,
                run=run,
                status="failed",
                error_message=error_msg,
            )
            if run.assessment_id is not None:
                recompute_assessment_status(
                    session=session, assessment_id=run.assessment_id
                )

            return {
                "run_id": run.id,
                "assessment_id": run.assessment_id,
                "run_name": run.run_name,
                "previous_status": previous_status,
                "current_status": "failed",
                "provider_status": provider_status,
                "action": "failed",
                "error": error_msg,
            }

        else:
            # Still processing
            return {
                "run_id": run.id,
                "assessment_id": run.assessment_id,
                "run_name": run.run_name,
                "previous_status": previous_status,
                "current_status": run.status,
                "provider_status": provider_status,
                "action": "no_change",
            }

    except Exception as e:
        logger.error(
            f"{log_prefix} Error checking assessment: {e}",
            exc_info=True,
        )
        update_assessment_run_status(
            session=session,
            run=run,
            status="failed",
            error_message="Processing failed. Check server logs for details.",
        )
        if run.assessment_id is not None:
            recompute_assessment_status(
                session=session, assessment_id=run.assessment_id
            )
        return {
            "run_id": run.id,
            "assessment_id": run.assessment_id,
            "run_name": run.run_name,
            "previous_status": previous_status,
            "current_status": "failed",
            "provider_status": "unknown",
            "action": "failed",
            "error": "Processing failed",
        }


async def poll_all_pending_assessments(session: Session) -> dict[str, Any]:
    """Backward-compatible wrapper for parent-first assessment polling."""
    from app.assessment.cron import poll_all_pending_assessment_evaluations

    return await poll_all_pending_assessment_evaluations(session=session)
