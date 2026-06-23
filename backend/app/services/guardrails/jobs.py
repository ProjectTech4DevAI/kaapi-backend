import logging
from typing import Any
from uuid import UUID, uuid4

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from opentelemetry import trace
from sqlmodel import Session

from app.celery.utils import start_guardrails_job
from app.core.db import engine
from app.core.telemetry import log_context
from app.crud.jobs import JobCrud
from app.models import Job, JobStatus, JobType, JobUpdate
from app.models.guardrails import (
    GuardrailsCallbackData,
    GuardrailsCallbackResponse,
    GuardrailsCallbackUsage,
    GuardrailsOutput,
    GuardrailsOutputContent,
    GuardrailsRequest,
)
from app.models.llm.request import Validator
from app.services.llm.guardrails import apply_guardrails
from app.utils import APIResponse, get_webhook_secret, send_callback

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def start_job(
    db: Session,
    request: GuardrailsRequest,
    project_id: int,
    organization_id: int,
) -> Job:
    """Create a guardrails-only job and schedule its Celery task."""
    trace_id = correlation_id.get() or "N/A"
    logger.debug(
        f"[start_job] Received guardrails request | text_len={len(request.text)}, "
        f"validators={len(request.config)}, callback={request.callback_url is not None}, "
        f"project_id={project_id}, organization_id={organization_id}"
    )

    with log_context(
        tag="guardrails",
        lifecycle="guardrails.start_job",
        project_id=project_id,
        organization_id=organization_id,
    ), tracer.start_as_current_span("guardrails.start_job") as span:
        span.set_attribute("kaapi.project_id", project_id)
        span.set_attribute("kaapi.organization_id", organization_id)

        job_crud = JobCrud(session=db)
        # Raw text persisted intentionally: this endpoint inspects unsafe content.
        job = job_crud.create(
            job_type=JobType.LLM_GUARDRAILS,
            trace_id=trace_id,
            project_id=project_id,
            meta={"request": request.model_dump(mode="json")},
        )
        span.set_attribute("guardrails.job_id", str(job.id))
        logger.debug(
            f"[start_job] Job row created | job_id={job.id}, trace_id={trace_id}"
        )

        try:
            task_id = start_guardrails_job(
                project_id=project_id,
                job_id=str(job.id),
                trace_id=trace_id,
                request_data=request.model_dump(mode="json"),
                organization_id=organization_id,
            )
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.error(
                f"[start_job] Error starting Celery task: {e} | job_id={job.id}, project_id={project_id}",
                exc_info=True,
            )
            job_crud.update(
                job_id=job.id,
                job_update=JobUpdate(status=JobStatus.FAILED, error_message=str(e)),
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error while scheduling guardrails job",
            )

        logger.info(
            f"[start_job] Job scheduled for guardrails | job_id={job.id}, "
            f"project_id={project_id}, task_id={task_id}"
        )
        return job


def _coerce_int(value: Any) -> int:
    """Best-effort int coercion that tolerates malformed upstream usage payloads."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_callback_payload(
    *,
    response_id: str,
    safe_text: str,
    raw: dict[str, Any],
    request_metadata: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    """Build the /guardrails webhook payload; sanitised text at data.response.output.content.value."""
    upstream_data = raw.get("data") if isinstance(raw, dict) else None
    usage_payload: dict[str, Any] = {}
    if isinstance(upstream_data, dict):
        raw_usage = upstream_data.get("usage")
        if isinstance(raw_usage, dict):
            usage_payload = raw_usage

    callback_data = GuardrailsCallbackData(
        response=GuardrailsCallbackResponse(
            response_id=response_id,
            output=GuardrailsOutput(
                content=GuardrailsOutputContent(value=safe_text),
            ),
        ),
        usage=GuardrailsCallbackUsage(
            input_tokens=_coerce_int(usage_payload.get("input_tokens")),
            output_tokens=_coerce_int(usage_payload.get("output_tokens")),
            total_tokens=_coerce_int(usage_payload.get("total_tokens")),
            reasoning_tokens=_coerce_int(usage_payload.get("reasoning_tokens")),
        ),
        provider_raw_response=None,
    )

    metadata = dict(request_metadata or {})
    if "warnings" in metadata:
        logger.info(
            "[_build_callback_payload] Caller-supplied 'warnings' key in "
            "request_metadata overwritten by server warnings."
        )
    metadata["warnings"] = warnings

    return APIResponse.success_response(
        data=callback_data.model_dump(mode="json"),
        metadata=metadata,
    ).model_dump()


def _send_failure_callback(
    *,
    callback_url: str | None,
    error: str,
    request_metadata: dict[str, Any] | None,
    project_id: int,
    organization_id: int,
) -> None:
    if not callback_url:
        return
    metadata = dict(request_metadata or {})
    metadata.setdefault("warnings", [])
    payload = APIResponse.failure_response(error=error, metadata=metadata).model_dump()
    webhook_secret = get_webhook_secret(project_id, organization_id)
    with tracer.start_as_current_span("guardrails.send_callback") as cb_span:
        cb_span.set_attribute("callback.url", callback_url)
        cb_span.set_attribute("callback.status", "failure")
        logger.debug(
            f"[_send_failure_callback] Dispatching failure callback | "
            f"callback_url={callback_url}, error={error}"
        )
        send_callback(
            callback_url=callback_url,
            data=payload,
            webhook_secret=webhook_secret,
        )


def execute_job(
    *,
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance: Any,
    request_data: dict[str, Any],
    organization_id: int,
    **_: Any,
) -> dict[str, Any]:
    """Celery worker entrypoint for /guardrails jobs."""
    job_uuid = UUID(job_id)
    request = GuardrailsRequest.model_validate(request_data)
    callback_url = str(request.callback_url) if request.callback_url else None
    logger.debug(
        f"[execute_job] Picked up job | job_id={job_id}, task_id={task_id}, "
        f"text_len={len(request.text)}, validators={len(request.config)}, "
        f"callback={callback_url is not None}"
    )

    with log_context(
        tag="guardrails",
        lifecycle="guardrails.execute_job",
        job_id=job_id,
        project_id=project_id,
        organization_id=organization_id,
    ), tracer.start_as_current_span("guardrails.execute_job") as span:
        span.set_attribute("guardrails.job_id", job_id)
        span.set_attribute("kaapi.project_id", project_id)
        span.set_attribute("kaapi.organization_id", organization_id)

        with Session(engine) as session:
            JobCrud(session=session).update(
                job_id=job_uuid,
                job_update=JobUpdate(status=JobStatus.PROCESSING, task_id=task_id),
            )

        warnings: list[str] = []

        try:
            # Dedupe validator IDs (preserve order) to avoid double-billing upstream.
            seen_ids: set[UUID] = set()
            validators: list[Validator] = []
            duplicates = 0
            for g in request.config:
                if g.validator_config_id in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(g.validator_config_id)
                validators.append(Validator(validator_config_id=g.validator_config_id))
            if duplicates:
                warnings.append(
                    f"Request contained {duplicates} duplicate validator_config_id "
                    "entries; duplicates were ignored before calling the guardrails service."
                )
            if not validators:
                warnings.append(
                    "Request contained no usable validators after deduplication; "
                    "original text was returned unchanged."
                )
            logger.debug(
                f"[execute_job] Calling guardrails service | job_id={job_id}, "
                f"unique_validators={len(validators)}, duplicates_skipped={duplicates}"
            )
            outcome = apply_guardrails(
                text=request.text,
                validators=validators,
                job_id=job_uuid,
                project_id=project_id,
                organization_id=organization_id,
            )
            logger.debug(
                f"[execute_job] Guardrails outcome | job_id={job_id}, "
                f"bypassed={outcome.bypassed}, has_error={outcome.error is not None}, "
                f"safe_text_present={outcome.safe_text is not None}, "
                f"raw_keys={list(outcome.raw.keys()) if isinstance(outcome.raw, dict) else None}"
            )
        except Exception as e:
            logger.error(
                f"[execute_job] Guardrails execution crashed | job_id={job_id}: {e}",
                exc_info=True,
            )
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            with Session(engine) as session:
                JobCrud(session=session).update(
                    job_id=job_uuid,
                    job_update=JobUpdate(status=JobStatus.FAILED, error_message=str(e)),
                )
            _send_failure_callback(
                callback_url=callback_url,
                error=str(e),
                request_metadata=request.request_metadata,
                project_id=project_id,
                organization_id=organization_id,
            )
            return {"success": False, "error": str(e)}

        if outcome.bypassed:
            warnings.append(
                "Guardrails service was unavailable; original text was returned unchanged."
            )
        elif not outcome.raw and validators:
            # Validators submitted but none resolved — likely upstream unreachable.
            warnings.append(
                "Validators were submitted but none resolved against the guardrails "
                "service; original text was returned unchanged."
            )
            logger.warning(
                f"[execute_job] Validators were submitted but none resolved against "
                f"the guardrails service; returning original text. job_id={job_id}"
            )

        if outcome.error is None and outcome.safe_text is None and validators:
            warnings.append(
                "Guardrails service did not return a sanitised text; original text "
                "was returned unchanged."
            )

        # Hard block: guardrails service rejected the text.
        if outcome.error is not None:
            logger.info(
                f"[execute_job] Guardrails hard-blocked | job_id={job_id}, "
                f"error={outcome.error}"
            )
            with Session(engine) as session:
                JobCrud(session=session).update(
                    job_id=job_uuid,
                    job_update=JobUpdate(
                        status=JobStatus.FAILED,
                        error_message=outcome.error,
                        meta={
                            "request": request.model_dump(mode="json"),
                            "response": outcome.raw,
                        },
                    ),
                )
            _send_failure_callback(
                callback_url=callback_url,
                error=outcome.error,
                request_metadata=request.request_metadata,
                project_id=project_id,
                organization_id=organization_id,
            )
            return {"success": False, "error": outcome.error}

        safe_text = outcome.safe_text if outcome.safe_text is not None else request.text
        # Server-minted so callers always have a stable correlation handle.
        response_id = str(uuid4())
        callback_payload = _build_callback_payload(
            response_id=response_id,
            safe_text=safe_text,
            raw=outcome.raw,
            request_metadata=request.request_metadata,
            warnings=warnings,
        )

        with Session(engine) as session:
            JobCrud(session=session).update(
                job_id=job_uuid,
                job_update=JobUpdate(
                    status=JobStatus.SUCCESS,
                    meta={
                        "request": request.model_dump(mode="json"),
                        "response": outcome.raw,
                        "callback": {
                            "response_id": response_id,
                            "delivered": callback_url is not None,
                            "warnings": warnings,
                        },
                    },
                ),
            )

        logger.info(
            f"[execute_job] Job completed | job_id={job_id}, "
            f"warnings={len(warnings)}, callback={callback_url is not None}"
        )

        if callback_url:
            webhook_secret = get_webhook_secret(project_id, organization_id)
            with tracer.start_as_current_span("guardrails.send_callback") as cb_span:
                cb_span.set_attribute("callback.url", callback_url)
                cb_span.set_attribute("callback.status", "success")
                logger.debug(
                    f"[execute_job] Dispatching success callback | job_id={job_id}, "
                    f"callback_url={callback_url}"
                )
                send_callback(
                    callback_url=callback_url,
                    data=callback_payload,
                    webhook_secret=webhook_secret,
                )

        return callback_payload
