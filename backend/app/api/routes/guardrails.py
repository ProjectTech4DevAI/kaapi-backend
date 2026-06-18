import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from opentelemetry import trace

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.rate_monitor import monitor_rate
from app.core.telemetry import log_context
from app.crud.jobs import JobCrud
from app.models import JobStatus, JobType
from app.models.guardrails import (
    GuardrailsCallbackData,
    GuardrailsJobImmediatePublic,
    GuardrailsJobPublic,
    GuardrailsRequest,
)
from app.services.guardrails.jobs import start_job
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Guardrails"])
guardrails_callback_router = APIRouter()


@guardrails_callback_router.post(
    "{$callback_url}",
    name="guardrails_callback",
)
def guardrails_callback_notification(body: APIResponse[GuardrailsCallbackData]):
    """
    Callback endpoint specification for /guardrails completion.

    The callback will receive:
    - On success: APIResponse with success=True and data containing
      GuardrailsCallbackData (sanitised text under data.response.output).
    - On hard-block / failure: APIResponse with success=False and error set.
    - metadata field will always include any request_metadata supplied with
      the original request, plus a `warnings` list.
    """
    ...


@router.post(
    "/guardrails",
    description=load_description("guardrails/apply_guardrails.md"),
    response_model=APIResponse[GuardrailsJobImmediatePublic],
    callbacks=guardrails_callback_router.routes,
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("llm_call")),
    ],
)
def apply_guardrails_endpoint(
    _current_user: AuthContextDep,
    session: SessionDep,
    request: GuardrailsRequest,
)  -> APIResponse[GuardrailsJobImmediatePublic]:
    """Initiate a guardrails-only job. Returns the job_id immediately; the
    sanitised text is delivered via callback_url (or polled via GET)."""
    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    with log_context(
        tag="guardrails",
        system="guardrails",
        lifecycle="api.guardrails.apply",
        project_id=project_id,
        organization_id=organization_id,
        callback_enabled=request.callback_url is not None,
    ):
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("kaapi.project_id", project_id)
            span.set_attribute("kaapi.organization_id", organization_id)
            span.set_attribute(
                "guardrails.callback_enabled", request.callback_url is not None
            )

        if request.callback_url:
            validate_callback_url(str(request.callback_url))

        job_id = start_job(
            db=session,
            request=request,
            project_id=project_id,
            organization_id=organization_id,
        )

        if span.is_recording():
            span.set_attribute("guardrails.job_id", str(job_id))

        job = JobCrud(session=session).get(job_id=job_id, project_id=project_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        message = (
            "Guardrails are being applied; the sanitised text will be delivered via callback."
            if request.callback_url
            else "Guardrails are being applied; poll GET /guardrails/{job_id} for the result."
        )

        return APIResponse.success_response(
            data=GuardrailsJobImmediatePublic(
                job_id=job.id,
                status=job.status.value,
                message=message,
                job_inserted_at=job.inserted_at,
                job_updated_at=job.updated_at,
            )
        )


@router.get(
    "/guardrails/{job_id}",
    response_model=APIResponse[GuardrailsJobPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_guardrails_job_status(
    _current_user: AuthContextDep,
    session: SessionDep,
    job_id: UUID,
) -> APIResponse[GuardrailsJobPublic]:
    """Poll for a /guardrails job's status and result.

    On SUCCESS the sanitised text is rehydrated from the persisted upstream
    response stored on ``job.meta``.
    """
    project_id = _current_user.project_.id

    with log_context(
        tag="guardrails",
        system="guardrails",
        lifecycle="api.guardrails.status",
        job_id=job_id,
        project_id=project_id,
        organization_id=_current_user.organization_.id,
    ):
        job = JobCrud(session=session).get(job_id=job_id, project_id=project_id)
        if not job or job.job_type != JobType.LLM_GUARDRAILS:
            # Hide non-guardrails jobs behind a 404 so this endpoint cannot
            # be used to enumerate or peek at LLM-call / chain job rows.
            raise HTTPException(status_code=404, detail="Job not found")

        meta = job.meta if isinstance(job.meta, dict) else {}
        callback_blob = meta.get("callback") if isinstance(meta, dict) else None

        warnings: list[str] = []
        if isinstance(callback_blob, dict):
            raw_warnings = callback_blob.get("warnings")
            if isinstance(raw_warnings, list):
                warnings = [w for w in raw_warnings if isinstance(w, str)]

        guardrails_response: GuardrailsCallbackData | None = None
        if job.status.value == JobStatus.SUCCESS:
            response_blob = meta.get("response") or {}
            data_blob = (
                response_blob.get("data") if isinstance(response_blob, dict) else None
            ) or {}
            safe_text = (
                data_blob.get("safe_text") if isinstance(data_blob, dict) else None
            )
            request_blob = meta.get("request") or {}
            original_text = (
                request_blob.get("text") if isinstance(request_blob, dict) else None
            )
            value = safe_text if isinstance(safe_text, str) else (original_text or "")

            # Prefer the server-minted response_id stamped on the callback
            # blob; fall back to None if (for older rows) it is absent.
            response_id: str | None = None
            if isinstance(callback_blob, dict):
                rid = callback_blob.get("response_id")
                if isinstance(rid, str):
                    response_id = rid

            guardrails_response = GuardrailsCallbackData.model_validate(
                {
                    "response": {
                        "response_id": response_id,
                        "output": {
                            "type": "text",
                            "content": {"format": "text", "value": value},
                        },
                    },
                    "usage": (
                        data_blob.get("usage")
                        if isinstance(data_blob, dict)
                        and isinstance(data_blob.get("usage"), dict)
                        else {}
                    ),
                    "provider_raw_response": None,
                }
            )

        return APIResponse.success_response(
            data=GuardrailsJobPublic(
                job_id=job.id,
                status=job.status.value,
                guardrails_response=guardrails_response,
                error_message=job.error_message,
                warnings=warnings,
            )
        )
