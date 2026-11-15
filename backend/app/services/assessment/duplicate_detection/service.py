"""Duplicate-detection job: load, run both checks, store the results, deliver them."""

import logging
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.instrumentation.utils import suppress_instrumentation
from pydantic import JsonValue, ValidationError
from sqlmodel import Session

from app.celery.utils import start_duplicate_detection
from app.core.db import engine
from app.core.telemetry import suppress_http_instrumentation
from app.crud.assessment import api
from app.crud.assessment import duplicate_detection as crud
from app.crud.config import ConfigVersionCrud
from app.crud.credentials import get_provider_credential
from app.models.assessment import Assessment, AssessmentRun, AssessmentStatus
from app.models.assessment.duplicate_detection import (
    DetectionErrors,
    DetectionLevel,
    DetectionParams,
    DuplicateDetectionCreate,
    DuplicateDetectionPublic,
    DuplicateDetectionResult,
    LenientMatch,
    StrictVerdict,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.models.provider_credentials import Provider
from app.services.assessment.duplicate_detection.detector import (
    DuplicateDetector,
    step_context,
)
from app.utils import send_callback, validate_callback_url

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _to_public(
    assessment: Assessment, runs: list[AssessmentRun]
) -> DuplicateDetectionPublic:
    """A job and its two runs as the API returns them."""
    by_level = {run.execution["level"]: run for run in runs}

    return DuplicateDetectionPublic(
        id=assessment.id,
        status=assessment.status,
        config={
            level: {"id": run.config_id, "version": run.config_version}
            for level, run in by_level.items()
        },
        request_metadata=assessment.input.get("request_metadata"),
        inserted_at=assessment.inserted_at,
        updated_at=assessment.updated_at,
        strict=by_level[DetectionLevel.STRICT].execution.get("result"),
        lenient=by_level[DetectionLevel.LENIENT].execution.get("result"),
        errors={level: run.error_message for level, run in by_level.items()},
    )


def _require_usable(level: str, config_blob: dict[str, JsonValue]) -> None:
    """422 unless the config can run a check: google-gcp, its params, and the output fields the check reads."""
    assessment = AssessmentConfigBlob.model_validate(config_blob).assessment

    try:
        if assessment is None or assessment.provider != Provider.GOOGLE_GCP:
            raise ValueError(f"it needs an assessment on {Provider.GOOGLE_GCP.value}")

        params = DetectionParams.model_validate(assessment.params)
        declared = params.json_output_schema.properties
        needed = StrictVerdict.model_fields
        where = ""

        if level == DetectionLevel.LENIENT:
            matches = declared.get("matches")
            declared = matches.items.properties if matches and matches.items else {}
            needed = LenientMatch.model_fields
            where = "matches[]."

        missing = [where + name for name in needed if name not in declared]

        if missing:
            raise ValueError(
                f"its json_output_schema must declare: {', '.join(missing)}"
            )

    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]) for error in exc.errors())

        raise HTTPException(
            status_code=422,
            detail=f"The {level} config cannot run a duplicate check: "
            f"missing or invalid assessment params: {fields}",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"The {level} config cannot run a duplicate check: {exc}",
        ) from exc


def submit(
    *,
    session: Session,
    request: DuplicateDetectionCreate,
    organization_id: int,
    project_id: int,
) -> DuplicateDetectionPublic:
    """Validate the request, store the job and hand it to Celery."""
    try:
        validate_callback_url(str(request.callback_url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # This feature's queries are a few ms each; spans for them only bury its steps.
    with suppress_instrumentation():
        for level, ref in request.config:
            version = ConfigVersionCrud(
                session=session,
                config_id=ref.id,
                project_id=project_id,
                tag=ConfigTag.ASSESSMENT,
            ).exists_or_raise(version_number=ref.version)

            _require_usable(level, version.config_blob)

        assessment, runs = crud.create_assessment(
            session=session,
            request=request,
            organization_id=organization_id,
            project_id=project_id,
        )

        trace.get_current_span().set_attribute(
            "duplicate_detection.assessment_id", str(assessment.id)
        )

    try:
        start_duplicate_detection(str(assessment.id), correlation_id.get() or "N/A")
    except Exception as exc:
        logger.error(
            f"[submit] Could not enqueue the job | assessment_id: {assessment.id}",
            exc_info=True,
        )
        api.update_status(
            session=session, obj=assessment, status=AssessmentStatus.FAILED
        )
        raise HTTPException(
            status_code=503,
            detail="Failed to dispatch the duplicate detection. Please retry.",
        ) from exc

    with suppress_instrumentation():
        return _to_public(assessment, runs)


def get_status(
    *, session: Session, assessment_id: UUID, organization_id: int, project_id: int
) -> DuplicateDetectionPublic:
    """A job's status and any results it has so far; 404 when it is not this project's."""
    with suppress_instrumentation():
        assessment = crud.get_assessment(
            session=session,
            assessment_id=assessment_id,
            organization_id=organization_id,
            project_id=project_id,
        )

        if assessment is None:
            raise HTTPException(
                status_code=404, detail=f"Duplicate detection {assessment_id} not found"
            )

        return _to_public(
            assessment,
            api.list_executions(session=session, assessment_id=assessment_id),
        )


def _store(
    *,
    session: Session,
    assessment: Assessment,
    runs: dict[str, AssessmentRun],
    result: DuplicateDetectionResult,
) -> AssessmentStatus:
    """Save both runs and the job's final status, which is decided by how many levels failed."""
    errors = [e for e in (result.errors.strict, result.errors.lenient) if e]
    span = trace.get_current_span()

    if not errors:
        status = AssessmentStatus.COMPLETED
    elif len(errors) == len(DetectionLevel):
        status = AssessmentStatus.FAILED
    else:
        status = AssessmentStatus.COMPLETED_WITH_ERRORS

    span.set_attribute("duplicate_detection.assessment_id", str(assessment.id))
    span.set_attribute("duplicate_detection.status", status)
    span.set_attribute("kaapi.organization_id", assessment.organization_id)
    span.set_attribute("kaapi.project_id", assessment.project_id)

    crud.save_result(
        session=session,
        assessment=assessment,
        status=status,
        outcomes=[
            (
                runs[DetectionLevel.STRICT],
                result.strict.model_dump() if result.strict else None,
                result.errors.strict,
            ),
            (
                runs[DetectionLevel.LENIENT],
                [match.model_dump() for match in result.lenient or []],
                result.errors.lenient,
            ),
        ],
    )

    return status


def _deliver(
    *,
    assessment_id: UUID,
    request: dict[str, JsonValue],
    status: AssessmentStatus,
    result: DuplicateDetectionResult,
    webhook: dict[str, str] | None,
) -> None:
    """Send the stored outcome to the caller's callback URL."""
    # Its own step, so it shows after the checks instead of inside the task's span.
    token = otel_context.attach(step_context())

    try:
        with (
            tracer.start_as_current_span("duplicate_detection.send_callback"),
            suppress_http_instrumentation(),
        ):
            sent = send_callback(
                request["callback_url"],
                {
                    "success": status != AssessmentStatus.FAILED,
                    "data": {
                        "id": str(assessment_id),
                        "status": status,
                        **result.model_dump(),
                        "request_metadata": request.get("request_metadata"),
                    },
                    "error": None,
                    "metadata": None,
                },
                webhook_secret=webhook.get("webhook_secret") if webhook else None,
            )

    finally:
        otel_context.detach(token)

    if not sent:
        logger.error(
            f"[_deliver] Webhook delivery failed | assessment_id: {assessment_id}"
        )


def run_duplicate_detection(*, assessment_id: UUID) -> None:
    """Run a claimed job: one session, no connection held during the provider calls."""
    with suppress_instrumentation(), Session(engine) as session:
        assessment = crud.claim_assessment(
            session=session,
            assessment_id=assessment_id,
            statuses=(AssessmentStatus.PENDING,),
        )

        if assessment is None:
            return

        request = assessment.input
        organization_id = assessment.organization_id
        project_id = assessment.project_id

        runs = {
            run.execution["level"]: run
            for run in api.list_executions(session=session, assessment_id=assessment_id)
        }

        webhook = get_provider_credential(
            session=session,
            org_id=organization_id,
            project_id=project_id,
            provider=Provider.WEBHOOK_SECRET,
        )

        try:
            blobs = {
                level: api.get_config_blob(
                    session=session, execution=run, project_id=project_id
                )
                for level, run in runs.items()
            }

            credentials = get_provider_credential(
                session=session,
                org_id=organization_id,
                project_id=project_id,
                provider=Provider.GOOGLE_GCP,
            )

            # Ends the read transaction so the connection is free during the provider calls.
            session.commit()

            detector = DuplicateDetector(DuplicateDetector.create_client(credentials))

            try:
                result = detector.detect(
                    query=request["query"],
                    strict=blobs[DetectionLevel.STRICT],
                    lenient=blobs[DetectionLevel.LENIENT],
                )
            finally:
                detector.close()

        except Exception as exc:
            error = (
                f"[KAAPI] Duplicate detection could not run "
                f"(code: {type(exc).__name__}): {exc}. Contact Kaapi."
            )

            logger.error(
                f"[run_duplicate_detection] {error} | assessment_id: {assessment_id}",
                exc_info=True,
            )

            session.rollback()
            result = DuplicateDetectionResult(
                errors=DetectionErrors(strict=error, lenient=error)
            )

        status = _store(
            session=session, assessment=assessment, runs=runs, result=result
        )

    _deliver(
        assessment_id=assessment_id,
        request=request,
        status=status,
        result=result,
        webhook=webhook,
    )


def fail_interrupted(*, assessment_id: UUID) -> None:
    """Fail a job its worker left unfinished and deliver that outcome; no-op once it has a result."""
    with suppress_instrumentation(), Session(engine) as session:
        assessment = crud.claim_assessment(
            session=session,
            assessment_id=assessment_id,
            # PENDING too: the worker can die after taking the task but before claiming it.
            statuses=(AssessmentStatus.PENDING, AssessmentStatus.PROCESSING),
        )

        if assessment is None:
            return

        logger.error(
            f"[fail_interrupted] Duplicate detection worker stopped before finishing; "
            f"failing the job | assessment_id: {assessment_id}"
        )

        request = assessment.input

        runs = {
            run.execution["level"]: run
            for run in api.list_executions(session=session, assessment_id=assessment_id)
        }

        webhook = get_provider_credential(
            session=session,
            org_id=assessment.organization_id,
            project_id=assessment.project_id,
            provider=Provider.WEBHOOK_SECRET,
        )

        error = (
            "[KAAPI] Duplicate detection was interrupted before it finished. "
            "Retry the request; contact Kaapi if it persists."
        )

        result = DuplicateDetectionResult(
            errors=DetectionErrors(strict=error, lenient=error)
        )

        status = _store(
            session=session, assessment=assessment, runs=runs, result=result
        )

    _deliver(
        assessment_id=assessment_id,
        request=request,
        status=status,
        result=result,
        webhook=webhook,
    )
