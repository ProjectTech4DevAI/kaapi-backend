"""API-client submit entrypoint.

Resolves the ASSESSMENT config and validates the rows. BATCH persists the parent
assessment + one execution, seeds the staged-batch runtime bag and hands the pipeline
off to Celery; RESPONSE persists the parent assessment and hands its single row off.
"""

import logging
from typing import Any
from uuid import UUID, uuid4

from asgi_correlation_id import correlation_id
from celery import Task
from celery.result import AsyncResult
from fastapi import HTTPException
from pydantic import ValidationError
from sqlmodel import Session

from app.crud.assessment import api
from app.crud.assessment.submission import get_submission_by_id
from app.crud.config import ConfigCrud, ConfigVersionCrud
from app.models.assessment import (
    Assessment,
    AssessmentCreate,
    AssessmentMethod,
    AssessmentRun,
    AssessmentStatus,
    AssessmentSubmitResponse,
    BatchInput,
    BatchRunState,
    derive_method,
)
from app.models.config.assessment_blob import (
    ATTACHMENT_COLUMN_TYPES,
    AssessmentConfigBlob,
)
from app.models.config.config import ConfigTag
from app.services.assessment.api import batch as batch_service
from app.services.assessment.api.submission_store import (
    open_uploaded_rows,
    upload_submission_rows,
)
from app.services.assessment.utils.attachments import (
    attachment_type_from_url,
    split_attachment_urls,
)
from app.utils import validate_callback_url

logger = logging.getLogger(__name__)

# Attachment cell values are provided as URLs (base64 is unsupported for batch).
_URL_PREFIXES = ("http://", "https://", "gs://")


def _validate_rows_against_schema(
    rows: list[dict[str, str]], input_schema: dict[str, dict[str, str | None]]
) -> None:
    """Validate every BATCH row against the config's input_schema.

    Raises 422 on the first offending row: a strict column absent or blank, a column
    not in the schema, or an attachment value that is not a URL. A non-strict column
    may be omitted or blank. The row index is named so the client can locate the row.
    """
    declared = set(input_schema)
    strict_columns = [
        column for column, spec in input_schema.items() if (spec or {}).get("strict")
    ]
    for idx, row in enumerate(rows):
        missing = [c for c in strict_columns if not (row.get(c) or "").strip()]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"input.data[{idx}] is missing required column(s): {missing}",
            )
        unexpected = set(row) - declared
        if unexpected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"input.data[{idx}] has column(s) not declared in input_schema: "
                    f"{sorted(unexpected)}"
                ),
            )
        for column, spec in input_schema.items():
            column_type = (spec or {}).get("type")
            value = (row.get(column) or "").strip()
            if not value or column_type not in ATTACHMENT_COLUMN_TYPES:
                continue
            # A cell may hold several URLs; the builder sends each one separately.
            for url in split_attachment_urls(value):
                if not url.startswith(_URL_PREFIXES):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"input.data[{idx}] column '{column}' must be a URL for a "
                            f"'{column_type}' column."
                        ),
                    )
                # The provider only reports a mistyped column later, as an opaque format error.
                actual = attachment_type_from_url(url)
                if actual is not None and actual != column_type:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"input.data[{idx}] column '{column}' is declared "
                            f"'{column_type}' but its URL points to a '{actual}' file. "
                            f"Change the column type to '{actual}'."
                        ),
                    )


def _rows_from_submission(
    *,
    session: Session,
    submission_doc_id: UUID,
    organization_id: int,
    project_id: int,
) -> BatchInput:
    """Stream an uploaded submission's rows so they can be validated before dispatch."""
    submission = get_submission_by_id(
        session=session,
        submission_id=submission_doc_id,
        organization_id=organization_id,
        project_id=project_id,
    )
    try:
        with open_uploaded_rows(session=session, submission=submission) as stream:
            rows = list(stream)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "[_rows_from_submission] Could not read submission | submission_id=%s",
            submission_doc_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=502, detail="Failed to read the submission file from storage."
        ) from exc

    if not rows:
        raise HTTPException(
            status_code=422, detail=f"Submission {submission_doc_id} has no rows."
        )
    return BatchInput(data=rows)


def _resolve_config(
    *, session: Session, request: AssessmentCreate, project_id: int
) -> tuple[AssessmentConfigBlob, str, str]:
    """Resolve + validate the ASSESSMENT config. Returns (blob, batch_provider, model)."""
    config_ref = request.config
    config_crud = ConfigCrud(session=session, project_id=project_id)
    parent = config_crud.read_one(config_ref.id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"Config {config_ref.id} not found")
    if parent.tag != ConfigTag.ASSESSMENT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Config {config_ref.id} has tag '{parent.tag}' and cannot be "
                f"used for assessment. Only configs tagged 'ASSESSMENT' are allowed."
            ),
        )

    version_crud = ConfigVersionCrud(
        session=session,
        config_id=config_ref.id,
        project_id=project_id,
        tag=ConfigTag.ASSESSMENT,
    )
    version = version_crud.exists_or_raise(version_number=config_ref.version)
    try:
        blob = AssessmentConfigBlob.model_validate(version.config_blob)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    provider = blob.assessment.provider
    if not batch_service.is_supported_provider(provider):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Config provider '{blob.assessment.provider}' is not supported for "
                f"batch assessment."
            ),
        )
    model = blob.assessment.params.get("model")
    if not model:
        raise HTTPException(
            status_code=422, detail="Config assessment params must include a model."
        )
    return blob, provider, model


def _enqueue(
    *,
    session: Session,
    task: Task,
    assessment: Assessment,
    execution: AssessmentRun | None = None,
    **task_kwargs: Any,
) -> AsyncResult:
    """Dispatch ``task``; a broker failure marks the rows FAILED and returns a 503."""
    try:
        return task.delay(**task_kwargs)
    except Exception as exc:
        logger.exception("[_enqueue] Failed to enqueue | task=%s", task.name)
        if execution is not None:
            api.update_status(
                session=session, obj=execution, status=AssessmentStatus.FAILED
            )
        api.update_status(
            session=session, obj=assessment, status=AssessmentStatus.FAILED
        )
        raise HTTPException(
            status_code=503,
            detail="Failed to dispatch the assessment for processing. Please retry.",
        ) from exc


def _initial_state(
    *,
    request: AssessmentCreate,
    blob: AssessmentConfigBlob,
    provider: str,
    model: str,
    total_items: int,
) -> BatchRunState:
    pipeline = batch_service.build_pipeline(blob.pre_filters)
    return {
        "pipeline": pipeline,
        "stage": pipeline[0]["stage"],
        "stage_status": AssessmentStatus.PENDING.value,
        "stage_batches": {},
        "stage_output_urls": {},
        "verdicts": {},
        "counters": {},
        "gate_passed": [True] * total_items,
        "provider": provider,
        "model": model,
        "input_schema": blob.input_columns() or None,
        "callback_url": str(request.callback_url) if request.callback_url else None,
        "request_metadata": request.request_metadata,
    }


def _start_response(
    *,
    session: Session,
    request: AssessmentCreate,
    blob: AssessmentConfigBlob,
    provider: str,
    model: str,
    organization_id: int,
    project_id: int,
) -> AssessmentSubmitResponse:
    from app.celery.tasks.job_execution import run_assessment_response

    assessment = api.create_assessment(
        session=session,
        method=AssessmentMethod.RESPONSE,
        input=request.input.model_dump(),
        experiment_name=request.experiment_name,
        organization_id=organization_id,
        project_id=project_id,
    )
    execution = api.create_execution(
        session=session,
        assessment_id=assessment.id,
        config_id=request.config.id,
        config_version=request.config.version,
        total_items=1,
    )
    api.save_execution_state(
        session=session,
        execution=execution,
        state=_initial_state(
            request=request, blob=blob, provider=provider, model=model, total_items=1
        ),
    )
    api.update_status(
        session=session, obj=assessment, status=AssessmentStatus.PROCESSING
    )
    _enqueue(
        session=session,
        task=run_assessment_response,
        assessment=assessment,
        execution=execution,
        execution_id=execution.id,
        trace_id=correlation_id.get() or "",
    )
    return AssessmentSubmitResponse(
        assessment_id=assessment.id,
        status=assessment.status,
        message="Your assessment is being processed",
        inserted_at=assessment.inserted_at,
        updated_at=assessment.updated_at,
    )


def submit(
    *,
    session: Session,
    request: AssessmentCreate,
    organization_id: int,
    project_id: int,
) -> AssessmentSubmitResponse:
    """Submit an assessment: persist state, then dispatch the task for its method."""
    method = derive_method(request.input)

    # Reject an unusable callback_url up front (HTTPS + SSRF/private-IP guard) instead
    # of after a full batch run is already paid for. Absent means the client polls.
    if request.callback_url is not None:
        try:
            validate_callback_url(str(request.callback_url))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    batch_input = (
        request.input
        if isinstance(request.input, BatchInput)
        else request.input.as_batch()
    )
    submission_id = batch_input.submission_doc_id
    if submission_id is not None:
        batch_input = _rows_from_submission(
            session=session,
            submission_doc_id=submission_id,
            organization_id=organization_id,
            project_id=project_id,
        )

    blob, provider, model = _resolve_config(
        session=session, request=request, project_id=project_id
    )

    input_columns = blob.input_columns()
    if submission_id is not None:
        batch_input = BatchInput(
            data=[
                {key: value for key, value in row.items() if key in input_columns}
                for row in batch_input.data or []
            ]
        )
    _validate_rows_against_schema(batch_input.data or [], input_columns)

    # Validate transposition up front so bad attachment shapes fail as 422, not async.
    try:
        rows, _, _ = batch_service.build_rows(batch_input, input_columns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total_items = len(rows)

    if method == AssessmentMethod.RESPONSE:
        return _start_response(
            session=session,
            request=request,
            blob=blob,
            provider=provider,
            model=model,
            organization_id=organization_id,
            project_id=project_id,
        )

    from app.celery.tasks.job_execution import run_assessment_api_batch

    assessment_id = uuid4()
    # Only inline rows need storing; an uploaded submission is already immutable in
    # storage, so the run reads it through `submission_id` instead of copying it.
    submission_url = None
    if submission_id is None:
        submission_url = upload_submission_rows(
            session=session,
            assessment_id=assessment_id,
            project_id=project_id,
            batch_input=batch_input,
        )
        if not submission_url:
            raise HTTPException(
                status_code=503,
                detail="Failed to store the assessment submission. Please retry.",
            )
    assessment = api.create_assessment(
        session=session,
        assessment_id=assessment_id,
        method=AssessmentMethod.BATCH,
        input=None,
        submission_id=submission_id,
        submission_input=submission_url,
        experiment_name=request.experiment_name,
        organization_id=organization_id,
        project_id=project_id,
    )

    execution = api.create_execution(
        session=session,
        assessment_id=assessment.id,
        config_id=request.config.id,
        config_version=request.config.version,
        total_items=total_items,
    )

    bag = _initial_state(
        request=request,
        blob=blob,
        provider=provider,
        model=model,
        total_items=total_items,
    )
    api.save_execution_state(session=session, execution=execution, state=bag)
    api.update_status(
        session=session, obj=assessment, status=AssessmentStatus.PROCESSING
    )

    dispatched = _enqueue(
        session=session,
        task=run_assessment_api_batch,
        assessment=assessment,
        execution=execution,
        execution_id=execution.id,
        organization_id=organization_id,
        project_id=project_id,
        trace_id=correlation_id.get() or "",
    )
    logger.info(
        "[submit] Dispatched BATCH assessment | assessment_id=%s | execution_id=%s | "
        "task_id=%s | provider=%s | stages=%s | rows=%s",
        assessment.id,
        execution.id,
        dispatched.id,
        provider,
        [s["stage"] for s in bag["pipeline"]],
        total_items,
    )

    return AssessmentSubmitResponse(
        assessment_id=assessment.id,
        status=assessment.status,
        message="Your assessment is being processed",
        inserted_at=assessment.inserted_at,
        updated_at=assessment.updated_at,
    )
