"""BATCH API-client submit entrypoint.

Resolves the ASSESSMENT config, persists the parent assessment + one execution,
seeds the staged-batch runtime bag, and hands the pipeline off to Celery. Only the
BATCH method is wired here; RESPONSE stays a route-level 501 stub (deferred).
"""

import logging

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from pydantic import ValidationError
from sqlmodel import Session

from app.crud.assessment import api
from app.crud.config import ConfigCrud, ConfigVersionCrud
from app.models.assessment import (
    AssessmentCreate,
    AssessmentMethod,
    AssessmentStatus,
    AssessmentSubmitResponse,
    BatchInput,
    BatchRunState,
    derive_method,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.services.assessment.api import batch as batch_service
from app.utils import validate_callback_url

logger = logging.getLogger(__name__)

# Attachment cell values are provided as URLs (base64 is unsupported for batch).
_URL_PREFIXES = ("http://", "https://", "gs://")
_ATTACHMENT_TYPES = ("image", "pdf")


def _validate_rows_against_schema(
    rows: list[dict[str, str]], input_schema: dict[str, dict[str, str | None]]
) -> None:
    """Validate every BATCH row against the config's input_schema.

    Raises 422 on the first offending row: a missing declared column, an
    unexpected column not in the schema, or an attachment column whose value is
    not a URL. The row index is named so the client can locate the bad row.
    """
    declared = set(input_schema)
    for idx, row in enumerate(rows):
        row_columns = set(row)
        missing = declared - row_columns
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"input.data[{idx}] is missing required column(s): {sorted(missing)}",
            )
        unexpected = row_columns - declared
        if unexpected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"input.data[{idx}] has column(s) not declared in input_schema: "
                    f"{sorted(unexpected)}"
                ),
            )
        for column, spec in input_schema.items():
            if (spec or {}).get("type") in _ATTACHMENT_TYPES:
                value = row.get(column, "")
                if not value.startswith(_URL_PREFIXES):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"input.data[{idx}] column '{column}' must be a URL for a "
                            f"'{spec.get('type')}' column."
                        ),
                    )


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


def submit(
    *,
    session: Session,
    request: AssessmentCreate,
    organization_id: int,
    project_id: int,
) -> AssessmentSubmitResponse:
    """Submit a BATCH assessment: persist state, seed the pipeline, dispatch the task."""
    from app.celery.tasks.job_execution import run_assessment_api_batch

    method = derive_method(request.input, dataset_id=None)
    if method != AssessmentMethod.BATCH:
        # RESPONSE is handled (stubbed) at the route; submit is BATCH-only for now.
        raise HTTPException(
            status_code=501, detail="Only BATCH input is wired in the assessment API."
        )

    # Delivery is webhook-only, so reject an unusable callback_url up front (HTTPS +
    # SSRF/private-IP guard) instead of after a full batch run is already paid for.
    try:
        validate_callback_url(str(request.callback_url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    batch_input: BatchInput = request.input
    blob, provider, model = _resolve_config(
        session=session, request=request, project_id=project_id
    )

    # input_schema is mandatory (enforced on the config), so every row must match it:
    # all declared columns present, no undeclared columns, attachments url-valued.
    input_columns = blob.assessment.params.get("input_schema") or {}
    _validate_rows_against_schema(batch_input.data, input_columns)

    # Validate transposition up front so bad attachment shapes fail as 422, not async.
    try:
        rows, _, _ = batch_service.build_rows(batch_input, input_columns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total_items = len(rows)

    assessment = api.create_assessment(
        session=session,
        method=AssessmentMethod.BATCH,
        input=batch_input.model_dump(mode="json"),
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

    pipeline = batch_service.build_pipeline(blob.pre_filters)
    bag: BatchRunState = {
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
        "input_schema": input_columns or None,
        "callback_url": str(request.callback_url),
        "request_metadata": request.request_metadata,
    }
    api.save_execution_state(session=session, execution=execution, state=bag)
    api.update_status(
        session=session, obj=assessment, status=AssessmentStatus.PROCESSING
    )

    trace_id = correlation_id.get() or ""
    try:
        run_assessment_api_batch.delay(
            execution_id=execution.id,
            organization_id=organization_id,
            project_id=project_id,
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.exception(
            "[submit] Failed to enqueue BATCH task | assessment_id=%s | execution_id=%s",
            assessment.id,
            execution.id,
        )
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
    logger.info(
        "[submit] Dispatched BATCH assessment | assessment_id=%s | execution_id=%s | "
        "provider=%s | stages=%s | rows=%s",
        assessment.id,
        execution.id,
        provider,
        [s["stage"] for s in pipeline],
        total_items,
    )

    return AssessmentSubmitResponse(
        assessment_id=assessment.id,
        status=assessment.status,
        message="Your assessment is being processed",
        inserted_at=assessment.inserted_at,
        updated_at=assessment.updated_at,
    )
