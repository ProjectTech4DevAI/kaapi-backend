"""Durable result dumps for the BATCH API-client path.

Records every provider dump on ``assessment.result_files`` as ``{kind: {object_store_url}}``,
builds ``errors.jsonl`` at terminal time, and presigns both into the callback envelope.
Nothing here raises into the terminal path: a missing dump degrades to a missing key.
"""

import json
import logging
from datetime import timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.config import settings
from app.core.storage_utils import upload_jsonl_to_object_store
from app.core.util import now
from app.crud.assessment import api
from app.crud.job import get_batch_job
from app.models.assessment import (
    Assessment,
    AssessmentRun,
    BatchRunState,
)
from app.models.batch_job import BatchJob
from app.services.assessment.api.batch import ApiStage, _build_batch_provider

logger = logging.getLogger(__name__)

RESULTS_FILE_KIND = "results"
ERRORS_FILE_KIND = "errors"
ERRORS_FILENAME = "errors.jsonl"

# 86400 is the storage layer's own ceiling, so the presigned urls live exactly one day.
SIGNED_URL_EXPIRY_SECONDS = settings.MAX_SIGNED_URL_EXPIRY_SECONDS


class ErrorRecordEnum(StrEnum):
    """``type`` tag on each errors.jsonl row, so the file is self-describing."""

    EXECUTION_ERROR = "execution_error"
    ROW_ERROR = "row_error"
    PROVIDER_ERROR_FILE = "provider_error_file"
    PROVIDER_ERROR_FILE_UNAVAILABLE = "provider_error_file_unavailable"


def assessment_subdirectory(assessment_id: UUID) -> str:
    """Object-store prefix holding every file one assessment produces."""
    return f"assessment/{assessment_id}"


def stage_file_kind(stage: str) -> str:
    """Result-file kind for a stage's dump; the assessment dump is the run's ``results``."""
    if stage == ApiStage.ASSESSMENT.value:
        return RESULTS_FILE_KIND
    return f"{stage}_results"


def record_stage_dump(
    *,
    session: Session,
    assessment: Assessment,
    stage: str,
    url: str | None,
) -> None:
    """Record one stage's dump on the parent row as soon as the stage completes.

    No-op on a falsy url: ``process_completed_batch`` swallows a failed upload.
    """
    if not url:
        logger.warning(
            "[record_stage_dump] No dump url to record | assessment_id=%s | stage=%s",
            assessment.id,
            stage,
        )
        return

    api.set_result_files(
        session=session,
        assessment=assessment,
        files={stage_file_kind(stage): {"object_store_url": url}},
    )


def _row_error_rows(stage_errors: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Per-row errors captured at parse time, flattened across stages."""
    rows: list[dict[str, Any]] = []
    for stage, errors in stage_errors.items():
        for row_index, error in errors.items():
            rows.append(
                {
                    "type": ErrorRecordEnum.ROW_ERROR.value,
                    "stage": stage,
                    "row_index": int(row_index) if row_index.isdigit() else row_index,
                    "error": error,
                }
            )
    return rows


def _error_file_rows(
    *, session: Session, assessment: Assessment, stage: str, batch_job: BatchJob
) -> list[dict[str, Any]]:
    """Parsed lines of one stage batch's provider error file (OpenAI only).

    Degrades to a single ``provider_error_file_unavailable`` row: the client still learns
    the file existed and could not be read.
    """
    file_id = batch_job.provider_error_file_id
    if not file_id:
        return []

    try:
        provider = _build_batch_provider(
            session=session,
            provider_name=batch_job.provider,
            organization_id=assessment.organization_id,
            project_id=assessment.project_id,
        )
        content = provider.download_file(file_id)
    except Exception as exc:
        # Provider SDKs raise heterogeneous types here and the dump is best-effort.
        message = (
            f"[KAAPI] Could not download the provider error file "
            f"(code: {type(exc).__name__}): {exc}"
        )
        logger.error(
            "[_error_file_rows] %s | batch_job_id=%s | stage=%s",
            message,
            batch_job.id,
            stage,
            exc_info=True,
        )
        return [
            {
                "type": ErrorRecordEnum.PROVIDER_ERROR_FILE_UNAVAILABLE.value,
                "stage": stage,
                "provider_error_file_id": file_id,
                "error": message,
            }
        ]

    rows: list[dict[str, Any]] = []
    for line in content.strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(
                "[_error_file_rows] Unparseable error-file line, skipping | "
                "batch_job_id=%s | stage=%s",
                batch_job.id,
                stage,
            )
            continue
        rows.append(
            {
                "type": ErrorRecordEnum.PROVIDER_ERROR_FILE.value,
                "stage": stage,
                "provider_error_file_id": file_id,
                "entry": entry,
            }
        )
    return rows


def build_and_upload_errors(
    *,
    session: Session,
    execution: AssessmentRun,
    assessment: Assessment,
    bag: BatchRunState,
    failure_message: str | None,
) -> str | None:
    """Assemble and upload the run's ``errors.jsonl``. Returns its object-store url.

    Uploaded even when there are no rows, so the "both a results and an errors url"
    promise holds on the clean-success path too.
    """
    rows: list[dict[str, Any]] = []
    if failure_message:
        rows.append(
            {
                "type": ErrorRecordEnum.EXECUTION_ERROR.value,
                "stage": bag.get("stage"),
                "error": failure_message,
            }
        )
    rows.extend(_row_error_rows(bag.get("stage_errors") or {}))
    for stage, batch_job_id in (bag.get("stage_batches") or {}).items():
        batch_job = (
            get_batch_job(session=session, batch_job_id=batch_job_id)
            if batch_job_id
            else None
        )
        if batch_job is not None:
            rows.extend(
                _error_file_rows(
                    session=session,
                    assessment=assessment,
                    stage=stage,
                    batch_job=batch_job,
                )
            )

    try:
        storage = get_cloud_storage(session=session, project_id=assessment.project_id)
    except Exception:
        logger.error(
            "[build_and_upload_errors] Storage unavailable, no errors dump | "
            "execution_id=%s | rows=%s",
            execution.id,
            len(rows),
            exc_info=True,
        )
        return None

    url = upload_jsonl_to_object_store(
        storage=storage,
        results=rows,
        filename=ERRORS_FILENAME,
        subdirectory=assessment_subdirectory(assessment.id),
    )
    logger.info(
        "[build_and_upload_errors] Errors dump %s | execution_id=%s | rows=%s | url=%s",
        "uploaded" if url else "upload failed",
        execution.id,
        len(rows),
        url,
    )
    return url


def finalize_result_files(
    *,
    session: Session,
    execution: AssessmentRun,
    assessment: Assessment,
    bag: BatchRunState,
    failure_message: str | None = None,
) -> None:
    """Persist every stage dump plus the run's errors.jsonl at terminal time.

    Idempotent (a re-merge replaces only its own kind), so a redelivered tick is
    harmless. Never raises: durability is best-effort, terminating the run is not.
    """
    try:
        files: dict[str, dict[str, Any]] = {}
        for stage, url in (bag.get("stage_output_urls") or {}).items():
            if not url:
                continue
            files[stage_file_kind(stage)] = {"object_store_url": url}

        errors_url = build_and_upload_errors(
            session=session,
            execution=execution,
            assessment=assessment,
            bag=bag,
            failure_message=failure_message,
        )
        if errors_url:
            files[ERRORS_FILE_KIND] = {"object_store_url": errors_url}

        if files:
            api.set_result_files(session=session, assessment=assessment, files=files)
    except Exception:
        logger.error(
            "[finalize_result_files] Could not persist result files | "
            "assessment_id=%s | execution_id=%s",
            assessment.id,
            execution.id,
            exc_info=True,
        )


def build_callback_metadata(
    *, session: Session, assessment: Assessment
) -> dict[str, Any]:
    """Presign every recorded result file for the callback envelope's ``metadata``.

    Always returns ``{"result_files": ..., "expires_at": ...}``; a per-key presign
    failure drops that entry rather than the whole envelope key.
    """
    expires_at = (now() + timedelta(seconds=SIGNED_URL_EXPIRY_SECONDS)).isoformat()
    signed: dict[str, dict[str, Any]] = {}

    try:
        storage = get_cloud_storage(session=session, project_id=assessment.project_id)
    except Exception:
        logger.error(
            "[build_callback_metadata] Storage unavailable, sending empty result_files | "
            "assessment_id=%s",
            assessment.id,
            exc_info=True,
        )
        return {"result_files": signed, "expires_at": expires_at}

    for kind, record in assessment.result_files.items():
        entry: dict[str, Any] = record or {}
        object_store_url = entry.get("object_store_url")
        if not object_store_url:
            continue
        try:
            signed_url = storage.get_signed_url(
                object_store_url, expires_in=SIGNED_URL_EXPIRY_SECONDS
            )
        except Exception:
            logger.error(
                "[build_callback_metadata] Presign failed, dropping kind | "
                "assessment_id=%s | kind=%s",
                assessment.id,
                kind,
                exc_info=True,
            )
            continue
        signed[kind] = {"signed_url": signed_url}

    return {"result_files": signed, "expires_at": expires_at}
