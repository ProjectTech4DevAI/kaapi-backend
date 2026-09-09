"""Object-store round trip for the API-client BATCH submission rows.
"""

import json
import logging
from uuid import UUID

from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import upload_jsonl_to_object_store
from app.models.assessment import Assessment, BatchInput
from app.services.assessment.api.result_files import assessment_subdirectory

logger = logging.getLogger(__name__)

SUBMISSION_FILENAME = "submission.jsonl"


class SubmissionUnavailableError(Exception):
    """The stored submission rows could not be read; the caller should retry the tick."""


def upload_submission_rows(
    *, session: Session, assessment_id: UUID, project_id: int, batch_input: BatchInput
) -> str | None:
    """Store the submission rows as JSONL. Returns the object-store url, None on failure."""
    url = upload_jsonl_to_object_store(
        storage=get_cloud_storage(session=session, project_id=project_id),
        results=batch_input.data,
        filename=SUBMISSION_FILENAME,
        subdirectory=assessment_subdirectory(assessment_id),
    )
    logger.info(
        "[upload_submission_rows] Submission %s | assessment_id=%s | rows=%s | url=%s",
        "stored" if url else "upload failed",
        assessment_id,
        len(batch_input.data),
        url,
    )
    return url


def load_submission_rows(*, session: Session, assessment: Assessment) -> BatchInput:
    """Stream the stored submission rows back.

    A storage read failure raises ``SubmissionUnavailableError`` so the tick retries: an
    S3 blip must not fail a paid-for run.
    """
    url = assessment.submission_input
    if not url:
        raise ValueError(
            f"[load_submission_rows] No submission_input on assessment {assessment.id}"
        )

    try:
        storage = get_cloud_storage(session=session, project_id=assessment.project_id)
        content = storage.stream(url).read().decode("utf-8")
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    except Exception as exc:
        raise SubmissionUnavailableError(
            f"[load_submission_rows] Could not read {url}: {exc}"
        ) from exc

    logger.info(
        "[load_submission_rows] Loaded | assessment_id=%s | rows=%s",
        assessment.id,
        len(rows),
    )
    return BatchInput(data=rows)
