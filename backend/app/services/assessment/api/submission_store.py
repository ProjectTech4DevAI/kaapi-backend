"""Object-store round trip for the API-client BATCH submission rows."""

import json
import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from uuid import UUID

from botocore.response import StreamingBody
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import upload_jsonl_to_object_store
from app.models.assessment import Assessment, BatchInput, Submission
from app.services.assessment.api.result_files import assessment_subdirectory

logger = logging.getLogger(__name__)

SUBMISSION_FILENAME = "submission.jsonl"


class SubmissionUnavailableError(Exception):
    """The stored submission rows could not be read; the caller should retry the task."""


def upload_submission_rows(
    *, session: Session, assessment_id: UUID, project_id: int, batch_input: BatchInput
) -> str | None:
    """Store the submission rows as JSONL. Returns the object-store url, None on failure."""
    rows = batch_input.data or []
    url = upload_jsonl_to_object_store(
        storage=get_cloud_storage(session=session, project_id=project_id),
        results=rows,
        filename=SUBMISSION_FILENAME,
        subdirectory=assessment_subdirectory(assessment_id),
    )
    logger.info(
        "[upload_submission_rows] Submission %s | assessment_id=%s | rows=%s | url=%s",
        "stored" if url else "upload failed",
        assessment_id,
        len(rows),
        url,
    )
    return url


def _rows(body: StreamingBody, url: str) -> Iterator[Submission]:
    try:
        for line in body.iter_lines():
            if line.strip():
                yield json.loads(line)
    except json.JSONDecodeError:
        raise
    except Exception as exc:
        raise SubmissionUnavailableError(
            f"[open_submission_rows] Read of {url} failed: {exc}"
        ) from exc


@contextmanager
def open_submission_rows(
    *, session: Session, assessment: Assessment
) -> Generator[Iterator[Submission], None, None]:
    """Stream the stored rows one per line; the storage body closes on exit.

    Storage errors raise ``SubmissionUnavailableError`` so the task retries (an S3 blip
    must not fail a paid-for execution); a corrupt line is a ``ValueError`` and terminal.
    """
    url = assessment.submission_input
    if not url:
        raise ValueError(
            f"[open_submission_rows] No submission_input on assessment {assessment.id}"
        )

    try:
        storage = get_cloud_storage(session=session, project_id=assessment.project_id)
        body = storage.stream(url)
    except Exception as exc:
        raise SubmissionUnavailableError(
            f"[open_submission_rows] Could not open {url}: {exc}"
        ) from exc

    try:
        yield _rows(body, url)
    finally:
        body.close()
