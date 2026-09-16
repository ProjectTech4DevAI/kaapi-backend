"""Object-store round trip for the rows a BATCH assessment runs against."""

import json
import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from uuid import UUID

from botocore.response import StreamingBody
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import upload_jsonl_to_object_store
from app.models.assessment import (
    Assessment,
    AssessmentSubmission,
    BatchInput,
    Submission,
)
from app.services.assessment.validators import (
    SUBMISSION_FILENAME,
    assessment_prefix,
    submission_rows_url,
)

logger = logging.getLogger(__name__)


class SubmissionUnavailableError(Exception):
    """The stored submission rows could not be read; the caller should retry the task."""


def upload_submission_rows(
    *, session: Session, assessment_id: UUID, project_id: int, batch_input: BatchInput
) -> str | None:
    """Store inline submission rows as JSONL. Returns the object-store url, None on failure."""
    rows = batch_input.data or []
    url = upload_jsonl_to_object_store(
        storage=get_cloud_storage(session=session, project_id=project_id),
        results=rows,
        filename=SUBMISSION_FILENAME,
        subdirectory=assessment_prefix(assessment_id),
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
        raise SubmissionUnavailableError(f"Read of {url} failed: {exc}") from exc


@contextmanager
def _stream(
    *, session: Session, project_id: int, url: str
) -> Generator[Iterator[Submission], None, None]:
    """Stream a JSONL of rows; the storage body closes on exit."""
    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        body = storage.stream(url)
    except Exception as exc:
        raise SubmissionUnavailableError(f"Could not open {url}: {exc}") from exc

    try:
        yield _rows(body, url)
    finally:
        body.close()


@contextmanager
def open_uploaded_rows(
    *, session: Session, submission: AssessmentSubmission
) -> Generator[Iterator[Submission], None, None]:
    """Stream an uploaded submission's rows, parsed once at upload.

    Falls back to parsing the original file for submissions stored before the
    upload started writing them, so old records keep working.
    """
    if not submission.object_store_url:
        raise ValueError(f"Submission {submission.id} has no object_store_url")

    url = submission_rows_url(submission.object_store_url)
    # Opened here, not via `_stream`: only a failure to open may fall back, never a
    # failure mid-iteration, which must surface as the retryable error it is.
    try:
        storage = get_cloud_storage(session=session, project_id=submission.project_id)
        body = storage.stream(url)
    except Exception:
        logger.info(
            "[open_uploaded_rows] No stored rows, parsing the original | submission_id=%s",
            submission.id,
        )
        from app.crud.assessment.batch import load_submission_file_rows

        yield iter(load_submission_file_rows(session=session, submission=submission))
        return

    try:
        yield _rows(body, url)
    finally:
        body.close()


@contextmanager
def open_submission_rows(
    *, session: Session, assessment: Assessment
) -> Generator[Iterator[Submission], None, None]:
    """Stream the rows an assessment runs against.

    An inline BATCH holds its own copy at ``submission_input``; one submitted by
    reference reads the uploaded submission directly, since that file never changes.
    """
    if assessment.submission_input:
        with _stream(
            session=session,
            project_id=assessment.project_id,
            url=assessment.submission_input,
        ) as stream:
            yield stream
        return

    submission = (
        session.get(AssessmentSubmission, assessment.submission_id)
        if assessment.submission_id
        else None
    )
    if submission is None:
        raise ValueError(f"Assessment {assessment.id} has no rows to read")

    with open_uploaded_rows(session=session, submission=submission) as stream:
        yield stream
