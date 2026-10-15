"""Tests for the submission-rows round trip (api/submission_store.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.models.assessment import Assessment, AssessmentMethod, BatchInput
from app.services.assessment.api.submission_store import (
    SubmissionUnavailableError,
    open_submission_rows,
    upload_submission_rows,
)

_STORAGE = "app.services.assessment.api.submission_store.get_cloud_storage"
_UPLOAD = "app.services.assessment.api.submission_store.upload_jsonl_to_object_store"


def _assessment(url: str | None) -> Assessment:
    return Assessment(
        id=uuid4(),
        method=AssessmentMethod.BATCH,
        submission_input=url,
        organization_id=1,
        project_id=1,
    )


class _Body:
    """The slice of botocore's StreamingBody the loader touches."""

    def __init__(self, payload: bytes, fail_after: int | None = None) -> None:
        self._lines = payload.splitlines()
        self._fail_after = fail_after
        self.closed = False

    def iter_lines(self):
        for index, line in enumerate(self._lines):
            if self._fail_after is not None and index >= self._fail_after:
                raise OSError("connection reset")
            yield line

    def close(self) -> None:
        self.closed = True


def _storage_with(body: _Body) -> MagicMock:
    storage = MagicMock()
    storage.stream.return_value = body
    return storage


class TestUpload:
    def test_writes_rows_under_the_assessment_prefix(self) -> None:
        assessment_id = uuid4()
        with patch(_STORAGE), patch(_UPLOAD, return_value="s3://b/sub.jsonl") as upload:
            url = upload_submission_rows(
                session=MagicMock(),
                assessment_id=assessment_id,
                project_id=1,
                batch_input=BatchInput(data=[{"a": "1"}]),
            )

        assert url == "s3://b/sub.jsonl"
        assert upload.call_args.kwargs["filename"] == "submission.jsonl"
        assert upload.call_args.kwargs["subdirectory"] == f"assessment/{assessment_id}"

    def test_returns_none_when_the_upload_fails(self) -> None:
        with patch(_STORAGE), patch(_UPLOAD, return_value=None):
            assert (
                upload_submission_rows(
                    session=MagicMock(),
                    assessment_id=uuid4(),
                    project_id=1,
                    batch_input=BatchInput(data=[{"a": "1"}]),
                )
                is None
            )


class TestOpen:
    def test_streams_the_stored_jsonl_and_closes_the_body(self) -> None:
        body = _Body(b'{"a": "1"}\n\n{"a": "2"}\n')
        with patch(_STORAGE, return_value=_storage_with(body)):
            with open_submission_rows(
                session=MagicMock(), assessment=_assessment("s3://b/sub.jsonl")
            ) as stream:
                assert list(stream) == [{"a": "1"}, {"a": "2"}]

        assert body.closed

    def test_no_rows_anywhere_is_a_value_error(self) -> None:
        session = MagicMock()
        session.get.return_value = None
        with pytest.raises(ValueError, match="no rows to read"):
            with open_submission_rows(session=session, assessment=_assessment(None)):
                pass

    def test_falls_back_to_the_uploaded_submission(self) -> None:
        body = _Body(b'{"a": "1"}\n')
        assessment = _assessment(None)
        assessment.submission_id = uuid4()
        submission = SimpleNamespace(
            id=assessment.submission_id,
            project_id=1,
            object_store_url="s3://b/subs/x/file.csv",
        )
        session = MagicMock()
        session.get.return_value = submission

        with patch(_STORAGE, return_value=_storage_with(body)):
            with open_submission_rows(session=session, assessment=assessment) as stream:
                assert list(stream) == [{"a": "1"}]

        assert body.closed

    def test_open_failure_is_retryable(self) -> None:
        storage = MagicMock()
        storage.stream.side_effect = RuntimeError("s3 down")
        with patch(_STORAGE, return_value=storage):
            with pytest.raises(SubmissionUnavailableError):
                with open_submission_rows(
                    session=MagicMock(), assessment=_assessment("s3://b/sub.jsonl")
                ):
                    pass

    def test_read_failure_mid_stream_is_retryable_and_still_closes(self) -> None:
        body = _Body(b'{"a": "1"}\n{"a": "2"}\n', fail_after=1)
        with patch(_STORAGE, return_value=_storage_with(body)):
            with pytest.raises(SubmissionUnavailableError):
                with open_submission_rows(
                    session=MagicMock(), assessment=_assessment("s3://b/sub.jsonl")
                ) as stream:
                    list(stream)

        assert body.closed

    def test_corrupt_line_is_terminal(self) -> None:
        body = _Body(b'{"a": "1"}\nnot json\n')
        with patch(_STORAGE, return_value=_storage_with(body)):
            with pytest.raises(ValueError):
                with open_submission_rows(
                    session=MagicMock(), assessment=_assessment("s3://b/sub.jsonl")
                ) as stream:
                    list(stream)
