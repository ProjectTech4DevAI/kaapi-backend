"""Tests for the submission-rows round trip (api/submission_store.py)."""

import io
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.models.assessment import Assessment, AssessmentMethod, BatchInput
from app.services.assessment.api.submission_store import (
    SubmissionUnavailableError,
    load_submission_rows,
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


class TestLoad:
    def test_parses_the_stored_jsonl(self) -> None:
        storage = MagicMock()
        storage.stream.return_value = io.BytesIO(b'{"a": "1"}\n\n{"a": "2"}\n')
        with patch(_STORAGE, return_value=storage):
            result = load_submission_rows(
                session=MagicMock(), assessment=_assessment("s3://b/sub.jsonl")
            )

        assert result.data == [{"a": "1"}, {"a": "2"}]

    def test_missing_url_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="No submission_input"):
            load_submission_rows(session=MagicMock(), assessment=_assessment(None))

    def test_storage_failure_is_retryable(self) -> None:
        storage = MagicMock()
        storage.stream.side_effect = RuntimeError("s3 down")
        with patch(_STORAGE, return_value=storage):
            with pytest.raises(SubmissionUnavailableError):
                load_submission_rows(
                    session=MagicMock(), assessment=_assessment("s3://b/sub.jsonl")
                )
