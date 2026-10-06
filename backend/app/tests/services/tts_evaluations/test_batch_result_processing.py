import base64
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.models import EvaluationRun
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.batch_result_processing import (
    execute_tts_result_processing,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
    use_test_session,
)

MODULE = "app.services.tts_evaluations.batch_result_processing"
GEMINI = "gemini-2.5-pro-preview-tts"
# 24 kHz / 16-bit / mono: 48000 bytes is one second; the WAV adds a 44-byte header.
ONE_SECOND_PCM_B64 = base64.b64encode(b"\x00\x01" * 24000).decode()


def _audio_response(
    data: str = ONE_SECOND_PCM_B64, key: str = "inlineData"
) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"parts": [{key: {"mimeType": "audio/L16", "data": data}}]}}
        ]
    }


def _ok(result_id: int | str, **kwargs: Any) -> dict[str, Any]:
    return {"custom_id": str(result_id), "response": _audio_response(**kwargs)}


@dataclass
class Boundaries:
    gemini_client: MagicMock
    download: MagicMock
    upload: MagicMock


def _fake_upload(*, subdirectory: str, filename: str, **_kwargs: Any) -> str:
    return f"s3://bucket/{subdirectory}/{filename}"


@pytest.fixture
def boundaries(db: Session) -> Iterator[Boundaries]:
    with (
        use_test_session(MODULE, db),
        patch(f"{MODULE}.GeminiClient") as gemini_client,
        patch(f"{MODULE}.GeminiBatchProvider") as batch_provider,
        patch(f"{MODULE}.get_cloud_storage"),
        patch(f"{MODULE}.upload_to_object_store", side_effect=_fake_upload) as upload,
    ):
        download = batch_provider.return_value.download_batch_results
        download.return_value = []
        yield Boundaries(gemini_client, download, upload)


@pytest.fixture
def run(db: Session, user_api_key: TestAuthContext) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[GEMINI, "bulbul:v3"],
    )


def _execute(run: EvaluationRun) -> dict[str, Any]:
    return execute_tts_result_processing(
        project_id=run.project_id,
        job_id="1",
        task_id="celery-task",
        task_instance=MagicMock(),
        organization_id=run.organization_id,
        evaluation_run_id=run.id,
        tts_provider=GEMINI,
        provider_batch_id="batches/abc",
    )


def _refresh(db: Session, *rows: TTSResult | EvaluationRun) -> None:
    for row in rows:
        db.refresh(row)


@pytest.mark.usefixtures("boundaries")
class TestExecuteTTSResultProcessing:
    @pytest.mark.parametrize("key", ["inlineData", "inline_data"])
    def test_success_uploads_wav_and_completes_run(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries, key: str
    ) -> None:
        rows = [create_test_tts_result_row(db, run=run) for _ in range(2)]
        boundaries.download.return_value = [_ok(r.id, key=key) for r in rows]

        outcome = _execute(run)

        assert outcome == {
            "success": True,
            "run_id": run.id,
            "processed": 2,
            "failed": 0,
            "run_status": "completed",
        }
        _refresh(db, run, *rows)
        for row in rows:
            assert row.status == JobStatus.SUCCESS.value
            assert row.object_store_url.startswith("s3://bucket/evaluations/tts/audio/")
            assert row.metadata_ == {"duration_seconds": 1.0, "size_bytes": 48044}
        assert run.status == "completed"
        boundaries.download.assert_called_once_with("batches/abc")

    def test_non_pending_and_unknown_results_are_skipped(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        done = create_test_tts_result_row(
            db,
            run=run,
            status=JobStatus.SUCCESS,
            object_store_url="s3://bucket/keep.wav",
        )
        other_model = create_test_tts_result_row(db, run=run, provider="bulbul:v3")
        pending = create_test_tts_result_row(db, run=run)
        boundaries.download.return_value = [
            _ok(done.id),
            _ok(other_model.id),
            _ok(pending.id),
        ]

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (1, 0)
        assert outcome["run_status"] == "processing"
        assert boundaries.upload.call_count == 1
        _refresh(db, done, other_model, pending)
        assert done.object_store_url == "s3://bucket/keep.wav"
        assert other_model.status == JobStatus.PENDING.value
        assert pending.status == JobStatus.SUCCESS.value

    def test_invalid_custom_id_is_counted_failed(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.download.return_value = [
            _ok("not-a-number"),
            {"custom_id": None, "response": {}},
        ]

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (0, 2)

    def test_failure_shapes_mark_results_failed(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        errored, no_error_text, silent, bad_b64, odd_shape = (
            create_test_tts_result_row(db, run=run) for _ in range(5)
        )
        boundaries.download.return_value = [
            {"custom_id": str(errored.id), "response": None, "error": "SAFETY block"},
            {"custom_id": str(no_error_text.id)},
            {
                "custom_id": str(silent.id),
                "response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]},
            },
            _ok(bad_b64.id, data="abc"),  # incorrect padding -> binascii.Error
            {"custom_id": str(odd_shape.id), "response": ["not", "a", "dict"]},
        ]

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (0, 5)
        assert outcome["run_status"] == "completed"
        _refresh(db, run, errored, no_error_text, silent, bad_b64, odd_shape)
        assert errored.error_message == "SAFETY block"
        assert no_error_text.error_message == "Unknown error"
        assert silent.error_message == "No audio data in response"
        assert bad_b64.error_message.startswith("Audio processing failed:")
        assert odd_shape.error_message.startswith("Audio processing failed:")
        assert {
            r.status for r in (errored, no_error_text, silent, bad_b64, odd_shape)
        } == {JobStatus.FAILED.value}
        assert run.error_message == "5 synthesis(es) failed"

    def test_non_base64_audio_is_not_stored_as_empty_success(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        row = create_test_tts_result_row(db, run=run)
        boundaries.download.return_value = [_ok(row.id, data="###")]

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (0, 1)
        boundaries.upload.assert_not_called()
        _refresh(db, row)
        assert row.status == JobStatus.FAILED.value
        assert row.error_message.startswith("Audio processing failed:")

    def test_upload_failure_marks_result_failed(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.upload.side_effect = None
        boundaries.upload.return_value = None
        row = create_test_tts_result_row(db, run=run)
        boundaries.download.return_value = [_ok(row.id)]

        outcome = _execute(run)

        assert outcome["failed"] == 1
        _refresh(db, row)
        assert (row.status, row.error_message) == (
            JobStatus.FAILED.value,
            "Audio upload to object store failed",
        )

    def test_empty_batch_finalizes_run(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)

        outcome = _execute(run)

        assert (outcome["processed"], outcome["run_status"]) == (0, "completed")

    def test_timeout_keeps_already_written_chunks(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        rows = [create_test_tts_result_row(db, run=run) for _ in range(60)]
        boundaries.download.return_value = [_ok(r.id) for r in rows]
        uploads = 0

        def upload_then_time_out(**kwargs: Any) -> str:
            nonlocal uploads
            uploads += 1
            if uploads == 55:
                raise Timeout()
            return _fake_upload(**kwargs)

        boundaries.upload.side_effect = upload_then_time_out

        with pytest.raises(Timeout):
            _execute(run)

        _refresh(db, run, *rows)
        statuses = [r.status for r in rows]
        # writes land in slices of 50, so results 51-54 were still buffered
        assert statuses.count(JobStatus.SUCCESS.value) == 50
        assert statuses.count(JobStatus.PENDING.value) == 10
        assert (run.status, run.error_message) == (
            "failed",
            "Task exceeded soft time limit",
        )

    def test_soft_time_limit_fails_run_and_reraises(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.download.side_effect = SoftTimeLimitExceeded()

        with pytest.raises(SoftTimeLimitExceeded):
            _execute(run)

        _refresh(db, run)
        assert run.status == "failed"

    def test_unexpected_error_fails_run(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.gemini_client.from_credentials.side_effect = ValueError(
            "no gemini key"
        )

        outcome = _execute(run)

        assert outcome == {"success": False, "error": "no gemini key"}
        _refresh(db, run)
        assert (run.status, run.error_message) == (
            "failed",
            "Result processing failed: no gemini key",
        )
