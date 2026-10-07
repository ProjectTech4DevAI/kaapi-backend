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
SARVAM = "bulbul:v3"
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
        models=[GEMINI, SARVAM],
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
    def test_success_stores_wav_without_finalizing_run(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries, key: str
    ) -> None:
        rows = [create_test_tts_result_row(db, run=run) for _ in range(2)]
        boundaries.download.return_value = [_ok(r.id, key=key) for r in rows]

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (2, 0)
        _refresh(db, run, *rows)
        for row in rows:
            assert row.status == JobStatus.SUCCESS.value
            assert row.object_store_url.startswith("s3://bucket/evaluations/tts/audio/")
            assert row.metadata_ == {"duration_seconds": 1.0, "size_bytes": 48044}
        assert run.status == "pending"

    def test_only_this_models_pending_rows_are_processed(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        done = create_test_tts_result_row(
            db,
            run=run,
            status=JobStatus.SUCCESS,
            object_store_url="s3://bucket/keep.wav",
        )
        other_model = create_test_tts_result_row(db, run=run, provider=SARVAM)
        pending = create_test_tts_result_row(db, run=run)
        boundaries.download.return_value = [
            _ok(done.id),
            _ok(other_model.id),
            _ok(pending.id),
        ]

        _execute(run)

        _refresh(db, done, other_model, pending)
        assert done.object_store_url == "s3://bucket/keep.wav"
        assert other_model.status == JobStatus.PENDING.value
        assert pending.status == JobStatus.SUCCESS.value

    def test_bad_results_fail_their_row_and_others_still_succeed(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        errored, no_error_text, silent, bad_padding, not_base64, odd_shape, ok = (
            create_test_tts_result_row(db, run=run) for _ in range(7)
        )
        boundaries.download.return_value = [
            {"custom_id": "not-a-number", "response": _audio_response()},
            {"custom_id": str(errored.id), "response": None, "error": "SAFETY block"},
            {"custom_id": str(no_error_text.id)},
            {
                "custom_id": str(silent.id),
                "response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]},
            },
            _ok(bad_padding.id, data="abc"),
            # Without strict decoding this became b"" and was stored as a 0s success.
            _ok(not_base64.id, data="###"),
            {"custom_id": str(odd_shape.id), "response": ["not", "a", "dict"]},
            _ok(ok.id),
        ]

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (1, 7)
        _refresh(
            db, errored, no_error_text, silent, bad_padding, not_base64, odd_shape, ok
        )
        assert ok.status == JobStatus.SUCCESS.value
        assert errored.error_message == "SAFETY block"
        assert no_error_text.error_message == "Unknown error"
        assert silent.error_message == "No audio data in response"
        for row in (bad_padding, not_base64, odd_shape):
            assert row.error_message.startswith("Audio processing failed:")
        for row in (errored, no_error_text, silent, bad_padding, not_base64, odd_shape):
            assert (row.status, row.object_store_url) == (JobStatus.FAILED.value, None)

    def test_upload_failure_fails_row(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.upload.side_effect = None
        boundaries.upload.return_value = None
        row = create_test_tts_result_row(db, run=run)
        boundaries.download.return_value = [_ok(row.id)]

        _execute(run)

        _refresh(db, row)
        assert (row.status, row.error_message) == (
            JobStatus.FAILED.value,
            "Audio upload to object store failed",
        )

    @pytest.mark.parametrize(
        "timeout", [Timeout(), SoftTimeLimitExceeded()], ids=["gevent", "soft"]
    )
    def test_timeout_fails_only_this_models_pending_rows_and_reraises(
        self,
        db: Session,
        run: EvaluationRun,
        boundaries: Boundaries,
        timeout: BaseException,
    ) -> None:
        boundaries.download.side_effect = timeout
        gemini_row = create_test_tts_result_row(db, run=run)
        sarvam_row = create_test_tts_result_row(db, run=run, provider=SARVAM)

        with pytest.raises(type(timeout)):
            _execute(run)

        _refresh(db, run, gemini_row, sarvam_row)
        assert (gemini_row.status, gemini_row.error_message) == (
            JobStatus.FAILED.value,
            "Result processing exceeded soft time limit",
        )
        assert sarvam_row.status == JobStatus.PENDING.value
        assert run.status == "pending"

    def test_unexpected_error_fails_only_this_models_pending_rows(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.gemini_client.from_credentials.side_effect = ValueError(
            "no gemini key"
        )
        gemini_row = create_test_tts_result_row(db, run=run)
        sarvam_row = create_test_tts_result_row(db, run=run, provider=SARVAM)

        outcome = _execute(run)

        assert outcome == {"success": False, "error": "no gemini key"}
        _refresh(db, run, gemini_row, sarvam_row)
        assert (gemini_row.status, gemini_row.error_message) == (
            JobStatus.FAILED.value,
            "Result processing failed: no gemini key",
        )
        assert sarvam_row.status == JobStatus.PENDING.value
        assert run.status == "pending"
