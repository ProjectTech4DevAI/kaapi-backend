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
from app.services.tts_evaluations.sync_generation import execute_tts_sync_chunk
from app.services.tts_evaluations.synthesizers import (
    SynthesizedAudio,
    TTSSynthesisError,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
    use_test_session,
)

MODULE = "app.services.tts_evaluations.sync_generation"
SARVAM_MODEL = "bulbul:v3"
FAIL_TEXT = "please fail"
CRASH_TEXT = "please crash"


@dataclass
class Boundaries:
    credentials: MagicMock
    storage: MagicMock
    upload: MagicMock
    create_client: MagicMock
    synthesize: MagicMock


def _fake_synthesize(*, text: str, **_kwargs: Any) -> SynthesizedAudio:
    if text == FAIL_TEXT:
        raise TTSSynthesisError(
            "[SARVAM] TTS request failed (code: 400): bad", retryable=False
        )
    if text == CRASH_TEXT:
        raise KeyError("audios")
    return SynthesizedAudio(wav_bytes=b"w" * 100, duration_seconds=1.23456)


def _fake_upload(*, subdirectory: str, filename: str, **_kwargs: Any) -> str:
    return f"s3://bucket/{subdirectory}/{filename}"


@pytest.fixture
def boundaries(db: Session) -> Iterator[Boundaries]:
    with (
        use_test_session(MODULE, db),
        patch(
            f"{MODULE}.get_provider_credential", return_value={"api_key": "k"}
        ) as creds,
        patch(f"{MODULE}.get_cloud_storage") as storage,
        patch(f"{MODULE}.upload_to_object_store", side_effect=_fake_upload) as upload,
        patch(f"{MODULE}.create_tts_client", return_value=MagicMock()) as create_client,
        patch(f"{MODULE}.synthesize_tts", side_effect=_fake_synthesize) as synthesize,
    ):
        yield Boundaries(creds, storage, upload, create_client, synthesize)


@pytest.fixture
def run(db: Session, user_api_key: TestAuthContext) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[SARVAM_MODEL],
    )


def _pending(db: Session, run: EvaluationRun, text: str | None = None) -> TTSResult:
    return create_test_tts_result_row(
        db, run=run, provider=SARVAM_MODEL, sample_text=text
    )


def _execute(run: EvaluationRun, result_ids: list[int]) -> dict[str, Any]:
    return execute_tts_sync_chunk(
        project_id=run.project_id,
        job_id=str(run.id),
        task_id="celery-task",
        task_instance=MagicMock(),
        organization_id=run.organization_id,
        model=SARVAM_MODEL,
        result_ids=result_ids,
        language_code="hi-IN",
    )


def _reload(db: Session, *rows: TTSResult | EvaluationRun) -> None:
    for row in rows:
        db.refresh(row)


@pytest.mark.usefixtures("boundaries")
class TestExecuteTTSSyncChunk:
    def test_success_writes_audio_and_completes_run(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        first = _pending(db, run)
        second = _pending(db, run)

        outcome = _execute(run, [first.id, second.id])

        assert outcome == {
            "success": True,
            "run_id": run.id,
            "processed": 2,
            "failed": 0,
            "run_status": "completed",
        }
        _reload(db, first, second, run)
        for row in (first, second):
            assert row.status == JobStatus.SUCCESS.value
            assert row.object_store_url.startswith("s3://bucket/evaluations/tts/audio/")
            assert row.object_store_url.endswith(".wav")
            assert row.metadata_ == {"duration_seconds": 1.235, "size_bytes": 100}
            assert row.error_message is None
        assert run.status == "completed"
        assert run.error_message is None
        assert boundaries.synthesize.call_args.kwargs["language_code"] == "hi-IN"
        assert boundaries.synthesize.call_args.kwargs["voice"] == "shubh"

    def test_other_pending_rows_keep_run_processing(
        self, db: Session, run: EvaluationRun
    ) -> None:
        mine = _pending(db, run)
        _pending(db, run)  # belongs to a different chunk

        outcome = _execute(run, [mine.id])

        assert outcome["run_status"] == "processing"
        _reload(db, run)
        assert run.status == "processing"

    def test_redelivered_chunk_skips_finished_rows(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        done = create_test_tts_result_row(
            db,
            run=run,
            provider=SARVAM_MODEL,
            status=JobStatus.SUCCESS,
            object_store_url="s3://bucket/original.wav",
        )
        pending = _pending(db, run)

        outcome = _execute(run, [done.id, pending.id])

        assert outcome["processed"] == 1
        assert boundaries.synthesize.call_count == 1
        _reload(db, done)
        assert done.object_store_url == "s3://bucket/original.wav"

    def test_row_finished_concurrently_is_not_counted_or_overwritten(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        row = _pending(db, run)

        def concurrent_writer_wins(**kwargs: Any) -> SynthesizedAudio:
            # A redelivered copy of this chunk lands its write first. The main
            # thread is parked in as_completed, so the session isn't shared concurrently.
            db.refresh(row)
            row.status = JobStatus.SUCCESS.value
            row.object_store_url = "s3://bucket/other-worker.wav"
            db.add(row)
            db.commit()
            return _fake_synthesize(**kwargs)

        boundaries.synthesize.side_effect = concurrent_writer_wins

        outcome = _execute(run, [row.id])

        assert (outcome["processed"], outcome["failed"]) == (0, 0)
        _reload(db, row)
        assert row.object_store_url == "s3://bucket/other-worker.wav"

    def test_no_pending_rows_is_a_noop_that_still_finalizes(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        done = create_test_tts_result_row(
            db, run=run, provider=SARVAM_MODEL, status=JobStatus.SUCCESS
        )

        outcome = _execute(run, [done.id])

        assert outcome == {
            "success": True,
            "run_id": run.id,
            "processed": 0,
            "failed": 0,
            "run_status": "completed",
        }
        boundaries.synthesize.assert_not_called()
        boundaries.credentials.assert_not_called()

    def test_missing_credentials_fail_the_chunk(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.credentials.return_value = None
        rows = [_pending(db, run), _pending(db, run)]

        outcome = _execute(run, [r.id for r in rows])

        assert outcome["success"] is False
        assert outcome["failed"] == 2
        assert outcome["run_status"] == "completed"
        assert "sarvamai credentials are not configured" in outcome["error"]
        _reload(db, run, *rows)
        for row in rows:
            assert row.status == JobStatus.FAILED.value
            assert row.error_message == outcome["error"]
        assert run.error_message == "2 synthesis(es) failed"
        boundaries.synthesize.assert_not_called()

    def test_invalid_credentials_fail_the_chunk(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.create_client.side_effect = ValueError(
            "API Key for SarvamAI Not Set"
        )
        row = _pending(db, run)

        outcome = _execute(run, [row.id])

        assert outcome["success"] is False
        assert outcome["error"].startswith(
            "[KAAPI] Invalid sarvamai credentials: API Key for SarvamAI Not Set."
        )
        _reload(db, row)
        assert row.status == JobStatus.FAILED.value

    def test_synthesis_errors_fail_individual_results(
        self, db: Session, run: EvaluationRun
    ) -> None:
        ok = _pending(db, run)
        rejected = _pending(db, run, text=FAIL_TEXT)
        crashed = _pending(db, run, text=CRASH_TEXT)

        outcome = _execute(run, [ok.id, rejected.id, crashed.id])

        assert (outcome["success"], outcome["processed"], outcome["failed"]) == (
            True,
            1,
            2,
        )
        _reload(db, ok, rejected, crashed, run)
        assert ok.status == JobStatus.SUCCESS.value
        assert rejected.status == JobStatus.FAILED.value
        assert rejected.error_message == "[SARVAM] TTS request failed (code: 400): bad"
        assert crashed.status == JobStatus.FAILED.value
        assert crashed.error_message.startswith(
            "[KAAPI] Unexpected error during sarvamai synthesis: 'audios'."
        )
        assert run.status == "completed"
        assert run.error_message == "2 synthesis(es) failed"

    def test_upload_failure_fails_result(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.upload.side_effect = None
        boundaries.upload.return_value = None
        row = _pending(db, run)

        outcome = _execute(run, [row.id])

        assert (outcome["processed"], outcome["failed"]) == (0, 1)
        _reload(db, row)
        assert row.status == JobStatus.FAILED.value
        assert row.object_store_url is None
        assert row.error_message.startswith(
            "[KAAPI] Audio upload to object store failed."
        )

    def test_gevent_timeout_fails_leftover_rows_and_reraises(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.synthesize.side_effect = Timeout()
        done = create_test_tts_result_row(
            db, run=run, provider=SARVAM_MODEL, status=JobStatus.SUCCESS
        )
        row = _pending(db, run)

        with pytest.raises(Timeout):
            _execute(run, [done.id, row.id])

        _reload(db, done, row, run)
        assert row.status == JobStatus.FAILED.value
        assert row.error_message.startswith("[KAAPI] Synthesis timed out")
        assert done.status == JobStatus.SUCCESS.value
        assert run.status == "completed"

    def test_soft_time_limit_fails_leftover_rows_and_reraises(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.upload.side_effect = SoftTimeLimitExceeded()
        row = _pending(db, run)

        with pytest.raises(SoftTimeLimitExceeded):
            _execute(run, [row.id])

        _reload(db, row, run)
        assert row.status == JobStatus.FAILED.value
        assert row.error_message.startswith("[KAAPI] Synthesis timed out")
        assert run.status == "completed"

    def test_unexpected_exception_fails_leftover_rows(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.upload.side_effect = RuntimeError("s3 exploded")
        row = _pending(db, run)

        outcome = _execute(run, [row.id])

        assert outcome == {
            "success": False,
            "run_id": run.id,
            "error": "s3 exploded",
            "run_status": "completed",
        }
        _reload(db, row)
        assert row.status == JobStatus.FAILED.value
        assert row.error_message.startswith(
            "[KAAPI] Synthesis chunk failed unexpectedly: s3 exploded."
        )

    def test_client_factory_returning_nothing_fails_chunk(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        boundaries.create_client.return_value = None
        row = _pending(db, run)

        outcome = _execute(run, [row.id])

        assert outcome["success"] is False
        assert "missing its client or storage" in outcome["error"]
        _reload(db, row)
        assert row.status == JobStatus.FAILED.value
