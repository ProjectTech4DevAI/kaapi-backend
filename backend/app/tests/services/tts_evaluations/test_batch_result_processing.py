import asyncio
import base64
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.core.audio_utils import pcm_to_wav
from app.core.batch import BatchJobState
from app.crud.tts_evaluations.cron import poll_tts_run
from app.models import EvaluationRun
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.batch_result_processing import (
    execute_tts_result_processing,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_batch_job,
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
    use_test_session,
)

MODULE = "app.services.tts_evaluations.batch_result_processing"
GEMINI = "gemini-3.8-flash-tts"
SARVAM = "bulbul:v3"
# 24 kHz / 16-bit / mono: 48000 bytes is one second; the WAV adds a 44-byte header.
ONE_SECOND_PCM_B64 = base64.b64encode(b"\x00\x01" * 24000).decode()
ONE_SECOND_WAV_B64 = base64.b64encode(pcm_to_wav(b"\x00\x01" * 24000)).decode()


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


class _TimesOutAfterFirst(list):
    """Batch output whose iteration is interrupted after the first row.

    The timeout is raised between rows rather than from a mocked call because
    SoftTimeLimitExceeded is an Exception and the per-row handler would swallow it.
    """

    def __init__(self, items: list[dict[str, Any]], timeout: BaseException) -> None:
        super().__init__(items)
        self._timeout = timeout

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield self[0]
        raise self._timeout


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

    def test_wav_response_is_not_double_wrapped(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        row = create_test_tts_result_row(db, run=run)
        boundaries.download.return_value = [_ok(row.id, data=ONE_SECOND_WAV_B64)]

        _execute(run)

        _refresh(db, row)
        assert row.status == JobStatus.SUCCESS.value
        # One 44-byte header: the model's WAV is unwrapped before re-wrapping.
        assert row.metadata_ == {"duration_seconds": 1.0, "size_bytes": 48044}
        uploaded = boundaries.upload.call_args.kwargs["content"]
        assert uploaded.count(b"RIFF") == 1

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

    @pytest.mark.parametrize(
        ("usage_metadata", "expected"),
        [
            (
                {
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 340,
                        "totalTokenCount": 352,
                    }
                },
                {
                    "unit": "tokens",
                    "input_tokens": 12,
                    "output_tokens": 340,
                    "total_tokens": 352,
                },
            ),
            (
                {
                    "usage_metadata": {
                        "prompt_token_count": 0,
                        "candidates_token_count": 340,
                        "total_token_count": 340,
                    }
                },
                {
                    "unit": "tokens",
                    "input_tokens": 0,
                    "output_tokens": 340,
                    "total_tokens": 340,
                },
            ),
            # A camelCase 0 must not fall through to the (absent) snake_case key.
            (
                {
                    "usageMetadata": {
                        "promptTokenCount": 0,
                        "candidatesTokenCount": 340,
                        "totalTokenCount": 340,
                    }
                },
                {
                    "unit": "tokens",
                    "input_tokens": 0,
                    "output_tokens": 340,
                    "total_tokens": 340,
                },
            ),
            ({}, None),
        ],
        ids=["camel_case", "snake_case", "camel_case_zero", "missing"],
    )
    def test_success_records_token_usage(
        self,
        db: Session,
        run: EvaluationRun,
        boundaries: Boundaries,
        usage_metadata: dict[str, Any],
        expected: dict[str, Any] | None,
    ) -> None:
        row = create_test_tts_result_row(db, run=run)
        response = {**_audio_response(), **usage_metadata}
        boundaries.download.return_value = [
            {"custom_id": str(row.id), "response": response}
        ]

        _execute(run)

        _refresh(db, row)
        assert row.status == JobStatus.SUCCESS.value
        assert row.usage == expected

    def test_usage_kept_when_audio_processing_fails(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        row = create_test_tts_result_row(db, run=run)
        response = {
            **_audio_response(data="###"),
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 7,
                "totalTokenCount": 12,
            },
        }
        boundaries.download.return_value = [
            {"custom_id": str(row.id), "response": response}
        ]

        _execute(run)

        _refresh(db, row)
        assert row.status == JobStatus.FAILED.value
        assert row.usage == {
            "unit": "tokens",
            "input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": 12,
        }

    def test_row_missing_from_batch_output_fails(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        # Regression: an omitted row stayed PENDING and the cron re-dispatched forever.
        covered = create_test_tts_result_row(db, run=run)
        omitted = create_test_tts_result_row(db, run=run)
        boundaries.download.return_value = [_ok(covered.id)]

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (1, 1)
        _refresh(db, covered, omitted)
        assert covered.status == JobStatus.SUCCESS.value
        assert (omitted.status, omitted.error_message) == (
            JobStatus.FAILED.value,
            "No result in batch output",
        )

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

    @pytest.mark.parametrize(
        "timeout", [Timeout(), SoftTimeLimitExceeded()], ids=["gevent", "soft"]
    )
    def test_timeout_mid_loop_keeps_rows_already_processed(
        self,
        db: Session,
        run: EvaluationRun,
        boundaries: Boundaries,
        timeout: BaseException,
    ) -> None:
        # Regression: rows uploaded before the timeout were lost and failed too.
        processed, unprocessed = (
            create_test_tts_result_row(db, run=run) for _ in range(2)
        )
        processed_id, unprocessed_id = processed.id, unprocessed.id
        output = _TimesOutAfterFirst([_ok(processed_id), _ok(unprocessed_id)], timeout)

        def download(_batch_id: str) -> list[dict[str, Any]]:
            # In production the prefetch session has closed by now, detaching the
            # rows; without this the shared test session autoflushes them anyway.
            db.expunge_all()
            return output

        boundaries.download.side_effect = download

        with pytest.raises(type(timeout)):
            _execute(run)

        db.expunge_all()
        processed = db.get(TTSResult, processed_id)
        unprocessed = db.get(TTSResult, unprocessed_id)
        assert processed.status == JobStatus.SUCCESS.value
        assert processed.object_store_url.startswith(
            "s3://bucket/evaluations/tts/audio/"
        )
        assert (unprocessed.status, unprocessed.error_message) == (
            JobStatus.FAILED.value,
            "Result processing exceeded soft time limit",
        )

    def test_write_back_skips_row_failed_while_processing(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        # The cron's dead-worker sweep can fail a claimed row mid-task; the late
        # write must not revive it.
        swept, claimed = (create_test_tts_result_row(db, run=run) for _ in range(2))
        swept_id, claimed_id = swept.id, claimed.id

        def download(_batch_id: str) -> list[dict[str, Any]]:
            db.expunge_all()
            swept_row = db.get(TTSResult, swept_id)
            swept_row.status = JobStatus.FAILED.value
            swept_row.error_message = "Result processing did not complete"
            db.add(swept_row)
            db.commit()
            db.expunge_all()
            return [_ok(swept_id), _ok(claimed_id)]

        boundaries.download.side_effect = download

        _execute(run)

        db.expunge_all()
        kept = db.get(TTSResult, swept_id)
        written = db.get(TTSResult, claimed_id)
        assert (kept.status, kept.object_store_url, kept.error_message) == (
            JobStatus.FAILED.value,
            None,
            "Result processing did not complete",
        )
        assert written.status == JobStatus.SUCCESS.value

    def test_nothing_to_claim_exits_without_touching_gemini(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        # A second task for a batch another task is already processing.
        held = create_test_tts_result_row(db, run=run, status=JobStatus.PROCESSING)

        outcome = _execute(run)

        assert (outcome["success"], outcome["processed"], outcome["failed"]) == (
            True,
            0,
            0,
        )
        boundaries.gemini_client.from_credentials.assert_not_called()
        boundaries.download.assert_not_called()
        _refresh(db, held)
        assert held.status == JobStatus.PROCESSING.value

    def test_cron_tick_during_processing_does_not_redispatch_batch(
        self, db: Session, run: EvaluationRun, boundaries: Boundaries
    ) -> None:
        # Regression: rows stayed PENDING while a task processed them, so every cron
        # tick re-dispatched the succeeded batch and it was processed twice.
        create_test_tts_batch_job(
            db, run=run, model=GEMINI, provider_status=BatchJobState.SUCCEEDED.value
        )
        row = create_test_tts_result_row(db, run=run)
        run_id, org_id, row_id = run.id, run.organization_id, row.id

        def download_while_cron_ticks(_batch_id: str) -> list[dict[str, Any]]:
            db.expunge_all()
            asyncio.run(
                poll_tts_run(
                    session=db, run=db.get(EvaluationRun, run_id), org_id=org_id
                )
            )
            db.expunge_all()
            return [_ok(row_id)]

        boundaries.download.side_effect = download_while_cron_ticks

        # Celery enqueue and the cron's Gemini client are the cron's external boundaries.
        with (
            patch("app.crud.tts_evaluations.cron.GeminiClient"),
            patch(
                "app.crud.tts_evaluations.cron.start_tts_result_processing"
            ) as redispatch,
        ):
            _execute(run)

        redispatch.assert_not_called()
        db.expunge_all()
        assert db.get(TTSResult, row_id).status == JobStatus.SUCCESS.value

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
