import base64
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.core.audio_utils import pcm_to_wav
from app.models import EvaluationRun
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.sync_generation import (
    execute_tts_sync_generation,
    synthesize_elevenlabs,
    synthesize_sarvam,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
    use_test_session,
)

MODULE = "app.services.tts_evaluations.sync_generation"
SARVAM = "bulbul:v3"
ELEVEN = "eleven_v3"
# 24 kHz / 16-bit / mono: 48000 bytes is one second; the WAV adds a 44-byte header.
ONE_SECOND_PCM = b"\x00\x01" * 24000
FAIL_TEXT = "provider rejects this"


def _sarvam_client(audios: list[str]) -> MagicMock:
    client = MagicMock()
    client.text_to_speech.convert.return_value = SimpleNamespace(audios=audios)
    return client


class TestSynthesizeSarvam:
    def test_joins_chunks_without_wav_headers(self) -> None:
        chunks = [base64.b64encode(pcm_to_wav(pcm)).decode() for pcm in (b"ab", b"cd")]
        client = _sarvam_client(chunks)

        pcm = synthesize_sarvam(client, "namaste", SARVAM, "hi-IN")

        assert pcm == b"abcd"
        assert (
            client.text_to_speech.convert.call_args.kwargs["target_language_code"]
            == "hi-IN"
        )

    def test_no_audio_raises(self) -> None:
        with pytest.raises(ValueError, match="Sarvam returned no audio"):
            synthesize_sarvam(_sarvam_client([]), "namaste", SARVAM, "hi-IN")


class TestSynthesizeElevenLabs:
    # od-IN has no ElevenLabs code; None lets ElevenLabs auto-detect.
    @pytest.mark.parametrize(
        ("language_code", "expected"), [("hi-IN", "hi"), ("od-IN", None)]
    )
    def test_maps_language_and_joins_pcm(
        self, language_code: str, expected: str | None
    ) -> None:
        client = MagicMock()
        client.text_to_speech.convert.return_value = iter([b"ab", b"cd"])

        pcm = synthesize_elevenlabs(client, "namaste", ELEVEN, language_code)

        assert pcm == b"abcd"
        sent = client.text_to_speech.convert.call_args.kwargs
        assert (sent["language_code"], sent["output_format"]) == (
            expected,
            "pcm_24000",
        )

    def test_no_audio_raises(self) -> None:
        client = MagicMock()
        client.text_to_speech.convert.return_value = iter([])

        with pytest.raises(ValueError, match="ElevenLabs returned no audio"):
            synthesize_elevenlabs(client, "namaste", ELEVEN, "hi-IN")


def _sarvam_convert(*, text: str, **_kwargs: Any) -> Any:
    if text == FAIL_TEXT:
        raise RuntimeError("bad request")
    return SimpleNamespace(
        audios=[base64.b64encode(pcm_to_wav(ONE_SECOND_PCM)).decode()]
    )


def _fake_upload(*, subdirectory: str, filename: str, **_kwargs: Any) -> str:
    return f"s3://bucket/{subdirectory}/{filename}"


@pytest.fixture
def boundaries(db: Session) -> Iterator[SimpleNamespace]:
    sarvam = MagicMock()
    sarvam.text_to_speech.convert.side_effect = _sarvam_convert
    eleven = MagicMock()
    eleven.text_to_speech.convert.side_effect = lambda **_kw: iter([ONE_SECOND_PCM])
    with (
        use_test_session(MODULE, db),
        patch(f"{MODULE}.SarvamAIProvider.create_client", return_value=sarvam),
        patch(f"{MODULE}.ElevenlabsAIProvider.create_client", return_value=eleven),
        patch(
            f"{MODULE}.get_provider_credential", return_value={"api_key": "k"}
        ) as credentials,
        patch(f"{MODULE}.get_cloud_storage") as storage,
        patch(f"{MODULE}.upload_to_object_store", side_effect=_fake_upload) as upload,
    ):
        yield SimpleNamespace(
            sarvam=sarvam, credentials=credentials, storage=storage, upload=upload
        )


@pytest.fixture
def run(db: Session, user_api_key: TestAuthContext) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[SARVAM, ELEVEN],
    )


def _row(
    db: Session,
    run: EvaluationRun,
    model: str = SARVAM,
    status: JobStatus = JobStatus.PENDING,
    text: str | None = None,
) -> TTSResult:
    return create_test_tts_result_row(
        db, run=run, provider=model, status=status, sample_text=text
    )


def _execute(run: EvaluationRun, model: str = SARVAM) -> dict[str, Any]:
    return execute_tts_sync_generation(
        project_id=run.project_id,
        job_id=str(run.id),
        task_id="celery-task",
        task_instance=MagicMock(),
        organization_id=run.organization_id,
        model=model,
        language_code="hi-IN",
    )


def _refresh(db: Session, *rows: TTSResult | EvaluationRun) -> None:
    for row in rows:
        db.refresh(row)


@pytest.mark.usefixtures("boundaries")
class TestExecuteTTSSyncGeneration:
    @pytest.mark.parametrize("model", [SARVAM, ELEVEN])
    def test_success_stores_audio_without_finalizing_run(
        self, db: Session, run: EvaluationRun, model: str
    ) -> None:
        rows = [_row(db, run, model), _row(db, run, model)]

        outcome = _execute(run, model)

        assert (outcome["success"], outcome["processed"], outcome["failed"]) == (
            True,
            2,
            0,
        )
        _refresh(db, run, *rows)
        for row in rows:
            assert row.status == JobStatus.SUCCESS.value
            assert row.object_store_url.startswith("s3://bucket/evaluations/tts/audio/")
            assert row.metadata_ == {"duration_seconds": 1.0, "size_bytes": 48044}
        assert run.status == "pending"

    def test_only_this_models_pending_rows_are_synthesized(
        self, db: Session, run: EvaluationRun, boundaries: SimpleNamespace
    ) -> None:
        done = _row(db, run, status=JobStatus.SUCCESS)
        done.object_store_url = "s3://bucket/original.wav"
        db.add(done)
        db.commit()
        pending = _row(db, run)
        other_model = _row(db, run, model=ELEVEN)

        _execute(run)

        # A redelivered task must not re-bill the provider for finished rows.
        assert boundaries.sarvam.text_to_speech.convert.call_count == 1
        _refresh(db, done, pending, other_model)
        assert done.object_store_url == "s3://bucket/original.wav"
        assert pending.status == JobStatus.SUCCESS.value
        assert other_model.status == JobStatus.PENDING.value

    def test_synthesis_failure_only_fails_that_row(
        self, db: Session, run: EvaluationRun
    ) -> None:
        ok = _row(db, run)
        rejected = _row(db, run, text=FAIL_TEXT)

        outcome = _execute(run)

        assert (outcome["processed"], outcome["failed"]) == (1, 1)
        _refresh(db, ok, rejected)
        assert ok.status == JobStatus.SUCCESS.value
        assert (rejected.status, rejected.error_message) == (
            JobStatus.FAILED.value,
            "sarvamai synthesis failed: bad request",
        )

    def test_upload_failure_fails_row(
        self, db: Session, run: EvaluationRun, boundaries: SimpleNamespace
    ) -> None:
        boundaries.upload.side_effect = None
        boundaries.upload.return_value = None
        row = _row(db, run)

        _execute(run)

        _refresh(db, row)
        assert (row.status, row.object_store_url, row.error_message) == (
            JobStatus.FAILED.value,
            None,
            "Audio upload to object store failed",
        )

    def test_unsupported_model_fails_its_rows(
        self, db: Session, run: EvaluationRun
    ) -> None:
        row = _row(db, run, model="mystery-tts")
        sarvam_row = _row(db, run)

        outcome = _execute(run, model="mystery-tts")

        assert outcome["success"] is False
        _refresh(db, row, sarvam_row)
        assert (row.status, row.error_message) == (
            JobStatus.FAILED.value,
            "Unsupported sync TTS model: mystery-tts",
        )
        assert sarvam_row.status == JobStatus.PENDING.value

    @pytest.mark.parametrize(
        ("break_boundary", "error"),
        [
            (
                lambda b: setattr(b.credentials, "return_value", None),
                "sarvamai credentials are not configured",
            ),
            (
                lambda b: setattr(b.storage, "side_effect", RuntimeError("no bucket")),
                "no bucket",
            ),
        ],
        ids=["missing_credentials", "crash"],
    )
    def test_setup_failure_fails_every_pending_row(
        self,
        db: Session,
        run: EvaluationRun,
        boundaries: SimpleNamespace,
        break_boundary: Any,
        error: str,
    ) -> None:
        break_boundary(boundaries)
        rows = [_row(db, run), _row(db, run)]

        outcome = _execute(run)

        assert (outcome["success"], outcome["error"]) == (False, error)
        _refresh(db, *rows)
        for row in rows:
            assert (row.status, row.error_message) == (
                JobStatus.FAILED.value,
                f"sarvamai synthesis failed: {error}",
            )

    @pytest.mark.parametrize(
        "timeout", [Timeout(), SoftTimeLimitExceeded()], ids=["gevent", "soft"]
    )
    def test_timeout_fails_leftover_rows_and_reraises(
        self,
        db: Session,
        run: EvaluationRun,
        boundaries: SimpleNamespace,
        timeout: BaseException,
    ) -> None:
        boundaries.storage.side_effect = timeout
        row = _row(db, run)

        with pytest.raises(type(timeout)):
            _execute(run)

        _refresh(db, row)
        assert (row.status, row.error_message) == (
            JobStatus.FAILED.value,
            "Synthesis timed out before this sample was processed",
        )
