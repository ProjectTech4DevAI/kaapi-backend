import base64
import io
import logging
import wave
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError as ElevenLabsApiError
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError as SarvamApiError
from tenacity import wait_none

from app.core.providers import Provider
from app.services.tts_evaluations.synthesizers import (
    SynthesizedAudio,
    TTSSynthesisError,
    create_tts_client,
    synthesize_elevenlabs,
    synthesize_sarvam,
    synthesize_tts,
)


def _wav(pcm: bytes, *, rate: int = 24000, width: int = 2, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _read_wav(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.readframes(w.getnframes()),
        )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _sarvam_client(
    *, audios: list[str] | None = None, error: Exception | None = None
) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.text_to_speech.convert.side_effect = error
    else:
        client.text_to_speech.convert.return_value = SimpleNamespace(
            audios=audios, request_id="req-1"
        )
    return client


def _elevenlabs_client(chunks: Iterator[bytes] | list[bytes]) -> MagicMock:
    client = MagicMock()
    client.text_to_speech.convert.return_value = chunks
    return client


def _call_sarvam(client: MagicMock) -> SynthesizedAudio:
    return synthesize_sarvam(
        client=client,
        text="namaste",
        model="bulbul:v3",
        language_code="hi-IN",
        speaker="shubh",
    )


def _call_elevenlabs(
    client: MagicMock, language_code: str = "hi-IN"
) -> SynthesizedAudio:
    return synthesize_elevenlabs(
        client=client,
        text="namaste",
        model="eleven_v3",
        language_code=language_code,
        voice_id="voice-1",
    )


# 24 kHz / 16-bit / mono: 48000 bytes is exactly one second.
ONE_SECOND_PCM = b"\x01\x00" * 24000


class TestSynthesizeSarvam:
    def test_multi_chunk_wavs_are_joined_under_one_header(self) -> None:
        first = b"\x01\x00" * 100
        second = b"\x02\x00" * 50
        client = _sarvam_client(audios=[_b64(_wav(first)), _b64(_wav(second))])

        audio = _call_sarvam(client)

        channels, width, rate, frames = _read_wav(audio.wav_bytes)
        assert (channels, width, rate) == (1, 2, 24000)
        assert frames == first + second
        # a second RIFF header left inside the PCM would play as a click
        assert audio.wav_bytes.count(b"RIFF") == 1
        assert audio.duration_seconds == pytest.approx(150 / 24000)

    def test_chunk_at_other_sample_rate_is_resampled_to_24k(self) -> None:
        ten_ms_at_16k = b"\x00\x10" * 160
        client = _sarvam_client(audios=[_b64(_wav(ten_ms_at_16k, rate=16000))])

        audio = _call_sarvam(client)

        _, _, rate, frames = _read_wav(audio.wav_bytes)
        assert rate == 24000
        # 10 ms at 24 kHz is 240 frames; ratecv may drop one to rounding.
        assert len(frames) // 2 in (239, 240)
        assert audio.duration_seconds == pytest.approx(0.01, abs=2 / 24000)

    def test_one_second_duration(self) -> None:
        client = _sarvam_client(audios=[_b64(_wav(ONE_SECOND_PCM))])

        assert _call_sarvam(client).duration_seconds == pytest.approx(1.0)

    @pytest.mark.parametrize("audios", [[], None])
    def test_empty_audio_list_is_non_retryable(self, audios: list[str] | None) -> None:
        with pytest.raises(TTSSynthesisError) as exc:
            _call_sarvam(_sarvam_client(audios=audios))

        assert exc.value.retryable is False
        assert exc.value.message.startswith("[SARVAM] TTS response contains no audio")

    @pytest.mark.parametrize(
        "bad_chunk",
        [_b64(b"definitely not a wav file"), "abc"],
        ids=["not-wav", "bad-base64-padding"],
    )
    def test_undecodable_audio_is_kaapi_error(self, bad_chunk: str) -> None:
        with pytest.raises(TTSSynthesisError) as exc:
            _call_sarvam(_sarvam_client(audios=[bad_chunk]))

        assert exc.value.retryable is False
        assert exc.value.message.startswith("[KAAPI] Failed to decode Sarvam audio")

    @pytest.mark.parametrize(
        ("status_code", "retryable", "hint"),
        [
            (400, False, "Review the input text"),
            (401, False, "Verify the API key"),
            (429, True, "Hit the provider's rate/quota"),
            (500, True, "Typically transient"),
            (503, True, "temporarily down"),
            (None, False, "If this persists, contact Kaapi."),
        ],
    )
    def test_api_error_maps_status_to_retryability(
        self, status_code: int | None, retryable: bool, hint: str
    ) -> None:
        error = SarvamApiError(status_code=status_code, body={"detail": "nope"})

        with pytest.raises(TTSSynthesisError) as exc:
            _call_sarvam(_sarvam_client(error=error))

        assert exc.value.retryable is retryable
        assert exc.value.message.startswith(
            f"[SARVAM] TTS request failed (code: {status_code})"
        )
        assert hint in exc.value.message

    def test_server_errors_log_at_error_and_client_errors_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            for status_code in (500, 404):
                with pytest.raises(TTSSynthesisError):
                    _call_sarvam(
                        _sarvam_client(error=SarvamApiError(status_code=status_code))
                    )

        levels = {
            r.message.split("(code: ")[1][:3]: r.levelno
            for r in caplog.records
            if "code: " in r.message
        }
        assert levels == {"500": logging.ERROR, "404": logging.WARNING}

    def test_network_error_is_retryable(self) -> None:
        with pytest.raises(TTSSynthesisError) as exc:
            _call_sarvam(_sarvam_client(error=httpx.ConnectError("dns failure")))

        assert exc.value.retryable is True
        assert (
            "[KAAPI] Could not reach Sarvam (code: ConnectError)" in exc.value.message
        )

    def test_timeout_is_retryable_with_timeout_message(self) -> None:
        with pytest.raises(TTSSynthesisError) as exc:
            _call_sarvam(_sarvam_client(error=httpx.ReadTimeout("slow")))

        assert exc.value.retryable is True
        assert (
            "[KAAPI] TTS request to Sarvam timed out (code: ReadTimeout)"
            in exc.value.message
        )


class TestSynthesizeElevenlabs:
    def test_streamed_pcm_is_wrapped_as_wav(self) -> None:
        client = _elevenlabs_client([ONE_SECOND_PCM[:1000], ONE_SECOND_PCM[1000:]])

        audio = _call_elevenlabs(client)

        channels, width, rate, frames = _read_wav(audio.wav_bytes)
        assert (channels, width, rate) == (1, 2, 24000)
        assert frames == ONE_SECOND_PCM
        assert audio.duration_seconds == pytest.approx(1.0)

    def test_language_hint_sent_for_supported_language(self) -> None:
        client = _elevenlabs_client([ONE_SECOND_PCM])

        _call_elevenlabs(client, language_code="hi-IN")

        kwargs = client.text_to_speech.convert.call_args.kwargs
        assert kwargs["language_code"] == "hi"
        assert kwargs["output_format"] == "pcm_24000"

    def test_language_hint_omitted_for_odia_auto_detect(self) -> None:
        client = _elevenlabs_client([ONE_SECOND_PCM])

        _call_elevenlabs(client, language_code="od-IN")

        assert "language_code" not in client.text_to_speech.convert.call_args.kwargs

    def test_empty_stream_is_non_retryable(self) -> None:
        with pytest.raises(TTSSynthesisError) as exc:
            _call_elevenlabs(_elevenlabs_client([]))

        assert exc.value.retryable is False
        assert exc.value.message.startswith(
            "[ELEVENLABS] TTS response contains no audio"
        )

    @pytest.mark.parametrize(
        ("status_code", "retryable"), [(422, False), (429, True), (502, True)]
    )
    def test_api_error_raised_while_streaming(
        self, status_code: int, retryable: bool
    ) -> None:
        def lazy_stream() -> Iterator[bytes]:
            yield b"\x00\x00"
            raise ElevenLabsApiError(status_code=status_code, body="bad")

        with pytest.raises(TTSSynthesisError) as exc:
            _call_elevenlabs(_elevenlabs_client(lazy_stream()))

        assert exc.value.retryable is retryable
        assert exc.value.message.startswith(
            f"[ELEVENLABS] TTS request failed (code: {status_code})"
        )

    def test_network_error_is_retryable(self) -> None:
        client = MagicMock()
        client.text_to_speech.convert.side_effect = httpx.ConnectTimeout("t")

        with pytest.raises(TTSSynthesisError) as exc:
            _call_elevenlabs(client)

        assert exc.value.retryable is True
        assert "TTS request to Elevenlabs timed out" in exc.value.message


@pytest.fixture
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(synthesize_tts.retry, "wait", wait_none())


def _call_tts(client: MagicMock, provider: Provider) -> SynthesizedAudio:
    return synthesize_tts(
        client=client,
        provider=provider,
        text="hello",
        model="bulbul:v3",
        language_code="en-IN",
        voice="shubh",
    )


@pytest.mark.usefixtures("no_retry_wait")
class TestSynthesizeTTS:
    def test_dispatches_to_sarvam(self) -> None:
        client = _sarvam_client(audios=[_b64(_wav(ONE_SECOND_PCM))])

        audio = _call_tts(client, Provider.SARVAMAI)

        assert audio.duration_seconds == pytest.approx(1.0)
        assert client.text_to_speech.convert.call_args.kwargs["speaker"] == "shubh"

    def test_dispatches_to_elevenlabs(self) -> None:
        client = _elevenlabs_client([ONE_SECOND_PCM])

        audio = _call_tts(client, Provider.ELEVENLABS)

        assert audio.duration_seconds == pytest.approx(1.0)
        assert client.text_to_speech.convert.call_args.kwargs["voice_id"] == "shubh"

    def test_transient_failures_are_retried_until_success(self) -> None:
        client = MagicMock()
        client.text_to_speech.convert.side_effect = [
            SarvamApiError(status_code=503),
            httpx.ConnectError("blip"),
            SimpleNamespace(audios=[_b64(_wav(ONE_SECOND_PCM))], request_id="r"),
        ]

        audio = _call_tts(client, Provider.SARVAMAI)

        assert audio.duration_seconds == pytest.approx(1.0)
        assert client.text_to_speech.convert.call_count == 3

    def test_gives_up_after_three_attempts(self) -> None:
        client = _sarvam_client(error=SarvamApiError(status_code=429))

        with pytest.raises(TTSSynthesisError) as exc:
            _call_tts(client, Provider.SARVAMAI)

        assert exc.value.retryable is True
        assert client.text_to_speech.convert.call_count == 3

    def test_non_retryable_failure_is_not_retried(self) -> None:
        client = _sarvam_client(error=SarvamApiError(status_code=400))

        with pytest.raises(TTSSynthesisError):
            _call_tts(client, Provider.SARVAMAI)

        assert client.text_to_speech.convert.call_count == 1

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(TTSSynthesisError) as exc:
            _call_tts(MagicMock(), Provider.GOOGLE_AISTUDIO)

        assert exc.value.retryable is False
        assert (
            "No synchronous synthesizer for provider 'google-aistudio'"
            in exc.value.message
        )


class TestCreateTTSClient:
    def test_sarvam_client(self) -> None:
        assert isinstance(
            create_tts_client(Provider.SARVAMAI, {"api_key": "k"}), SarvamAI
        )

    def test_elevenlabs_client(self) -> None:
        assert isinstance(
            create_tts_client(Provider.ELEVENLABS, {"api_key": "k"}), ElevenLabs
        )

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API Key"):
            create_tts_client(Provider.SARVAMAI, {})

    def test_batch_only_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="has no synchronous TTS client"):
            create_tts_client(Provider.GOOGLE_AISTUDIO, {"api_key": "k"})
