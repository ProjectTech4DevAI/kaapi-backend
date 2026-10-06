"""Synchronous TTS synthesizers for evaluation runs (Sarvam, ElevenLabs).

Every synthesizer returns 24 kHz / 16-bit / mono WAV so clips from different
providers are stored and reviewed in one format. SDK and network failures are
mapped to `TTSSynthesisError`, carrying a source-tagged message and whether the
failure is transient (429 / 5xx / network), which drives the retry policy.
"""

import base64
import io
import logging
import wave
from dataclasses import dataclass
from typing import Any, cast

import httpx
from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError as ElevenLabsApiError
from elevenlabs.core.request_options import (
    RequestOptions as ElevenLabsRequestOptions,
)
from pydub import AudioSegment
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError as SarvamApiError
from sarvamai.core.request_options import RequestOptions as SarvamRequestOptions
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.audio_utils import calculate_duration, pcm_to_wav
from app.core.providers import Provider
from app.services.llm.providers.eleven_ai import ElevenlabsAIProvider
from app.services.llm.providers.sarvam_ai import SarvamAIProvider
from app.services.tts_evaluations.constants import (
    ELEVENLABS_OUTPUT_FORMAT,
    KAAPI_TAG,
    SARVAM_PACE,
    TTS_CHANNELS,
    TTS_RETRYABLE_STATUS_CODES,
    TTS_SAMPLE_RATE_HZ,
    TTS_SAMPLE_WIDTH_BYTES,
    TTS_SYNC_BACKOFF_BASE_SECONDS,
    TTS_SYNC_BACKOFF_MAX_SECONDS,
    TTS_SYNC_MAX_ATTEMPTS,
    TTS_SYNC_REQUEST_TIMEOUT_SECONDS,
)
from app.services.tts_evaluations.language import to_elevenlabs_language_code

logger = logging.getLogger(__name__)

TTSClient = SarvamAI | ElevenLabs

SARVAM_TAG = "SARVAM"
ELEVENLABS_TAG = "ELEVENLABS"

_BITS_PER_BYTE = 8

# Our tenacity policy owns retries; SDK-level retries would multiply attempts.
_SARVAM_REQUEST_OPTIONS = SarvamRequestOptions(
    max_retries=0, timeout_in_seconds=TTS_SYNC_REQUEST_TIMEOUT_SECONDS
)
_ELEVENLABS_REQUEST_OPTIONS = ElevenLabsRequestOptions(
    max_retries=0, timeout_in_seconds=TTS_SYNC_REQUEST_TIMEOUT_SECONDS
)

_STATUS_HINTS: dict[int, str] = {
    400: "Review the input text, model, language and voice — the request shape or content may be invalid.",
    401: "Verify the API key is valid, not expired, and configured for this project.",
    403: "The API key lacks access to the requested model/voice — check plan and key scopes.",
    404: "Verify the model name and voice ID are correct and available on your plan.",
    408: "The provider timed out handling the request — retry in a few seconds.",
    409: "Request conflicts with current resource state — review concurrent requests before retrying.",
    413: "Text exceeds the provider's size limit — shorten the sample.",
    422: "Provider rejected the payload — check input format and parameter values against the API spec.",
    429: "Hit the provider's rate/quota — wait at least 1 min and retry. Request a quota increase or contact Kaapi if persistent.",
    500: "Typically transient — retry in a few seconds. If it persists, contact Kaapi.",
    502: "Provider gateway error — typically transient, retry in a few seconds.",
    503: "Provider temporarily down or overloaded — retry in a few seconds.",
    504: "Provider took too long — retry with a shorter sample.",
}
_DEFAULT_STATUS_HINT = "If this persists, contact Kaapi."


class TTSSynthesisError(Exception):
    """A single synthesis failed; `message` is safe to store on the result row."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class SynthesizedAudio:
    wav_bytes: bytes
    duration_seconds: float


def create_tts_client(provider: Provider, credentials: dict[str, Any]) -> TTSClient:
    """Build the SDK client for a sync TTS provider from stored credentials.

    Raises:
        ValueError: If the credentials lack an API key or the provider isn't sync-capable
    """
    if provider == Provider.SARVAMAI:
        return SarvamAIProvider.create_client(credentials)
    if provider == Provider.ELEVENLABS:
        return ElevenlabsAIProvider.create_client(credentials)
    raise ValueError(f"Provider '{provider.value}' has no synchronous TTS client")


def _api_error(
    *,
    tag: str,
    status_code: int | None,
    body: Any,
    model: str,
    function_name: str,
) -> TTSSynthesisError:
    hint = _STATUS_HINTS.get(status_code or 0, _DEFAULT_STATUS_HINT)
    message = f"[{tag}] TTS request failed (code: {status_code}): {body}. {hint}"
    # 5xx is provider-side (alert-worthy); 4xx is caller's fault (noise if alerted)
    log = logger.error if status_code and status_code >= 500 else logger.warning
    log(
        f"[{function_name}] {message} | provider={tag.lower()}, model={model}",
        exc_info=True,
    )
    return TTSSynthesisError(
        message, retryable=status_code in TTS_RETRYABLE_STATUS_CODES
    )


def _network_error(
    *,
    err: httpx.TransportError,
    tag: str,
    model: str,
    function_name: str,
) -> TTSSynthesisError:
    code = type(err).__name__
    if isinstance(err, httpx.TimeoutException):
        message = (
            f"[{KAAPI_TAG}] TTS request to {tag.title()} timed out (code: {code}). "
            f"Retry with a shorter sample; if persistent, contact Kaapi."
        )
    else:
        message = (
            f"[{KAAPI_TAG}] Could not reach {tag.title()} (code: {code}): {err}. "
            f"Network/DNS issue reaching the provider — check connectivity; if "
            f"persistent, contact Kaapi."
        )
    logger.error(
        f"[{function_name}] {message} | provider={tag.lower()}, model={model}",
        exc_info=True,
    )
    return TTSSynthesisError(message, retryable=True)


def _normalize_pcm(
    pcm_data: bytes, *, channels: int, sample_width: int, frame_rate: int
) -> bytes:
    """Convert PCM to the evaluation format (24 kHz, 16-bit, mono) if it differs."""
    if (
        channels == TTS_CHANNELS
        and sample_width == TTS_SAMPLE_WIDTH_BYTES
        and frame_rate == TTS_SAMPLE_RATE_HZ
    ):
        return pcm_data

    segment = AudioSegment(
        data=pcm_data,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels,
    )
    segment = segment.set_channels(TTS_CHANNELS)
    segment = segment.set_sample_width(TTS_SAMPLE_WIDTH_BYTES)
    segment = segment.set_frame_rate(TTS_SAMPLE_RATE_HZ)
    return bytes(segment.raw_data or b"")


def _wav_chunk_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    return _normalize_pcm(
        frames, channels=channels, sample_width=sample_width, frame_rate=frame_rate
    )


def _to_synthesized_audio(pcm_data: bytes) -> SynthesizedAudio:
    wav_bytes = pcm_to_wav(
        pcm_data,
        sample_rate=TTS_SAMPLE_RATE_HZ,
        bits_per_sample=TTS_SAMPLE_WIDTH_BYTES * _BITS_PER_BYTE,
        channels=TTS_CHANNELS,
    )
    duration = calculate_duration(
        len(pcm_data),
        sample_rate=TTS_SAMPLE_RATE_HZ,
        bits_per_sample=TTS_SAMPLE_WIDTH_BYTES * _BITS_PER_BYTE,
        channels=TTS_CHANNELS,
    )
    return SynthesizedAudio(wav_bytes=wav_bytes, duration_seconds=duration)


def synthesize_sarvam(
    *,
    client: SarvamAI,
    text: str,
    model: str,
    language_code: str,
    speaker: str,
) -> SynthesizedAudio:
    """Synthesize one text with Sarvam and return it as normalized WAV.

    Raises:
        TTSSynthesisError: On any provider, network, or decoding failure
    """
    try:
        response = client.text_to_speech.convert(
            text=text,
            model=model,
            target_language_code=language_code,
            speaker=speaker,
            pace=SARVAM_PACE,
            speech_sample_rate=TTS_SAMPLE_RATE_HZ,
            request_options=_SARVAM_REQUEST_OPTIONS,
        )
    except SarvamApiError as e:
        raise _api_error(
            tag=SARVAM_TAG,
            status_code=e.status_code,
            body=e.body,
            model=model,
            function_name="synthesize_sarvam",
        ) from e
    except httpx.TransportError as e:
        raise _network_error(
            err=e, tag=SARVAM_TAG, model=model, function_name="synthesize_sarvam"
        ) from e

    if not response.audios:
        message = (
            f"[{SARVAM_TAG}] TTS response contains no audio data. Sarvam accepted "
            f"the request but returned an empty audio list — retry later; if it "
            f"persists, contact Kaapi."
        )
        logger.warning(
            f"[synthesize_sarvam] {message} | provider=sarvam, model={model}, "
            f"request_id={response.request_id}"
        )
        raise TTSSynthesisError(message)

    # Long inputs come back as several base64 chunks, each a complete WAV file.
    # Joining the base64 strings (or the decoded bytes) would leave the later
    # chunks' RIFF headers inside the PCM stream as audible clicks, so each chunk
    # is decoded and stripped separately and the PCM re-wrapped once.
    pcm_parts: list[bytes] = []
    try:
        for audio_b64 in response.audios:
            pcm_parts.append(_wav_chunk_to_pcm(base64.b64decode(audio_b64)))
    except (ValueError, wave.Error, EOFError) as e:
        message = (
            f"[{KAAPI_TAG}] Failed to decode Sarvam audio (code: {type(e).__name__}): "
            f"{e}. The response was not valid base64 WAV — contact Kaapi if this persists."
        )
        logger.error(
            f"[synthesize_sarvam] {message} | provider=sarvam, model={model}, "
            f"request_id={response.request_id}",
            exc_info=True,
        )
        raise TTSSynthesisError(message) from e

    logger.info(
        f"[synthesize_sarvam] Synthesized | model={model}, "
        f"chunks={len(response.audios)}, request_id={response.request_id}"
    )
    return _to_synthesized_audio(b"".join(pcm_parts))


def synthesize_elevenlabs(
    *,
    client: ElevenLabs,
    text: str,
    model: str,
    language_code: str,
    voice_id: str,
) -> SynthesizedAudio:
    """Synthesize one text with ElevenLabs (raw 24 kHz PCM) and return it as WAV.

    Raises:
        TTSSynthesisError: On any provider or network failure
    """
    convert_kwargs: dict[str, Any] = {}
    elevenlabs_language = to_elevenlabs_language_code(language_code)
    if elevenlabs_language:
        convert_kwargs["language_code"] = elevenlabs_language

    # The SDK streams lazily, so HTTP errors can surface while joining the chunks.
    try:
        chunks = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
            request_options=_ELEVENLABS_REQUEST_OPTIONS,
            **convert_kwargs,
        )
        pcm_data = b"".join(chunks)
    except ElevenLabsApiError as e:
        raise _api_error(
            tag=ELEVENLABS_TAG,
            status_code=e.status_code,
            body=e.body,
            model=model,
            function_name="synthesize_elevenlabs",
        ) from e
    except httpx.TransportError as e:
        raise _network_error(
            err=e,
            tag=ELEVENLABS_TAG,
            model=model,
            function_name="synthesize_elevenlabs",
        ) from e

    if not pcm_data:
        message = (
            f"[{ELEVENLABS_TAG}] TTS response contains no audio data. ElevenLabs "
            f"accepted the request but returned an empty audio stream — retry "
            f"later; if it persists, contact Kaapi."
        )
        logger.warning(
            f"[synthesize_elevenlabs] {message} | provider=elevenlabs, "
            f"model={model}, voice_id={voice_id}"
        )
        raise TTSSynthesisError(message)

    logger.info(
        f"[synthesize_elevenlabs] Synthesized | model={model}, "
        f"voice_id={voice_id}, pcm_bytes={len(pcm_data)}"
    )
    return _to_synthesized_audio(pcm_data)


def _is_retryable_synthesis_error(err: BaseException) -> bool:
    return isinstance(err, TTSSynthesisError) and err.retryable


@retry(
    retry=retry_if_exception(_is_retryable_synthesis_error),
    wait=wait_random_exponential(
        multiplier=TTS_SYNC_BACKOFF_BASE_SECONDS, max=TTS_SYNC_BACKOFF_MAX_SECONDS
    ),
    stop=stop_after_attempt(TTS_SYNC_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)
def synthesize_tts(
    *,
    client: TTSClient,
    provider: Provider,
    text: str,
    model: str,
    language_code: str,
    voice: str,
) -> SynthesizedAudio:
    """Dispatch to the provider's synthesizer, retrying transient failures.

    Raises:
        TTSSynthesisError: On a non-retryable failure or after the final attempt
    """
    # Dispatch on the registry provider rather than isinstance so the client
    # can be substituted (e.g. a mock) without matching the SDK class.
    if provider == Provider.SARVAMAI:
        return synthesize_sarvam(
            client=cast(SarvamAI, client),
            text=text,
            model=model,
            language_code=language_code,
            speaker=voice,
        )
    if provider == Provider.ELEVENLABS:
        return synthesize_elevenlabs(
            client=cast(ElevenLabs, client),
            text=text,
            model=model,
            language_code=language_code,
            voice_id=voice,
        )

    message = (
        f"[{KAAPI_TAG}] No synchronous synthesizer for provider "
        f"'{provider.value}'. Contact Kaapi."
    )
    logger.error(f"[synthesize_tts] {message} | model={model}")
    raise TTSSynthesisError(message)
