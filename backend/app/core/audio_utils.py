"""Audio processing utilities: format conversion + STT input carrier."""

import io
import logging
import os
import tempfile
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from pydub import AudioSegment

logger = logging.getLogger(__name__)

_MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
    "audio/mp4": ".mp4",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/aiff": ".aiff",
}


@dataclass(frozen=True)
class AudioRef:
    """In-memory STT input. Providers consume ``bytes_`` directly or call
    ``to_path()`` when an SDK needs a filesystem path. Temp files are owned
    by the provider's ``with`` scope — no framework-level cleanup needed.
    """

    bytes_: bytes
    mime_type: str = "audio/wav"

    @contextmanager
    def to_path(self) -> Iterator[str]:
        ext = _MIME_TO_EXT.get(self.mime_type, ".audio")
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False, prefix="audio_")
        try:
            tmp.write(self.bytes_)
            tmp.close()
            yield tmp.name
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def convert_pcm_to_mp3(
    pcm_bytes: bytes, sample_rate: int = 24000
) -> tuple[bytes | None, str | None]:
    try:
        audio = AudioSegment(
            data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1
        )

        output_buffer = io.BytesIO()
        audio.export(output_buffer, format="mp3", bitrate="192k")
        return output_buffer.getvalue(), None
    except Exception as e:
        return None, str(e)


def convert_pcm_to_ogg(
    pcm_bytes: bytes, sample_rate: int = 24000
) -> tuple[bytes | None, str | None]:
    """Convert raw PCM to OGG with Opus codec."""
    try:
        audio = AudioSegment(
            data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1
        )

        output_buffer = io.BytesIO()
        audio.export(
            output_buffer, format="ogg", codec="libopus", parameters=["-b:a", "64k"]
        )
        return output_buffer.getvalue(), None
    except Exception as e:
        return None, str(e)


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    bits_per_sample: int = 16,
    channels: int = 1,
) -> bytes:
    """Wrap raw PCM audio data in a WAV container.

    Args:
        pcm_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz (default: 24000 for Gemini TTS)
        bits_per_sample: Bits per sample (default: 16)
        channels: Number of audio channels (default: 1 mono)

    Returns:
        WAV file bytes with proper headers
    """
    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return output.getvalue()


def calculate_duration(
    pcm_size: int,
    sample_rate: int = 24000,
    bits_per_sample: int = 16,
    channels: int = 1,
) -> float:
    """Calculate audio duration from PCM data size.

    Args:
        pcm_size: Size of raw PCM data in bytes
        sample_rate: Sample rate in Hz
        bits_per_sample: Bits per sample
        channels: Number of audio channels

    Returns:
        Duration in seconds
    """
    bytes_per_sample = bits_per_sample // 8
    bytes_per_second = sample_rate * bytes_per_sample * channels
    if bytes_per_second == 0:
        return 0.0
    return pcm_size / bytes_per_second
