"""
Audio processing utilities for format conversion.

This module provides utilities for converting audio between different formats,
particularly for TTS output post-processing.
"""
import io
import logging
from pydub import AudioSegment


logger = logging.getLogger(__name__)


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
