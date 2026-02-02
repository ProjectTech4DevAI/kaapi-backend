"""
Audio processing utilities for format conversion.

This module provides utilities for converting audio between different formats,
particularly for TTS output post-processing.
"""

import logging
import subprocess
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_pcm_to_wav(
    pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2
) -> bytes:
    """Convert raw PCM audio bytes to WAV format with headers.

    Args:
        pcm_bytes: Raw PCM audio data (16-bit little-endian)
        sample_rate: Sample rate in Hz (default: 24000 for Gemini TTS)
        channels: Number of audio channels (default: 1 for mono)
        sample_width: Sample width in bytes (default: 2 for 16-bit)

    Returns:
        WAV file bytes with proper headers
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)

        with open(temp_path, "rb") as f:
            wav_bytes = f.read()

        temp_path.unlink(missing_ok=True)
        return wav_bytes

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise e


def _convert_audio_with_ffmpeg(
    wav_bytes: bytes,
    output_format: str,
    codec: str,
    quality_arg: str,
    quality_value: str,
    func_name: str,
) -> tuple[bytes | None, str | None]:
    """Helper function to convert audio using ffmpeg.

    Args:
        wav_bytes: WAV audio data with headers
        output_format: Output format extension (mp3, ogg)
        codec: ffmpeg codec name (libmp3lame, libvorbis)
        quality_arg: Quality argument flag (-qscale:a)
        quality_value: Quality value
        func_name: Calling function name for logging

    Returns:
        Tuple of (converted_bytes, error_message)
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_file.write(wav_bytes)
            wav_path = Path(wav_file.name)

        output_path = wav_path.with_suffix(f".{output_format}")

        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(wav_path),
                "-codec:a",
                codec,
                quality_arg,
                quality_value,
                "-y",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = f"ffmpeg conversion failed: {result.stderr}"
            logger.error(f"[{func_name}] {error_msg}")
            wav_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            return None, error_msg

        with open(output_path, "rb") as f:
            output_bytes = f.read()

        wav_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

        logger.info(
            f"[{func_name}] Successfully converted WAV ({len(wav_bytes)} bytes) "
            f"to {output_format.upper()} ({len(output_bytes)} bytes)"
        )

        return output_bytes, None

    except FileNotFoundError:
        error_msg = "ffmpeg not found. Please install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        logger.error(f"[{func_name}] {error_msg}")
        return None, error_msg

    except Exception as e:
        error_msg = f"Unexpected error during audio conversion: {str(e)}"
        logger.error(f"[{func_name}] {error_msg}", exc_info=True)
        return None, error_msg


def convert_wav_to_mp3(
    wav_bytes: bytes, is_raw_pcm: bool = True
) -> tuple[bytes | None, str | None]:
    """Convert WAV audio bytes to MP3 format using ffmpeg.

    Args:
        wav_bytes: WAV audio data or raw PCM data (16-bit)
        is_raw_pcm: If True, treat input as raw PCM and add WAV headers first

    Returns:
        Tuple of (mp3_bytes, error_message)
    """
    if is_raw_pcm:
        logger.info("[convert_wav_to_mp3] Converting raw PCM to WAV format first")
        wav_bytes = convert_pcm_to_wav(wav_bytes)

    return _convert_audio_with_ffmpeg(
        wav_bytes, "mp3", "libmp3lame", "-qscale:a", "2", "convert_wav_to_mp3"
    )


def convert_wav_to_ogg(
    wav_bytes: bytes, is_raw_pcm: bool = True
) -> tuple[bytes | None, str | None]:
    """Convert WAV audio bytes to OGG format using ffmpeg.

    Args:
        wav_bytes: WAV audio data or raw PCM data (16-bit)
        is_raw_pcm: If True, treat input as raw PCM and add WAV headers first

    Returns:
        Tuple of (ogg_bytes, error_message)
    """
    if is_raw_pcm:
        logger.info("[convert_wav_to_ogg] Converting raw PCM to WAV format first")
        wav_bytes = convert_pcm_to_wav(wav_bytes)

    return _convert_audio_with_ffmpeg(
        wav_bytes, "ogg", "libvorbis", "-qscale:a", "5", "convert_wav_to_ogg"
    )
