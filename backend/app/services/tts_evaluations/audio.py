"""Audio processing utilities for TTS evaluation."""

import io
import wave


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
