"""Tests for audio utility functions."""
import subprocess
import pytest
from app.core.audio_utils import convert_pcm_to_mp3, convert_pcm_to_ogg


def _is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# Skip all tests in this module if ffmpeg is not available
pytestmark = pytest.mark.skipif(
    not _is_ffmpeg_available(),
    reason="ffmpeg not available in environment",
)


def test_convert_pcm_to_mp3_success() -> None:
    """Test successful PCM to MP3 conversion."""
    # Create minimal valid PCM data (1 second of silence at 24kHz, 16-bit, mono)
    sample_rate = 24000
    duration_seconds = 1
    num_samples = sample_rate * duration_seconds
    pcm_bytes = b"\x00\x00" * num_samples  # 16-bit silence

    result, error = convert_pcm_to_mp3(pcm_bytes, sample_rate=sample_rate)

    assert error is None
    assert result is not None
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_pcm_to_mp3_custom_sample_rate() -> None:
    """Test PCM to MP3 conversion with custom sample rate."""
    sample_rate = 16000
    num_samples = sample_rate  # 1 second
    pcm_bytes = b"\x00\x00" * num_samples

    result, error = convert_pcm_to_mp3(pcm_bytes, sample_rate=sample_rate)

    assert error is None
    assert result is not None
    assert isinstance(result, bytes)


def test_convert_pcm_to_mp3_short_audio() -> None:
    """Test PCM to MP3 conversion with short audio."""
    # Very short audio (100ms)
    sample_rate = 24000
    num_samples = int(sample_rate * 0.1)
    pcm_bytes = b"\x00\x00" * num_samples

    result, error = convert_pcm_to_mp3(pcm_bytes, sample_rate=sample_rate)

    assert error is None
    assert result is not None


def test_convert_pcm_to_ogg_success() -> None:
    """Test successful PCM to OGG conversion."""
    sample_rate = 24000
    duration_seconds = 1
    num_samples = sample_rate * duration_seconds
    pcm_bytes = b"\x00\x00" * num_samples

    result, error = convert_pcm_to_ogg(pcm_bytes, sample_rate=sample_rate)

    assert error is None
    assert result is not None
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_pcm_to_ogg_custom_sample_rate() -> None:
    """Test PCM to OGG conversion with custom sample rate."""
    sample_rate = 16000
    num_samples = sample_rate  # 1 second
    pcm_bytes = b"\x00\x00" * num_samples

    result, error = convert_pcm_to_ogg(pcm_bytes, sample_rate=sample_rate)

    assert error is None
    assert result is not None
    assert isinstance(result, bytes)


def test_convert_pcm_to_ogg_short_audio() -> None:
    """Test PCM to OGG conversion with short audio."""
    # Very short audio (100ms)
    sample_rate = 24000
    num_samples = int(sample_rate * 0.1)
    pcm_bytes = b"\x00\x00" * num_samples

    result, error = convert_pcm_to_ogg(pcm_bytes, sample_rate=sample_rate)

    assert error is None
    assert result is not None
