"""Unit tests for LLM input resolver functions."""

import base64
import os
from pathlib import Path

from app.core.audio_utils import AudioRef
from app.models.llm.request import (
    AudioContent,
    AudioInput,
    TextContent,
    TextInput,
)
from app.utils import (
    cleanup_temp_file,
    get_file_extension,
    resolve_audio_base64,
    resolve_input,
)


class TestGetFileExtension:
    def test_common_audio_formats(self):
        assert get_file_extension("audio/wav") == ".wav"
        assert get_file_extension("audio/mp3") == ".mp3"
        assert get_file_extension("audio/mpeg") == ".mp3"
        assert get_file_extension("audio/ogg") == ".ogg"

    def test_wav_variants(self):
        assert get_file_extension("audio/wave") == ".wav"
        assert get_file_extension("audio/x-wav") == ".wav"

    def test_unknown_mime_type(self):
        assert get_file_extension("audio/unknown") == ".audio"
        assert get_file_extension("application/octet-stream") == ".audio"


class TestResolveInput:
    def test_text_input(self):
        text_input = TextInput(content=TextContent(value="Hello world"))
        content, error = resolve_input(text_input)
        assert content == "Hello world"
        assert error is None

    def test_audio_base64_input_returns_audio_ref(self):
        audio_data = b"RIFF" + b"\x00" * 36  # Minimal WAV header
        encoded = base64.b64encode(audio_data).decode()

        audio_input = AudioInput(
            content=AudioContent(value=encoded, mime_type="audio/wav")
        )
        ref, error = resolve_input(audio_input)

        assert error is None
        assert isinstance(ref, AudioRef)
        assert ref.bytes_ == audio_data
        assert ref.mime_type == "audio/wav"

    def test_invalid_base64_data(self):
        audio_input = AudioInput(
            content=AudioContent(value="not-valid-base64!!!", mime_type="audio/wav")
        )
        content, error = resolve_input(audio_input)
        assert content is None
        assert error is not None
        assert "base64" in error.lower()


class TestResolveAudioBase64:
    def test_valid_base64_audio(self):
        audio_data = b"Test audio content"
        encoded = base64.b64encode(audio_data).decode()

        ref, error = resolve_audio_base64(encoded, "audio/mp3")

        assert error is None
        assert isinstance(ref, AudioRef)
        assert ref.bytes_ == audio_data
        assert ref.mime_type == "audio/mp3"

    def test_invalid_base64_string(self):
        ref, error = resolve_audio_base64("invalid!!!base64", "audio/wav")
        assert ref is None
        assert error is not None
        assert "Invalid base64" in error


class TestAudioRefToPath:
    def test_to_path_writes_and_cleans_up(self):
        audio_data = b"Audio bytes"
        ref = AudioRef(bytes_=audio_data, mime_type="audio/wav")

        with ref.to_path() as p:
            assert Path(p).exists()
            assert p.endswith(".wav")
            with open(p, "rb") as f:
                assert f.read() == audio_data

        # File must be cleaned up after the context exits.
        assert not Path(p).exists()

    def test_to_path_cleans_up_on_exception(self):
        ref = AudioRef(bytes_=b"x", mime_type="audio/ogg")
        captured_path = None
        try:
            with ref.to_path() as p:
                captured_path = p
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert captured_path is not None
        assert not Path(captured_path).exists()


class TestCleanupTempFile:
    """cleanup_temp_file remains in app.utils for non-AudioRef callers."""

    def test_cleanup_nonexistent_file(self):
        cleanup_temp_file("/tmp/nonexistent_file_12345.wav")

    def test_cleanup_invalid_path(self):
        cleanup_temp_file("")
