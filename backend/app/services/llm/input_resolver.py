import base64
import logging
import tempfile
from pathlib import Path

from app.models.llm.request import (
    TextInput,
    AudioInput,
    QueryInput,
)


logger = logging.getLogger(__name__)


def get_file_extension(mime_type: str) -> str:
    """Map MIME type to file extension."""
    mime_to_ext = {
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/webm": ".webm",
        "audio/mp4": ".mp4",
        "audio/m4a": ".m4a",
    }
    return mime_to_ext.get(mime_type, ".audio")


# important!!
def resolve_input(query_input: QueryInput) -> tuple[str, str | None]:
    """Resolve discriminated union input to content string.

    Args:
        query_input: The input from QueryParams (TextInput or AudioInput)

    Returns:
        (content_string, None) on success - for text returns content value, for audio returns temp file path
        ("", error_message) on failure
    """
    try:
        if isinstance(query_input, TextInput):
            return query_input.content.value, None

        elif isinstance(query_input, AudioInput):
            # AudioInput content is base64-encoded audio
            mime_type = query_input.content.mime_type or "audio/wav"
            return resolve_audio_base64(query_input.content.value, mime_type)

        else:
            return "", f"Unknown input type: {type(query_input)}"

    except Exception as e:
        logger.error(f"[resolve_input] Failed to resolve input: {e}", exc_info=True)
        return "", f"Failed to resolve input: {str(e)}"


def resolve_audio_base64(data: str, mime_type: str) -> tuple[str, str | None]:
    """Decode base64 audio and write to temp file. Returns (file_path, error)."""
    try:
        audio_bytes = base64.b64decode(data)
    except Exception as e:
        return "", f"Invalid base64 audio data: {str(e)}"

    ext = get_file_extension(mime_type)
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, prefix="audio_"
        ) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        logger.info(f"[resolve_audio_base64] Wrote audio to temp file: {temp_path}")
        return temp_path, None
    except Exception as e:
        return "", f"Failed to write audio to temp file: {str(e)}"


def cleanup_temp_file(file_path: str) -> None:
    """Clean up a temporary file if it exists."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"[cleanup_temp_file] Failed to delete temp file: {e}")
