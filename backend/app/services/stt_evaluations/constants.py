"""Shared constants for STT evaluation services."""

# Supported audio formats for STT evaluation
SUPPORTED_AUDIO_FORMATS: set[str] = {"mp3", "wav", "flac", "m4a", "ogg", "webm"}

# Maximum audio file size (200 MB)
MAX_FILE_SIZE_BYTES: int = 200 * 1024 * 1024

# Mapping from file extension to MIME type
EXTENSION_TO_MIME: dict[str, str] = {
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
}

# Mapping from MIME type to file extension
MIME_TO_EXTENSION: dict[str, str] = {
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
}
