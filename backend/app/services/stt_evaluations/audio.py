"""Audio file validation and upload service for STT evaluation."""

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlmodel import Session

from app.core.cloud.storage import get_cloud_storage
from app.core.exception_handlers import HTTPException
from app.models.stt_evaluation import AudioUploadResponse
from app.services.stt_evaluations.constants import (
    MAX_FILE_SIZE_BYTES,
    MIME_TO_EXTENSION,
    SUPPORTED_AUDIO_FORMATS,
)

logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    """Exception raised for audio validation errors."""

    pass


def get_extension_from_filename(filename: str) -> str | None:
    """Extract and validate file extension from filename.

    Args:
        filename: Original filename

    Returns:
        str: Lowercase file extension (without dot)
        None: If no valid extension found
    """
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


def get_extension_from_content_type(content_type: str) -> str | None:
    """Get file extension from MIME content type.

    Args:
        content_type: MIME content type

    Returns:
        str: File extension
        None: If content type not recognized
    """
    if not content_type:
        return None
    return MIME_TO_EXTENSION.get(content_type.lower())


def validate_audio_file(file: UploadFile) -> str:
    """Validate an uploaded audio file.

    Args:
        file: FastAPI UploadFile object

    Returns:
        str: Validated file extension

    Raises:
        AudioValidationError: If file is invalid
    """
    # Check filename exists
    if not file.filename:
        logger.error("[validate_audio_file] No filename provided")
        raise AudioValidationError("Filename is required")

    # Get extension from filename
    extension = get_extension_from_filename(file.filename)

    # If no extension from filename, try content type
    if not extension:
        extension = get_extension_from_content_type(file.content_type)

    # Validate extension is supported
    if not extension or extension not in SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        logger.error(
            f"[validate_audio_file] Unsupported audio format | "
            f"filename: {file.filename}, extension: {extension}, "
            f"content_type: {file.content_type}"
        )
        raise AudioValidationError(
            f"Unsupported audio format: {extension or 'unknown'}. "
            f"Supported formats: {supported}"
        )

    # Check file size (if available)
    if file.size and file.size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        file_mb = file.size / (1024 * 1024)
        logger.error(
            f"[validate_audio_file] File too large | "
            f"filename: {file.filename}, size_mb: {file_mb:.2f}, max_mb: {max_mb}"
        )
        raise AudioValidationError(
            f"File too large: {file_mb:.2f} MB. Maximum size: {max_mb:.0f} MB"
        )

    logger.info(
        f"[validate_audio_file] Audio file validated | "
        f"filename: {file.filename}, extension: {extension}, "
        f"content_type: {file.content_type}"
    )

    return extension


def upload_audio_file(
    session: Session,
    file: UploadFile,
    project_id: int,
) -> AudioUploadResponse:
    """Upload an audio file to S3.

    Args:
        session: Database session
        file: FastAPI UploadFile object
        project_id: Project ID

    Returns:
        AudioUploadResponse: Upload result with S3 URL

    Raises:
        HTTPException: If validation or upload fails
    """
    logger.info(
        f"[upload_audio_file] Starting audio upload | "
        f"project_id: {project_id}, filename: {file.filename}"
    )

    try:
        # Validate the audio file
        extension = validate_audio_file(file)
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate unique filename
    file_uuid = uuid.uuid4()
    new_filename = f"{file_uuid}.{extension}"

    # Construct S3 path: stt/audio/{uuid}.{ext}
    file_path = Path("stt") / "audio" / new_filename

    try:
        # Get cloud storage for project
        storage = get_cloud_storage(session=session, project_id=project_id)

        # Upload file to S3
        destination = storage.put(source=file, file_path=file_path)
        s3_url = str(destination)

        # Get file size
        try:
            size_kb = storage.get_file_size_kb(s3_url)
            size_bytes = int(size_kb * 1024)
        except Exception:
            # If we can't get size from S3, use the upload file size
            size_bytes = file.size or 0

        logger.info(
            f"[upload_audio_file] Audio uploaded successfully | "
            f"project_id: {project_id}, s3_url: {s3_url}, size_bytes: {size_bytes}"
        )

        return AudioUploadResponse(
            s3_url=s3_url,
            filename=file.filename or new_filename,
            size_bytes=size_bytes,
            content_type=file.content_type or f"audio/{extension}",
        )

    except Exception as e:
        logger.error(
            f"[upload_audio_file] Failed to upload audio | "
            f"project_id: {project_id}, error: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to upload audio file. Please try again later.",
        )


def validate_s3_audio_url(url: str) -> bool:
    """Validate that a URL is a valid S3 audio file URL.

    Args:
        url: S3 URL to validate

    Returns:
        bool: True if URL appears valid
    """
    if not url:
        return False

    # Check URL format
    if not url.startswith("s3://"):
        return False

    # Check file extension
    extension = get_extension_from_filename(url)
    if not extension or extension not in SUPPORTED_AUDIO_FORMATS:
        return False

    return True
