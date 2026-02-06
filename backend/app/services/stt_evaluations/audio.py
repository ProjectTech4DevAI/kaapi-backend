"""Audio file upload service for STT evaluation."""

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlmodel import Session

from app.core.cloud.storage import get_cloud_storage
from app.core.exception_handlers import HTTPException
from app.crud.file import create_file
from app.models.file import FileType
from app.models.stt_evaluation import AudioUploadResponse
from app.services.stt_evaluations.constants import (
    MAX_FILE_SIZE_BYTES,
    MIME_TO_EXTENSION,
    SUPPORTED_AUDIO_FORMATS,
)

logger = logging.getLogger(__name__)


def _resolve_extension(file: UploadFile) -> str | None:
    """Get audio file extension from filename, falling back to content type."""
    if file.filename and "." in file.filename:
        return file.filename.rsplit(".", 1)[-1].lower()
    if file.content_type:
        return MIME_TO_EXTENSION.get(file.content_type.lower())
    return None


def _validate_audio_file(file: UploadFile) -> str:
    """Validate an uploaded audio file and return its extension.

    Raises:
        HTTPException: If file is invalid
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    extension = _resolve_extension(file)

    if not extension or extension not in SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format: {extension or 'unknown'}. "
                f"Supported formats: {supported}"
            ),
        )

    if file.size and file.size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        file_mb = file.size / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_mb:.2f} MB. Maximum size: {max_mb:.0f} MB",
        )

    return extension


def upload_audio_file(
    session: Session,
    file: UploadFile,
    organization_id: int,
    project_id: int,
) -> AudioUploadResponse:
    """Upload an audio file to S3 and create a file record.

    Raises:
        HTTPException: If validation or upload fails
    """
    logger.info(
        f"[upload_audio_file] Starting audio upload | "
        f"project_id: {project_id}, filename: {file.filename}"
    )

    extension = _validate_audio_file(file)

    file_uuid = uuid.uuid4()
    new_filename = f"{file_uuid}.{extension}"
    file_path = Path("stt") / "audio" / new_filename

    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        s3_url = str(storage.put(source=file, file_path=file_path))

        try:
            size_bytes = int(storage.get_file_size_kb(s3_url) * 1024)
        except Exception:
            size_bytes = file.size or 0

        original_filename = file.filename or new_filename
        content_type = file.content_type or f"audio/{extension}"

        file_record = create_file(
            session=session,
            object_store_url=s3_url,
            filename=original_filename,
            size_bytes=size_bytes,
            content_type=content_type,
            file_type=FileType.AUDIO.value,
            organization_id=organization_id,
            project_id=project_id,
        )

        logger.info(
            f"[upload_audio_file] Audio uploaded successfully | "
            f"project_id: {project_id}, file_id: {file_record.id}, "
            f"s3_url: {s3_url}, size_bytes: {size_bytes}"
        )

        return AudioUploadResponse(
            file_id=file_record.id,
            s3_url=s3_url,
            filename=original_filename,
            size_bytes=size_bytes,
            content_type=content_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[upload_audio_file] Failed to upload audio | "
            f"project_id: {project_id}, error: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to upload audio file. Please try again later.",
        )
