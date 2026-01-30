"""Google Files API operations for STT evaluation."""

import logging
import os
import tempfile
from typing import BinaryIO

import requests
from google import genai
from google.genai import types

from app.services.stt_evaluations.constants import EXTENSION_TO_MIME

logger = logging.getLogger(__name__)


class GeminiFilesError(Exception):
    """Exception raised for Gemini Files API errors."""

    pass


def get_mime_type(file_path: str) -> str:
    """Get MIME type based on file extension.

    Args:
        file_path: Path or URL of the audio file

    Returns:
        str: MIME type string
    """
    extension = file_path.lower().split(".")[-1]
    return EXTENSION_TO_MIME.get(extension, "audio/mpeg")


def get_extension(file_path: str) -> str:
    """Get file extension from path or URL.

    Args:
        file_path: Path or URL of the audio file

    Returns:
        str: File extension (lowercase, without dot)
    """
    return file_path.lower().split(".")[-1]


class GeminiFilesManager:
    """Manage file uploads to Google Files API."""

    # Base URL for Gemini Files API
    FILES_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, client: genai.Client):
        """Initialize files manager.

        Args:
            client: Gemini client instance
        """
        self._client = client

    def upload_from_bytes(
        self,
        content: bytes,
        filename: str,
        mime_type: str | None = None,
    ) -> str:
        """Upload audio content directly to Google Files API.

        Args:
            content: Audio file content as bytes
            filename: Display name for the file
            mime_type: MIME type (auto-detected if not provided)

        Returns:
            str: Full Google Files API URI (HTTPS format for batch API)

        Raises:
            GeminiFilesError: If upload fails
        """
        if mime_type is None:
            mime_type = get_mime_type(filename)

        logger.info(
            f"[upload_from_bytes] Uploading file to Google Files API | "
            f"filename: {filename}, mime_type: {mime_type}, size_bytes: {len(content)}"
        )

        try:
            # Write content to a temporary file
            extension = get_extension(filename)
            with tempfile.NamedTemporaryFile(
                suffix=f".{extension}", delete=False
            ) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                # Upload to Google Files API
                uploaded_file = self._client.files.upload(
                    file=tmp_path,
                    config=types.UploadFileConfig(
                        display_name=filename,
                        mime_type=mime_type,
                    ),
                )

                # CRITICAL: Return the full HTTPS URI for batch API compatibility
                # The batch API requires full URLs, not short form like "files/xxx"
                file_uri = f"{self.FILES_API_BASE}/{uploaded_file.name}"

                logger.info(
                    f"[upload_from_bytes] File uploaded successfully | "
                    f"filename: {filename}, file_uri: {file_uri}"
                )

                return file_uri

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(
                f"[upload_from_bytes] Failed to upload file | "
                f"filename: {filename}, error: {str(e)}"
            )
            raise GeminiFilesError(f"Failed to upload file {filename}: {str(e)}") from e

    def upload_from_url(
        self,
        signed_url: str,
        filename: str,
        mime_type: str | None = None,
    ) -> str:
        """Upload audio from a signed URL to Google Files API.

        Args:
            signed_url: Signed URL to download the audio file
            filename: Display name for the file
            mime_type: MIME type (auto-detected if not provided)

        Returns:
            str: Full Google Files API URI (HTTPS format for batch API)

        Raises:
            GeminiFilesError: If download or upload fails
        """
        logger.info(
            f"[upload_from_url] Downloading file from URL | filename: {filename}"
        )

        try:
            # Download the file from the signed URL
            response = requests.get(signed_url, timeout=300)  # 5 minute timeout
            response.raise_for_status()

            # Upload to Google Files API
            return self.upload_from_bytes(
                content=response.content,
                filename=filename,
                mime_type=mime_type,
            )

        except requests.RequestException as e:
            logger.error(
                f"[upload_from_url] Failed to download file | "
                f"filename: {filename}, error: {str(e)}"
            )
            raise GeminiFilesError(
                f"Failed to download file {filename}: {str(e)}"
            ) from e

    def upload_from_stream(
        self,
        stream: BinaryIO,
        filename: str,
        mime_type: str | None = None,
    ) -> str:
        """Upload audio from a stream to Google Files API.

        Args:
            stream: Binary stream (e.g., from S3 StreamingBody)
            filename: Display name for the file
            mime_type: MIME type (auto-detected if not provided)

        Returns:
            str: Full Google Files API URI (HTTPS format for batch API)

        Raises:
            GeminiFilesError: If upload fails
        """
        try:
            content = stream.read()
            return self.upload_from_bytes(
                content=content,
                filename=filename,
                mime_type=mime_type,
            )
        except Exception as e:
            logger.error(
                f"[upload_from_stream] Failed to read stream | "
                f"filename: {filename}, error: {str(e)}"
            )
            raise GeminiFilesError(
                f"Failed to read stream for {filename}: {str(e)}"
            ) from e

    def delete_file(self, file_uri: str) -> bool:
        """Delete a file from Google Files API.

        Args:
            file_uri: Full Google Files API URI or short form (files/xxx)

        Returns:
            bool: True if deletion was successful
        """
        # Extract the file name from the URI
        if file_uri.startswith(self.FILES_API_BASE):
            file_name = file_uri.replace(f"{self.FILES_API_BASE}/", "")
        else:
            file_name = file_uri

        logger.info(f"[delete_file] Deleting file | file_name: {file_name}")

        try:
            self._client.files.delete(name=file_name)
            logger.info(
                f"[delete_file] File deleted successfully | file_name: {file_name}"
            )
            return True
        except Exception as e:
            logger.warning(
                f"[delete_file] Failed to delete file | "
                f"file_name: {file_name}, error: {str(e)}"
            )
            return False

    def get_file_status(self, file_uri: str) -> dict:
        """Get the status of a file in Google Files API.

        Args:
            file_uri: Full Google Files API URI or short form (files/xxx)

        Returns:
            dict: File metadata including state
        """
        # Extract the file name from the URI
        if file_uri.startswith(self.FILES_API_BASE):
            file_name = file_uri.replace(f"{self.FILES_API_BASE}/", "")
        else:
            file_name = file_uri

        try:
            file_info = self._client.files.get(name=file_name)
            return {
                "name": file_info.name,
                "display_name": file_info.display_name,
                "mime_type": file_info.mime_type,
                "size_bytes": file_info.size_bytes,
                "state": file_info.state.name if file_info.state else None,
                "uri": file_info.uri,
            }
        except Exception as e:
            logger.error(
                f"[get_file_status] Failed to get file status | "
                f"file_name: {file_name}, error: {str(e)}"
            )
            raise GeminiFilesError(
                f"Failed to get file status for {file_name}: {str(e)}"
            ) from e
