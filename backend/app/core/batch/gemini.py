"""Gemini batch provider implementation."""

import json
import logging
import os
import tempfile
import time
from enum import Enum
from typing import Any

from google import genai
from google.genai import types

from app.core.storage_utils import get_mime_from_url

from .base import BATCH_KEY, BatchProvider

logger = logging.getLogger(__name__)


class BatchJobState(str, Enum):
    """Gemini batch job states."""

    PENDING = "JOB_STATE_PENDING"
    RUNNING = "JOB_STATE_RUNNING"
    SUCCEEDED = "JOB_STATE_SUCCEEDED"
    FAILED = "JOB_STATE_FAILED"
    CANCELLED = "JOB_STATE_CANCELLED"
    EXPIRED = "JOB_STATE_EXPIRED"


# Terminal states that indicate the batch is done
_TERMINAL_STATES = {
    BatchJobState.SUCCEEDED.value,
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}

# Failed terminal states
_FAILED_STATES = {
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}


class GeminiBatchProvider(BatchProvider):
    """Gemini implementation of the BatchProvider interface.

    Supports both inline requests and JSONL file-based batch submissions.
    Each JSONL line follows the Gemini format:
        {"key": "request-1", "request": {"contents": [{"parts": [...]}]}}
    """

    DEFAULT_MODEL = "models/gemini-2.5-pro"

    def __init__(self, client: genai.Client, model: str | None = None) -> None:
        """Initialize the Gemini batch provider.

        Args:
            client: Configured Gemini client
            model: Model to use (defaults to gemini-2.5-pro)
        """
        self._client = client
        self._model = model or self.DEFAULT_MODEL

    def create_batch(
        self, jsonl_data: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Upload JSONL file and create a batch job with Gemini.

        Args:
            jsonl_data: List of dictionaries in Gemini JSONL format.
                Each dict should have the structure:
                {"key": "request-1", "request": {"contents": [{"parts": [...]}]}}
            config: Provider-specific configuration with:
                - display_name: Optional batch display name
                - model: Optional model override

        Returns:
            Dictionary containing:
                - provider_batch_id: Gemini batch job name
                - provider_file_id: Uploaded JSONL file name
                - provider_status: Initial status from Gemini
                - total_items: Number of items in the batch
        """
        model = config.get("model", self._model)
        display_name = config.get("display_name", f"batch-{int(time.time())}")

        logger.info(
            f"[create_batch] Creating Gemini batch | items={len(jsonl_data)} | "
            f"model={model} | display_name={display_name}"
        )

        try:
            # Create JSONL content
            jsonl_content = "\n".join(json.dumps(item) for item in jsonl_data)

            # Upload JSONL file to Gemini File API
            uploaded_file = self.upload_file(jsonl_content, purpose="batch")

            logger.info(
                f"[create_batch] Uploaded JSONL file | file_name={uploaded_file}"
            )

            # Create batch job using uploaded file
            batch_job = self._client.batches.create(
                model=model,
                src=uploaded_file,
                config={"display_name": display_name},
            )

            initial_state = batch_job.state.name if batch_job.state else "UNKNOWN"

            result = {
                "provider_batch_id": batch_job.name,
                "provider_file_id": uploaded_file,
                "provider_status": initial_state,
                "total_items": len(jsonl_data),
            }

            logger.info(
                f"[create_batch] Created Gemini batch | batch_id={batch_job.name} | "
                f"file_id={uploaded_file} | status={initial_state} | items={len(jsonl_data)}"
            )

            return result

        except Exception as e:
            logger.error(f"[create_batch] Failed to create Gemini batch | {e}")
            raise

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Poll Gemini for batch job status.

        Args:
            batch_id: Gemini batch job name

        Returns:
            Dictionary containing:
                - provider_status: Current Gemini state
                - provider_output_file_id: batch_id (used to fetch results)
                - error_message: Error message (if failed)
        """
        logger.info(
            f"[get_batch_status] Polling Gemini batch status | batch_id={batch_id}"
        )

        try:
            batch_job = self._client.batches.get(name=batch_id)
            state = batch_job.state.name if batch_job.state else "UNKNOWN"

            result: dict[str, Any] = {
                "provider_status": state,
                # Gemini uses the same batch name to fetch results
                "provider_output_file_id": batch_id,
            }

            if state in _FAILED_STATES:
                result["error_message"] = f"Batch {state}"

            logger.info(
                f"[get_batch_status] Gemini batch status | batch_id={batch_id} | "
                f"status={state}"
            )

            return result

        except Exception as e:
            logger.error(
                f"[get_batch_status] Failed to poll Gemini batch status | "
                f"batch_id={batch_id} | {e}"
            )
            raise

    def download_batch_results(
        self, output_file_id: str, request_keys: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Download and parse batch results from Gemini.

        Gemini returns results as a downloadable JSONL file where each line
        contains the key and response.

        Args:
            output_file_id: Gemini batch job name (used to fetch the batch)
            request_keys: Deprecated, kept for interface compatibility.

        Returns:
            List of result dictionaries, each containing:
                - BATCH_KEY: Item key from input
                - response: Dict with "text" key containing the generated text
                - error: Error info (if item failed), None otherwise
        """
        logger.info(
            f"[download_batch_results] Downloading Gemini batch results | "
            f"batch_id={output_file_id}"
        )

        try:
            batch_job = self._client.batches.get(name=output_file_id)
            state = batch_job.state.name if batch_job.state else "UNKNOWN"

            if state != BatchJobState.SUCCEEDED.value:
                raise ValueError(f"Batch job not complete. Current state: {state}")

            results: list[dict[str, Any]] = []

            # Handle file-based results (keys are included in each response line)
            if (
                batch_job.dest
                and hasattr(batch_job.dest, "file_name")
                and batch_job.dest.file_name
            ):
                file_content = self.download_file(batch_job.dest.file_name)
                lines = file_content.strip().split("\n")
                for i, line in enumerate(lines):
                    try:
                        parsed = json.loads(line)
                        custom_id = parsed.get("key", str(i))

                        # Extract text from response
                        response_obj = parsed.get("response")
                        if response_obj:
                            text = self._extract_text_from_response_dict(response_obj)
                            results.append(
                                {
                                    BATCH_KEY: custom_id,
                                    "response": {"text": text},
                                    "error": None,
                                }
                            )
                        elif parsed.get("error"):
                            results.append(
                                {
                                    BATCH_KEY: custom_id,
                                    "response": None,
                                    "error": str(parsed["error"]),
                                }
                            )
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"[download_batch_results] Failed to parse JSON | "
                            f"line={i + 1} | {e}"
                        )
                        continue

            logger.info(
                f"[download_batch_results] Downloaded Gemini batch results | "
                f"batch_id={output_file_id} | results={len(results)}"
            )

            return results

        except Exception as e:
            logger.error(
                f"[download_batch_results] Failed to download Gemini batch results | "
                f"batch_id={output_file_id} | {e}"
            )
            raise

    @staticmethod
    def _extract_text_from_response_dict(response: dict[str, Any]) -> str:
        """Extract text content from a Gemini response dictionary.

        Args:
            response: Gemini response as a dictionary

        Returns:
            str: Extracted text
        """
        # Try direct text field first
        if "text" in response:
            return response["text"]

        # Extract from candidates structure
        text = ""
        for candidate in response.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    text += part["text"]
        return text

    def upload_file(self, content: str, purpose: str = "batch") -> str:
        """Upload a JSONL file to Gemini Files API.

        Args:
            content: File content (JSONL string)
            purpose: Purpose of the file (unused for Gemini, kept for interface)

        Returns:
            Gemini file name (e.g., "files/xxx")
        """
        logger.info(f"[upload_file] Uploading file to Gemini | bytes={len(content)}")

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                uploaded_file = self._client.files.upload(
                    file=tmp_path,
                    config=types.UploadFileConfig(
                        display_name=f"batch-input-{int(time.time())}",
                        mime_type="jsonl",
                    ),
                )

                logger.info(
                    f"[upload_file] Uploaded file to Gemini | "
                    f"file_name={uploaded_file.name}"
                )

                return uploaded_file.name

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"[upload_file] Failed to upload file to Gemini | {e}")
            raise

    def download_file(self, file_id: str) -> str:
        """Download a file from Gemini Files API.

        Args:
            file_id: Gemini file name (e.g., "files/xxx")

        Returns:
            File content as UTF-8 string
        """
        logger.info(f"[download_file] Downloading file from Gemini | file_id={file_id}")

        try:
            file_content = self._client.files.download(file=file_id)
            content = file_content.decode("utf-8")

            logger.info(
                f"[download_file] Downloaded file from Gemini | "
                f"file_id={file_id} | bytes={len(content)}"
            )

            return content

        except Exception as e:
            logger.error(
                f"[download_file] Failed to download file from Gemini | "
                f"file_id={file_id} | {e}"
            )
            raise

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        """Extract text content from a Gemini response object.

        Args:
            response: Gemini GenerateContentResponse

        Returns:
            str: Extracted text
        """
        if hasattr(response, "text"):
            return response.text

        text = ""
        if hasattr(response, "candidates"):
            for candidate in response.candidates:
                if hasattr(candidate, "content"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            text += part.text
        return text


def create_stt_batch_requests(
    signed_urls: list[str],
    prompt: str,
    keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Create batch API requests for Gemini STT using signed URLs.

    This function generates request payloads in Gemini's JSONL batch format
    using signed URLs directly. MIME types are automatically detected from the URL path.

    Args:
        signed_urls: List of signed URLs pointing to audio files
        prompt: Transcription prompt/instructions for the model
        keys: Optional list of custom IDs for tracking results. If not provided,
              uses 0-indexed integers as strings.

    Returns:
        List of batch request dicts in Gemini JSONL format:
            {"key": "sample-1", "request": {"contents": [...]}}

    Example:
        >>> urls = ["https://bucket.s3.amazonaws.com/audio.mp3?..."]
        >>> prompt = "Transcribe this audio file."
        >>> requests = create_stt_batch_requests(urls, prompt, keys=["sample-1"])
        >>> provider.create_batch(requests, {"display_name": "stt-batch"})
    """
    if keys is not None and len(keys) != len(signed_urls):
        raise ValueError(
            f"Length of keys ({len(keys)}) must match signed_urls ({len(signed_urls)})"
        )

    requests = []
    for i, url in enumerate(signed_urls):
        mime_type = get_mime_from_url(url)
        if mime_type is None:
            logger.warning(
                f"[create_stt_batch_requests] Could not determine MIME type for URL | "
                f"index={i} | defaulting to audio/mpeg"
            )
            mime_type = "audio/mpeg"

        # Use provided key or generate from index
        key = keys[i] if keys is not None else str(i)

        # Gemini JSONL format: {"key": "...", "request": {...}}
        request = {
            "key": key,
            "request": {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"file_data": {"mime_type": mime_type, "file_uri": url}},
                        ],
                        "role": "user",
                    }
                ]
            },
        }
        requests.append(request)

    logger.info(f"[create_stt_batch_requests] Created {len(requests)} batch requests")

    return requests
