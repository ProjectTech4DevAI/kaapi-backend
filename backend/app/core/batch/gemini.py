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

from .base import BatchProvider

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

    def __init__(self, client: genai.Client, model: str | None = None):
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
        """Upload JSONL data and create a batch job with Gemini.

        Args:
            jsonl_data: List of dictionaries representing JSONL lines.
                Each dict should be a valid GenerateContentRequest, e.g.:
                {"contents": [{"parts": [{"text": "..."}]}]}
            config: Provider-specific configuration with:
                - display_name: Optional batch display name
                - model: Optional model override

        Returns:
            Dictionary containing:
                - provider_batch_id: Gemini batch job name
                - provider_file_id: Uploaded JSONL file name (or None for inline)
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
            # Use inline requests for the batch
            batch_job = self._client.batches.create(
                model=model,
                src=jsonl_data,
                config={"display_name": display_name},
            )

            initial_state = batch_job.state.name if batch_job.state else "UNKNOWN"

            result = {
                "provider_batch_id": batch_job.name,
                "provider_file_id": None,
                "provider_status": initial_state,
                "total_items": len(jsonl_data),
            }

            logger.info(
                f"[create_batch] Created Gemini batch | batch_id={batch_job.name} | "
                f"status={initial_state} | items={len(jsonl_data)}"
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

    def download_batch_results(self, output_file_id: str) -> list[dict[str, Any]]:
        """Download and parse batch results from Gemini.

        Gemini returns results either as inlined responses or as a
        downloadable JSONL file. This method handles both formats and
        normalizes the output to match the BatchProvider interface.

        Args:
            output_file_id: Gemini batch job name (used to fetch the batch)

        Returns:
            List of result dictionaries, each containing:
                - custom_id: Item key from input (or index as string)
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

            # Handle inline responses
            if batch_job.dest and batch_job.dest.inlined_responses:
                for i, response in enumerate(batch_job.dest.inlined_responses):
                    if response.response:
                        text = self._extract_text_from_response(response.response)
                        results.append(
                            {
                                "custom_id": str(i),
                                "response": {"text": text},
                                "error": None,
                            }
                        )
                    elif response.error:
                        results.append(
                            {
                                "custom_id": str(i),
                                "response": None,
                                "error": str(response.error),
                            }
                        )

            # Handle file-based results
            elif (
                batch_job.dest
                and hasattr(batch_job.dest, "file_name")
                and batch_job.dest.file_name
            ):
                file_content = self.download_file(batch_job.dest.file_name)
                lines = file_content.strip().split("\n")
                for i, line in enumerate(lines):
                    try:
                        parsed = json.loads(line)
                        text = parsed.get("response", {}).get("text", "")
                        custom_id = parsed.get("key", str(i))
                        results.append(
                            {
                                "custom_id": custom_id,
                                "response": {"text": text},
                                "error": None,
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
