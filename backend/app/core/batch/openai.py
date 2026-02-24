"""OpenAI batch provider implementation."""

import json
import logging
from typing import Any

from openai import OpenAI

from .base import BatchProvider

logger = logging.getLogger(__name__)


class OpenAIBatchProvider(BatchProvider):
    """OpenAI implementation of the BatchProvider interface."""

    def __init__(self, client: OpenAI):
        """
        Initialize the OpenAI batch provider.

        Args:
            client: Configured OpenAI client
        """
        self.client = client

    def create_batch(
        self, jsonl_data: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Upload JSONL data and create a batch job with OpenAI.

        Args:
            jsonl_data: List of dictionaries representing JSONL lines
            config: Provider-specific configuration with:
                - endpoint: OpenAI endpoint (e.g., "/v1/responses")
                - description: Optional batch description
                - completion_window: Optional completion window (default "24h")

        Returns:
            Dictionary containing:
                - provider_batch_id: OpenAI batch ID
                - provider_file_id: OpenAI input file ID
                - provider_status: Initial status from OpenAI
                - total_items: Number of items in the batch

        Raises:
            Exception: If batch creation fails
        """
        endpoint = config.get("endpoint", "/v1/responses")
        description = config.get("description", "LLM batch job")
        completion_window = config.get("completion_window", "24h")

        logger.info(
            f"[create_batch] Creating OpenAI batch | items={len(jsonl_data)} | endpoint={endpoint}"
        )

        try:
            # Step 1: Upload file
            file_id = self.upload_file(
                content="\n".join([json.dumps(line) for line in jsonl_data]),
                purpose="batch",
            )

            # Step 2: Create batch job
            batch = self.client.batches.create(
                input_file_id=file_id,
                endpoint=endpoint,
                completion_window=completion_window,
                metadata={"description": description},
            )

            result = {
                "provider_batch_id": batch.id,
                "provider_file_id": file_id,
                "provider_status": batch.status,
                "total_items": len(jsonl_data),
            }

            logger.info(
                f"[create_batch] Created OpenAI batch | batch_id={batch.id} | status={batch.status} | items={len(jsonl_data)}"
            )

            return result

        except Exception as e:
            logger.error(f"[create_batch] Failed to create OpenAI batch | {e}")
            raise

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """
        Poll OpenAI for batch job status.

        Args:
            batch_id: OpenAI batch ID

        Returns:
            Dictionary containing:
                - provider_status: Current OpenAI status
                - provider_output_file_id: Output file ID (if completed)
                - error_message: Error message (if failed)
                - request_counts: Dict with total/completed/failed counts

        Raises:
            Exception: If status check fails
        """
        logger.info(
            f"[get_batch_status] Polling OpenAI batch status | batch_id={batch_id}"
        )

        try:
            batch = self.client.batches.retrieve(batch_id)

            result = {
                "provider_status": batch.status,
                "provider_output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "request_counts": {
                    "total": batch.request_counts.total,
                    "completed": batch.request_counts.completed,
                    "failed": batch.request_counts.failed,
                },
            }

            # Add error message if batch failed
            if batch.status in ["failed", "expired", "cancelled"]:
                error_msg = f"Batch {batch.status}"
                if batch.error_file_id:
                    error_msg += f" (error_file_id: {batch.error_file_id})"
                result["error_message"] = error_msg

            logger.info(
                f"[get_batch_status] OpenAI batch status | batch_id={batch_id} | status={batch.status} | completed={batch.request_counts.completed}/{batch.request_counts.total}"
            )

            return result

        except Exception as e:
            logger.error(
                f"[get_batch_status] Failed to poll OpenAI batch status | batch_id={batch_id} | {e}"
            )
            raise

    def download_batch_results(self, output_file_id: str) -> list[dict[str, Any]]:
        """
        Download and parse batch results from OpenAI.

        Args:
            output_file_id: OpenAI output file ID

        Returns:
            List of result dictionaries, each containing:
                - BATCH_KEY: Item identifier from input
                - response: OpenAI response data (body, status_code, request_id)
                - error: Error info (if item failed)

        Raises:
            Exception: If download or parsing fails
        """
        logger.info(
            f"[download_batch_results] Downloading OpenAI batch results | output_file_id={output_file_id}"
        )

        try:
            # Download file content
            jsonl_content = self.download_file(output_file_id)

            # Parse JSONL into list of dicts
            results = []
            lines = jsonl_content.strip().split("\n")

            for line_num, line in enumerate(lines, 1):
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"[download_batch_results] Failed to parse JSON | line={line_num} | {e}"
                    )
                    continue

            logger.info(
                f"[download_batch_results] Downloaded and parsed results from OpenAI batch output | results={len(results)}"
            )

            return results

        except Exception as e:
            logger.error(
                f"[download_batch_results] Failed to download OpenAI batch results | {e}"
            )
            raise

    def upload_file(self, content: str, purpose: str = "batch") -> str:
        """
        Upload a file to OpenAI file storage.

        Args:
            content: File content (typically JSONL string)
            purpose: Purpose of the file (e.g., "batch")

        Returns:
            OpenAI file ID

        Raises:
            Exception: If upload fails
        """
        logger.info(f"[upload_file] Uploading file to OpenAI | bytes={len(content)}")

        try:
            file_response = self.client.files.create(
                file=("batch_input.jsonl", content.encode("utf-8")),
                purpose=purpose,
            )

            logger.info(
                f"[upload_file] Uploaded file to OpenAI | file_id={file_response.id}"
            )

            return file_response.id

        except Exception as e:
            logger.error(f"[upload_file] Failed to upload file to OpenAI | {e}")
            raise

    def download_file(self, file_id: str) -> str:
        """
        Download a file from OpenAI file storage.

        Args:
            file_id: OpenAI file ID

        Returns:
            File content as string

        Raises:
            Exception: If download fails
        """
        logger.info(f"[download_file] Downloading file from OpenAI | file_id={file_id}")

        try:
            file_content = self.client.files.content(file_id)
            content = file_content.read().decode("utf-8")

            logger.info(
                f"[download_file] Downloaded file from OpenAI | file_id={file_id} | bytes={len(content)}"
            )

            return content

        except Exception as e:
            logger.error(
                f"[download_file] Failed to download file from OpenAI | file_id={file_id} | {e}"
            )
            raise
