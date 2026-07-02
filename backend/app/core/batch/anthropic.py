"""Anthropic batch provider implementation."""

import json
import logging
from enum import Enum
from typing import Any

from anthropic import Anthropic

from app.models.llm.constants import (
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    DEFAULT_TEXT_MODELS,
)

from .base import BATCH_KEY, BatchProvider

logger = logging.getLogger(__name__)

STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-12-15"


class MessageBatchStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    CANCELING = "canceling"
    ENDED = "ended"


class AnthropicBatchProvider(BatchProvider):
    """Anthropic implementation of the BatchProvider interface."""

    def __init__(self, client: Anthropic):
        """
        Initialize the Anthropic batch provider.

        Args:
            client: Configured Anthropic client
        """
        self.client = client

    def create_batch(
        self, jsonl_data: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create a batch job with Anthropic from inline requests.

        Args:
            jsonl_data: List of dictionaries in Anthropic batch request format:
                {"custom_id": "request-1", "params": {...Messages API params...}}
            config: Provider-specific configuration with:
                - model: Default model applied to requests missing one
                - max_tokens: Default max_tokens applied to requests missing one

        Returns:
            Dictionary containing:
                - provider_batch_id: Anthropic message batch ID
                - provider_file_id: Always None (Anthropic batches are inline)
                - provider_status: Initial processing status from Anthropic
                - total_items: Number of items in the batch

        Raises:
            Exception: If batch creation fails
        """
        default_model = config.get("model") or DEFAULT_TEXT_MODELS["anthropic"]
        default_max_tokens = config.get("max_tokens") or DEFAULT_ANTHROPIC_MAX_TOKENS

        logger.info(
            f"[create_batch] Creating Anthropic batch | items={len(jsonl_data)} | "
            f"model={default_model}"
        )

        try:
            requests = []
            needs_structured_outputs = False
            for item in jsonl_data:
                params = {**item.get("params", {})}
                params["model"] = params.get("model") or default_model
                params["max_tokens"] = params.get("max_tokens") or default_max_tokens
                if "output_config" in params:
                    needs_structured_outputs = True
                requests.append({"custom_id": item[BATCH_KEY], "params": params})

            betas = [STRUCTURED_OUTPUTS_BETA] if needs_structured_outputs else []
            batch = self.client.beta.messages.batches.create(
                requests=requests, betas=betas
            )

            result = {
                "provider_batch_id": batch.id,
                "provider_file_id": None,
                "provider_status": batch.processing_status,
                "total_items": len(jsonl_data),
            }

            logger.info(
                f"[create_batch] Created Anthropic batch | batch_id={batch.id} | "
                f"status={batch.processing_status} | items={len(jsonl_data)}"
            )

            return result

        except Exception as e:
            logger.error(f"[create_batch] Failed to create Anthropic batch | {e}")
            raise

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """
        Poll Anthropic for batch job status.

        Args:
            batch_id: Anthropic message batch ID

        Returns:
            Dictionary containing:
                - provider_status: Current processing status
                  ("in_progress", "canceling", or "ended")
                - provider_output_file_id: batch_id (results are streamed by
                  batch ID; there is no separate output file)
                - error_message: Error message (if batch ended with no successes)
                - request_counts: Dict with total/completed/failed counts

        Raises:
            Exception: If status check fails
        """
        logger.info(
            f"[get_batch_status] Polling Anthropic batch status | batch_id={batch_id}"
        )

        try:
            batch = self.client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts

            succeeded = counts.succeeded
            failed = counts.errored + counts.canceled + counts.expired
            total = counts.processing + succeeded + failed

            result: dict[str, Any] = {
                "provider_status": batch.processing_status,
                "provider_output_file_id": batch_id,
                "request_counts": {
                    "total": total,
                    "completed": succeeded,
                    "failed": failed,
                },
            }

            if (
                batch.processing_status == MessageBatchStatus.ENDED.value
                and succeeded == 0
                and total > 0
            ):
                result["error_message"] = (
                    f"Batch ended with no successful requests "
                    f"(errored={counts.errored}, canceled={counts.canceled}, "
                    f"expired={counts.expired})"
                )

            logger.info(
                f"[get_batch_status] Anthropic batch status | batch_id={batch_id} | "
                f"status={batch.processing_status} | completed={succeeded}/{total}"
            )

            return result

        except Exception as e:
            logger.error(
                f"[get_batch_status] Failed to poll Anthropic batch status | "
                f"batch_id={batch_id} | {e}"
            )
            raise

    def download_batch_results(self, output_file_id: str) -> list[dict[str, Any]]:
        """
        Download and parse batch results from Anthropic.

        Anthropic streams results as individual responses keyed by custom_id.
        Results may arrive in any order relative to the input.

        Args:
            output_file_id: Anthropic message batch ID (results are fetched
                by batch ID; there is no separate output file)

        Returns:
            List of result dictionaries, each containing:
                - BATCH_KEY: Item identifier from input
                - response: Anthropic Message as a dict (if succeeded)
                - error: Error info string (if item failed), None otherwise

        Raises:
            Exception: If download or parsing fails
        """
        logger.info(
            f"[download_batch_results] Downloading Anthropic batch results | "
            f"batch_id={output_file_id}"
        )

        try:
            results: list[dict[str, Any]] = []

            for item in self.client.messages.batches.results(output_file_id):
                result_type = item.result.type

                if result_type == "succeeded":
                    results.append(
                        {
                            BATCH_KEY: item.custom_id,
                            "response": item.result.message.model_dump(mode="json"),
                            "error": None,
                        }
                    )
                elif result_type == "errored":
                    results.append(
                        {
                            BATCH_KEY: item.custom_id,
                            "response": None,
                            "error": str(item.result.error.model_dump(mode="json")),
                        }
                    )
                else:
                    results.append(
                        {
                            BATCH_KEY: item.custom_id,
                            "response": None,
                            "error": f"Request {result_type}",
                        }
                    )

            logger.info(
                f"[download_batch_results] Downloaded Anthropic batch results | "
                f"batch_id={output_file_id} | results={len(results)}"
            )

            return results

        except Exception as e:
            logger.error(
                f"[download_batch_results] Failed to download Anthropic batch "
                f"results | batch_id={output_file_id} | {e}"
            )
            raise

    def upload_file(self, content: str, purpose: str = "batch") -> str:
        """Not supported: Anthropic batches take requests inline.

        Raises:
            NotImplementedError: Always — there is no file upload step.
        """
        raise NotImplementedError(
            "Anthropic Message Batches accept requests inline; "
            "no file upload is needed."
        )

    def download_file(self, file_id: str) -> str:
        """Not supported: Anthropic batch results are streamed by batch ID.

        Use download_batch_results() instead.

        Raises:
            NotImplementedError: Always — there is no output file.
        """
        raise NotImplementedError(
            "Anthropic Message Batches stream results by batch ID; "
            "use download_batch_results() instead."
        )
