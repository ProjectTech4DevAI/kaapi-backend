"""Abstract interface for LLM batch providers."""

from abc import ABC, abstractmethod
from typing import Any

# Unified key used across all batch providers to identify individual requests/responses.
# OpenAI uses "custom_id" natively; Gemini uses "key" but we normalize to this constant.
BATCH_KEY = "custom_id"


class BatchProvider(ABC):
    """Abstract base class for LLM batch providers (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    def create_batch(
        self, jsonl_data: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Upload JSONL data and create a batch job with the provider.

        Args:
            jsonl_data: List of dictionaries representing JSONL lines
            config: Provider-specific configuration (model, temperature, etc.)

        Returns:
            Dictionary containing:
                - provider_batch_id: Provider's batch job ID
                - provider_file_id: Provider's input file ID
                - provider_status: Initial status from provider
                - total_items: Number of items in the batch
                - Any other provider-specific metadata

        Raises:
            Exception: If batch creation fails
        """
        pass

    @abstractmethod
    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """
        Poll the provider for batch job status.

        Args:
            batch_id: Provider's batch job ID

        Returns:
            Dictionary containing:
                - provider_status: Current status from provider
                - provider_output_file_id: Output file ID (if completed)
                - error_message: Error message (if failed)
                - Any other provider-specific status info

        Raises:
            Exception: If status check fails
        """
        pass

    @abstractmethod
    def download_batch_results(self, output_file_id: str) -> list[dict[str, Any]]:
        """
        Download and parse batch results from the provider.

        Args:
            output_file_id: Provider's output file ID

        Returns:
            List of result dictionaries, each containing:
                - BATCH_KEY: Item identifier from input
                - response: Provider's response data
                - error: Error info (if item failed)
                - Any other provider-specific result data

        Raises:
            Exception: If download or parsing fails
        """
        pass

    @abstractmethod
    def upload_file(self, content: str, purpose: str = "batch") -> str:
        """
        Upload a file to the provider's file storage.

        Args:
            content: File content (typically JSONL string)
            purpose: Purpose of the file (e.g., "batch")

        Returns:
            Provider's file ID

        Raises:
            Exception: If upload fails
        """
        pass

    @abstractmethod
    def download_file(self, file_id: str) -> str:
        """
        Download a file from the provider's file storage.

        Args:
            file_id: Provider's file ID

        Returns:
            File content as string

        Raises:
            Exception: If download fails
        """
        pass
