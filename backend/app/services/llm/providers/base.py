"""Base provider interface for LLM providers.

This module defines the abstract base class that all LLM providers must implement.
It provides a provider-agnostic interface for executing LLM calls.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.models.llm import NativeCompletionConfig, LLMCallResponse, QueryParams


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations (OpenAI, Anthropic, etc.) must inherit from
    this class and implement the required methods.

    Providers directly pass user configuration to their respective APIs.
    User is responsible for providing valid provider-specific parameters.

    Attributes:
        client: The provider-specific client instance
    """

    def __init__(self, client: Any):
        """Initialize provider with client.

        Args:
            client: Provider-specific client instance
        """
        self.client = client

    @abstractmethod
    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute LLM API call.

        Directly passes the user's config params to provider API along with input.

        Args:
            completion_config: LLM completion configuration, pass params as-is to provider API
            query: Query parameters including input and conversation_id
            include_provider_raw_response: Whether to include the raw LLM provider response in the output

        Returns:
            Tuple of (response, error_message)
            - If successful: (LLMCallResponse, None)
            - If failed: (None, error_message)
        """
        raise NotImplementedError("Providers must implement execute method")

    def get_provider_name(self) -> str:
        """Get the name of the provider.

        Returns:
            Provider name (e.g., "openai", "anthropic", "google")
        """
        return self.__class__.__name__.replace("Provider", "").lower()
