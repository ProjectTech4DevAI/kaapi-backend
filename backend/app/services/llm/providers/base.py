"""Base provider interface for LLM providers.

This module defines the abstract base class that all LLM providers must implement.
It provides a provider-agnostic interface for executing LLM calls.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import model_validator
from sqlmodel import SQLModel

from app.models.llm import NativeCompletionConfig, LLMCallResponse, QueryParams
from app.models.llm.request import TextContent, ImageContent, PDFContent


class MultiModalInput(SQLModel):
    """Resolved multimodal input containing a list of content parts."""

    parts: list[TextContent | ImageContent | PDFContent]

    @model_validator(mode="after")
    def validate_parts(self):
        if not self.parts:
            raise ValueError("MultiModalInput requires at least one content part")
        return self


COMPLETION_TYPE_ALLOWED_INPUT: dict[str, set[type]] = {
    "text": {str},
    "stt": {str},
    "tts": {str},
    "image": {list},
    "pdf": {list},
    "multimodal": {MultiModalInput},
}


def validate_completion_input(completion_type: str, resolved_input: Any) -> str | None:
    """Returns error message if mismatch, else None."""
    allowed = COMPLETION_TYPE_ALLOWED_INPUT.get(completion_type)
    if allowed is None:
        return f"Unknown completion type: '{completion_type}'"
    if type(resolved_input) not in allowed:
        expected = " or ".join(t.__name__ for t in allowed)
        return (
            f"completion type '{completion_type}' expects {expected} input, "
            f"got {type(resolved_input).__name__}"
        )
    return None


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

    @staticmethod
    @abstractmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        """
        Static method to instantiate a client instance of the provider
        """
        raise NotImplementedError("Providers must implement create_client method")

    @abstractmethod
    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | list[TextContent | ImageContent | PDFContent],
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute LLM API call.

        Directly passes the user's config params to provider API along with input.

        Args:
            completion_config: LLM completion configuration, pass params as-is to provider API
            query: Query parameters including input and conversation_id
            resolved_input: The resolved input content (text string or file path for audio)
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
