"""Base provider interface for LLM providers.

This module defines the abstract base class that all LLM providers must implement.
It provides a provider-agnostic interface for executing LLM calls.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import model_validator
from sqlmodel import SQLModel

from app.models.llm import NativeCompletionConfig, LLMCallResponse, QueryParams
from app.models.llm.request import TextContent, AudioContent, ImageContent, PDFContent

MULTIMODAL_ALLOWED_PARTS = (TextContent, ImageContent, PDFContent)


class MultiModalInput(SQLModel):
    """Resolved multimodal input containing a list of content parts."""

    parts: list[TextContent | ImageContent | PDFContent]

    @model_validator(mode="after")
    def validate_parts(self):
        if not self.parts:
            raise ValueError("MultiModalInput requires at least one content part")
        return self


CONTENT_TYPE_LABEL: dict[type, str] = {
    TextContent: "text",
    AudioContent: "audio",
    ImageContent: "image",
    PDFContent: "pdf",
}

INPUT_TYPE_LABEL: dict[type, str] = {
    str: "text",
    list: "list",
    MultiModalInput: "multimodal (mixed input types)",
}

COMPLETION_TYPE_RULES: dict[str, dict] = {
    "text": {"type": str, "label": "text"},
    "stt": {"type": str, "label": "audio"},
    "tts": {"type": str, "label": "text"},
    "image": {"type": list, "element_type": ImageContent, "label": "image"},
    "pdf": {"type": list, "element_type": PDFContent, "label": "pdf"},
    "multimodal": {"type": MultiModalInput, "label": "multimodal"},
}


def _get_content_label(content: Any) -> str:
    return CONTENT_TYPE_LABEL.get(type(content), type(content).__name__)


def validate_completion_input(completion_type: str, resolved_input: Any) -> str | None:
    """Returns error message if input type doesn't match completion type, else None."""
    rule = COMPLETION_TYPE_RULES.get(completion_type)
    if rule is None:
        return f"Unknown completion type: '{completion_type}'"

    expected_type = rule["type"]
    label = rule["label"]

    if not isinstance(resolved_input, expected_type):
        actual_label = INPUT_TYPE_LABEL.get(
            type(resolved_input), type(resolved_input).__name__
        )
        hint = (
            " Please set completion type to 'multimodal' when sending mixed input types."
            if isinstance(resolved_input, MultiModalInput)
            else f" Please ensure the input type matches the completion type."
        )
        return (
            f"Input type mismatch: completion type '{completion_type}' expects "
            f"'{label}' input, but received {actual_label}.{hint}"
        )

    if isinstance(resolved_input, list):
        element_type = rule.get("element_type")
        if element_type:
            for item in resolved_input:
                if not isinstance(item, element_type):
                    return (
                        f"Input type mismatch: completion type '{completion_type}' expects "
                        f"'{label}' input, but received '{_get_content_label(item)}' content. "
                        f"Please ensure the input type matches the completion type."
                    )

    if isinstance(resolved_input, MultiModalInput):
        for part in resolved_input.parts:
            if not isinstance(part, MULTIMODAL_ALLOWED_PARTS):
                return (
                    f"Unsupported content in multimodal input: '{_get_content_label(part)}'. "
                    f"Multimodal supports text, image, and pdf only. Audio is not supported."
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
