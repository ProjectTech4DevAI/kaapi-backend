"""Gemini client wrapper for credential management."""

import logging
from typing import Any

from google import genai
from sqlmodel import Session

from fastapi import HTTPException
from app.crud.credentials import get_provider_credential

from .base import BatchProvider

logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    """Exception raised for Gemini client errors."""

    pass


class GeminiClient:
    """Wrapper for Google GenAI client with credential management."""

    def __init__(self, api_key: str) -> None:
        """Initialize Gemini client with API key.

        Args:
            api_key: Google AI API key
        """
        self._api_key = api_key
        self._client = genai.Client(api_key=api_key)

    @property
    def client(self) -> genai.Client:
        """Get the underlying GenAI client."""
        return self._client

    @classmethod
    def from_credentials(
        cls,
        session: Session,
        org_id: int,
        project_id: int,
    ) -> "GeminiClient":
        """Create client from stored credentials.

        Args:
            session: Database session
            org_id: Organization ID
            project_id: Project ID

        Returns:
            GeminiClient: Configured Gemini client

        Raises:
            HTTPException: If credentials not found
            GeminiClientError: If credentials are invalid
        """
        logger.info(
            f"[from_credentials] Fetching Gemini credentials | "
            f"org_id: {org_id}, project_id: {project_id}"
        )

        credentials = get_provider_credential(
            session=session,
            org_id=org_id,
            project_id=project_id,
            provider="google-aistudio",
        )

        if not credentials:
            logger.warning(
                f"[from_credentials] Gemini credentials not found | "
                f"org_id: {org_id}, project_id: {project_id}"
            )
            raise HTTPException(
                status_code=404,
                detail="Gemini credentials not configured for this project",
            )

        api_key = credentials.get("api_key")
        if not api_key:
            logger.warning(
                f"[from_credentials] Invalid Gemini credentials (missing api_key) | "
                f"org_id: {org_id}, project_id: {project_id}"
            )
            raise GeminiClientError("Invalid Gemini credentials: missing api_key")

        logger.info(
            f"[from_credentials] Gemini client created successfully | "
            f"org_id: {org_id}, project_id: {project_id}"
        )
        return cls(api_key=api_key)


# Providers whose batch jobs run on Vertex AI (GCS-backed) rather than AI-Studio.
VERTEX_BATCH_PROVIDERS = {"google-gcp", "google-gcp-native"}


def is_vertex_batch_provider(provider_name: str) -> bool:
    """Whether a provider name routes batch jobs through Vertex AI (vs AI-Studio)."""
    return provider_name in VERTEX_BATCH_PROVIDERS


def get_gemini_batch_provider(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    provider_name: str,
    model: str | None = None,
) -> BatchProvider:
    """Resolve a Gemini-family batch provider from the provider name.

    ``google-gcp`` routes to Vertex (GCS-backed); ``google``/``google-aistudio``
    to AI-Studio (File API). Both honor the same BatchProvider contract, so any
    service (assessment, evaluations, STT/TTS) can call this single resolver.
    """
    from .gemini import GeminiBatchProvider
    from .vertex import VertexBatchProvider

    if is_vertex_batch_provider(provider_name):
        cred = get_provider_credential(
            session=session,
            provider="google-gcp",
            project_id=project_id,
            org_id=organization_id,
        )
        if not cred:
            raise HTTPException(
                status_code=404,
                detail="google-gcp credentials not configured for this project",
            )
        if not isinstance(cred, dict):
            raise GeminiClientError("Expected decrypted google-gcp credentials dict")
        return VertexBatchProvider.from_credentials(cred, model=model)

    gemini = GeminiClient.from_credentials(
        session=session, org_id=organization_id, project_id=project_id
    )
    return GeminiBatchProvider(client=gemini.client, model=model)
