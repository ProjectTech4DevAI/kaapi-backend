"""Gemini client wrapper for STT evaluation."""

import logging
from typing import Any

from google import genai
from sqlmodel import Session

from app.core.exception_handlers import HTTPException
from app.crud.credentials import get_provider_credential

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
            provider="google",
        )

        if not credentials:
            logger.error(
                f"[from_credentials] Gemini credentials not found | "
                f"org_id: {org_id}, project_id: {project_id}"
            )
            raise HTTPException(
                status_code=404,
                detail="Gemini credentials not configured for this project",
            )

        api_key = credentials.get("api_key")
        if not api_key:
            logger.error(
                f"[from_credentials] Invalid Gemini credentials (missing api_key) | "
                f"org_id: {org_id}, project_id: {project_id}"
            )
            raise GeminiClientError("Invalid Gemini credentials: missing api_key")

        logger.info(
            f"[from_credentials] Gemini client created successfully | "
            f"org_id: {org_id}, project_id: {project_id}"
        )
        return cls(api_key=api_key)
