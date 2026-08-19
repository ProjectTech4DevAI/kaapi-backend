"""Base provider interface for object-storage bucket providers."""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseBucketProvider(ABC):
    """Abstract base class for bucket providers."""

    # URI scheme this provider handles (e.g. "gs", "s3").
    SCHEME: str = ""

    # Cap on signed-URL lifetime (24h), matching AmazonCloudStorage.
    MAX_SIGNED_URL_EXPIRY: int = 86400

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    @abstractmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        """Instantiate a storage client from decrypted credentials."""
        raise NotImplementedError("Bucket providers must implement create_client")

    @abstractmethod
    def get_signed_url(self, uri: str, expires_in: int = 3600) -> str:
        """Generate a time-limited signed URL for a single object."""
        raise NotImplementedError("Bucket providers must implement get_signed_url")

    def get_bulk_signed_urls(
        self, uris: list[str], expires_in: int = 3600
    ) -> dict[str, str]:
        """Sign many object URIs, reusing this provider's single client."""
        signed: dict[str, str] = {}
        for uri in uris:
            signed[uri] = self.get_signed_url(uri, expires_in=expires_in)
        return signed

    def to_public_url(self, uri: str, expires_in: int = 3600) -> str:
        """Resolve a private object URI to a fetchable signed URL."""
        return self.get_signed_url(uri, expires_in=expires_in)

    def get_provider_name(self) -> str:
        """Return the provider name derived from the class name."""
        return self.__class__.__name__.replace("BucketProvider", "").lower()
