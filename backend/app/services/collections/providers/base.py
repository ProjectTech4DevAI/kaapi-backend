from abc import ABC, abstractmethod
from typing import Any

from app.core.cloud.storage import CloudStorage
from app.models import Collection, Document


class BaseProvider(ABC):
    """Abstract base class for collection providers.

    All provider implementations (OpenAI, Bedrock, etc.) must inherit from
    this class and implement the required methods.

    Providers handle creation of vector store collections.

    Attributes:
        client: The provider-specific client instance
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    @abstractmethod
    def upload_files(
        self,
        storage: CloudStorage,
        docs: list[Document],
        project_id: int,
    ) -> None:
        """Upload all documents to the provider's file storage and persist their file IDs.

        Args:
            storage: Cloud storage instance to fetch raw file bytes from
            docs: Documents to upload
            project_id: Project ID used to persist the provider file IDs to the DB
        """
        raise NotImplementedError("Providers must implement upload_files method")

    @abstractmethod
    def create(
        self,
        docs: list[Document],
        vector_store_id: str | None = None,
    ) -> Collection:
        """Upload docs batch to vector store (creating it if vector_store_id is None).
        Returns Collection with llm_service_id set to the vector store ID."""
        raise NotImplementedError("Providers must implement create method")

    @abstractmethod
    def delete(self, collection: Collection) -> None:
        """Delete remote resources associated with a collection."""
        raise NotImplementedError("Providers must implement delete method")

    def get_existing_file_id(self, _doc: Document) -> str | None:
        """Return the already-uploaded file ID for this provider, or None to trigger upload."""
        return None

    def get_provider_name(self) -> str:
        return self.__class__.__name__.replace("Provider", "").lower()
