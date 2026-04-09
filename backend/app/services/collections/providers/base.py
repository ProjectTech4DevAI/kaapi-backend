from abc import ABC, abstractmethod
from typing import Any

from app.core.cloud.storage import CloudStorage
from app.models import CreationRequest, Collection, Document


class BaseProvider(ABC):
    """Abstract base class for collection providers.

    All provider implementations (OpenAI, Bedrock, etc.) must inherit from
    this class and implement the required methods.

    Providers handle creation of collection and
    optional assistant/agent creation backed by those collections.

    Attributes:
        client: The provider-specific client instance
    """

    def __init__(self, client: Any) -> None:
        """Initialize provider with client.

        Args:
            client: Provider-specific client instance
        """
        self.client = client

    @abstractmethod
    def create(
        self,
        collection_request: CreationRequest,
        storage: CloudStorage,
        docs_batches: list[list[Document]],
    ) -> Collection:
        """Create collection with documents and optionally an assistant.

        Args:
            collection_request: Collection parameters (name, description, document list, etc.)
            storage: Cloud storage instance for file access
            docs_batches: Pre-fetched document batches (DB reads must happen before this call)

        Returns:
            llm_service_id: ID of the resource to delete
            llm_service_name: Name of the service (determines resource type)
        """
        raise NotImplementedError("Providers must implement execute method")

    @abstractmethod
    def delete(self, collection: Collection) -> None:
        """Delete remote resources associated with a collection.

        Called when a collection is being deleted and remote resources need to be cleaned up.

        Args:
            llm_service_id: ID of the resource to delete
            llm_service_name: Name of the service (determines resource type)
        """
        raise NotImplementedError("Providers must implement delete method")

    def get_provider_name(self) -> str:
        """Get the name of the provider.

        Returns:
            Provider name (e.g., "openai", "bedrock", "pinecone")
        """
        return self.__class__.__name__.replace("Provider", "").lower()
