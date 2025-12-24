from abc import ABC, abstractmethod
from typing import Any

from app.crud import DocumentCrud
from app.core.cloud.storage import CloudStorage
from app.models import CreationRequest, CreateCollectionResult, Collection


class BaseProvider(ABC):
    """Abstract base class for collection providers.

    All provider implementations (OpenAI, Bedrock, etc.) must inherit from
    this class and implement the required methods.

    Providers handle creation of knowledge bases (vector stores) and
    optional assistant/agent creation backed by those knowledge bases.

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
    def create(
        self,
        collection_request: CreationRequest,
        storage: CloudStorage,
        document_crud: DocumentCrud,
    ) -> CreateCollectionResult:
        """Create collection with documents and optionally an assistant.

        Args:
            collection_params: Collection parameters (name, description, chunking_params, etc.)
            storage: Cloud storage instance for file access
            document_crud: DocumentCrud instance for fetching documents
            batch_size: Number of documents to process per batch
            with_assistant: Whether to create an assistant/agent
            assistant_options: Options for assistant creation (provider-specific)

        Returns:
            CreateCollectionresult containing:
            - llm_service_id: ID of the created resource (vector store or assistant)
            - llm_service_name: Name of the service
            - kb_blob: All collection params except documents
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

    @abstractmethod
    def cleanup(self, collection_result: CreateCollectionResult) -> None:
        """Clean up/rollback resources created during execute.

        Called when collection creation fails and remote resources need to be deleted.

        Args:
            collection_result: The CreateCollectionresult returned from execute, containing resource IDs
        """
        raise NotImplementedError("Providers must implement cleanup method")

    def get_provider_name(self) -> str:
        """Get the name of the provider.

        Returns:
            Provider name (e.g., "openai", "bedrock", "pinecone")
        """
        return self.__class__.__name__.replace("Provider", "").lower()
