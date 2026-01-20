import logging

from openai import OpenAI

from app.services.collections.providers import BaseProvider
from app.crud import DocumentCrud
from app.core.cloud.storage import CloudStorage
from app.crud.rag import OpenAIVectorStoreCrud, OpenAIAssistantCrud
from app.services.collections.helpers import batch_documents, get_service_name
from app.models import CreationRequest, Collection


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI-specific collection provider for vector stores and assistants."""

    def __init__(self, client: OpenAI):
        super().__init__(client)
        self.client = client

    def create(
        self,
        collection_request: CreationRequest,
        storage: CloudStorage,
        document_crud: DocumentCrud,
    ) -> Collection:
        """
        Create OpenAI vector store with documents and optionally an assistant.
        """
        try:
            docs_batches = batch_documents(
                document_crud,
                collection_request.documents,
                collection_request.batch_size,
            )

            vector_store_crud = OpenAIVectorStoreCrud(self.client)
            vector_store = vector_store_crud.create()

            list(vector_store_crud.update(vector_store.id, storage, docs_batches))

            logger.info(
                "[OpenAIProvider.create] Vector store created | "
                f"vector_store_id={vector_store.id}, batches={len(docs_batches)}"
            )

            # Check if we need to create an assistant (based on assistant options in request)
            with_assistant = (
                collection_request.model is not None
                and collection_request.instructions is not None
            )
            if with_assistant:
                assistant_crud = OpenAIAssistantCrud(self.client)

                assistant_options = {
                    "model": collection_request.model,
                    "instructions": collection_request.instructions,
                    "temperature": collection_request.temperature,
                }
                filtered_options = {
                    k: v for k, v in assistant_options.items() if v is not None
                }

                assistant = assistant_crud.create(vector_store.id, **filtered_options)

                logger.info(
                    "[OpenAIProvider.create] Assistant created | "
                    f"assistant_id={assistant.id}, vector_store_id={vector_store.id}"
                )

                return Collection(
                    llm_service_id=assistant.id,
                    llm_service_name=filtered_options.get("model", "assistant"),
                )
            else:
                logger.info(
                    "[OpenAIProvider.create] Skipping assistant creation | with_assistant=False"
                )

                return Collection(
                    llm_service_id=vector_store.id,
                    llm_service_name=get_service_name("openai"),
                )

        except Exception as e:
            logger.error(
                f"[OpenAIProvider.create] Failed to create collection: {str(e)}",
                exc_info=True,
            )
            raise

    def delete(self, collection: Collection) -> None:
        """Delete OpenAI resources (assistant or vector store).

        Determines what to delete based on llm_service_name:
        - If assistant was created, delete the assistant (which also removes the vector store)
        - If only vector store was created, delete the vector store
        """
        try:
            if collection.llm_service_name != get_service_name("openai"):
                OpenAIAssistantCrud(self.client).delete(collection.llm_service_id)
                logger.info(
                    f"[OpenAIProvider.delete] Deleted assistant | assistant_id={collection.llm_service_id}"
                )
            else:
                OpenAIVectorStoreCrud(self.client).delete(collection.llm_service_id)
                logger.info(
                    f"[OpenAIProvider.delete] Deleted vector store | vector_store_id={collection.llm_service_id}"
                )
        except Exception as e:
            logger.error(
                f"[OpenAIProvider.delete] Failed to delete resource | "
                f"llm_service_id={collection.llm_service_id}, error={str(e)}",
                exc_info=True,
            )
            raise
