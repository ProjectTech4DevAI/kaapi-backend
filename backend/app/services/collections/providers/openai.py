import logging
from io import BytesIO
from typing import List

from openai import OpenAI
from sqlmodel import Session

from app.services.collections.providers import BaseProvider
from app.core.cloud.storage import CloudStorage
from app.core.db import engine
from app.crud import DocumentCrud
from app.crud.rag import OpenAIVectorStoreCrud, OpenAIAssistantCrud
from app.services.collections.helpers import get_service_name
from app.models import CreationRequest, Collection, Document


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI-specific collection provider for vector stores and assistants."""

    def __init__(self, client: OpenAI):
        super().__init__(client)
        self.client = client

    def get_existing_file_id(self, doc: Document) -> str | None:
        return doc.openai_file_id

    def upload_files(
        self,
        storage: CloudStorage,
        docs: list[Document],
        project_id: int,
    ) -> None:
        for doc in docs:
            if self.get_existing_file_id(doc):
                continue
            try:
                content = storage.get(doc.object_store_url)
                if doc.file_size_kb is None:
                    doc.file_size_kb = round(len(content) / 1024, 2)
                f_obj = BytesIO(content)
                f_obj.name = doc.fname
                uploaded = self.client.files.create(file=f_obj, purpose="assistants")
                doc.openai_file_id = uploaded.id
                with Session(engine) as session:
                    document_crud = DocumentCrud(session, project_id)
                    db_doc = document_crud.read_one(doc.id)
                    db_doc.openai_file_id = uploaded.id
                    db_doc.file_size_kb = doc.file_size_kb
                    document_crud.update(db_doc)
            except Exception as err:
                logger.error(
                    "[OpenAIProvider.upload_files] Failed to upload file | doc_id=%s, error=%s",
                    doc.id,
                    str(err),
                    exc_info=True,
                )
                raise

    def create(
        self,
        collection_request: CreationRequest,
        docs: List[Document],
        vector_store_id: str | None = None,
        is_final: bool = False,
    ) -> Collection:
        try:
            vector_store_crud = OpenAIVectorStoreCrud(self.client)

            if vector_store_id is None:
                vector_store = vector_store_crud.create()
                vector_store_id = vector_store.id
                logger.info(
                    "[OpenAIProvider.create] Vector store created | vector_store_id=%s",
                    vector_store_id,
                )

            if docs:
                vector_store_crud.update(vector_store_id, docs)
                logger.info(
                    "[OpenAIProvider.create] Batch uploaded | vector_store_id=%s, doc_count=%d",
                    vector_store_id,
                    len(docs),
                )

            if not is_final:
                return Collection(
                    llm_service_id=vector_store_id,
                    llm_service_name=get_service_name("openai"),
                )
            # if "is_final" is true then only will assistant creation happen -
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

                assistant = assistant_crud.create(vector_store_id, **filtered_options)

                logger.info(
                    "[OpenAIProvider.create] Assistant created | assistant_id=%s, vector_store_id=%s",
                    assistant.id,
                    vector_store_id,
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
                    llm_service_id=vector_store_id,
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
