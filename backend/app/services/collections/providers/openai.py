import logging
import tempfile
from typing import IO, List, cast

from openai import OpenAI
from sqlmodel import Session

from app.services.collections.providers import BaseProvider
from app.core.cloud.storage import CloudStorage
from app.core.db import engine
from app.crud import DocumentCrud
from app.crud.rag import OpenAIVectorStoreCrud
from app.services.collections.helpers import get_service_name
from app.models import Collection, Document, ProviderType


logger = logging.getLogger(__name__)

OPENAI_PROVIDER = ProviderType.openai.value


class OpenAIProvider(BaseProvider):
    """OpenAI-specific collection provider for vector stores."""

    def __init__(self, client: OpenAI):
        super().__init__(client)
        self.client = client

    def get_existing_file_id(self, doc: Document) -> str | None:
        return (doc.file_id or {}).get(OPENAI_PROVIDER)

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
                with tempfile.NamedTemporaryFile() as tmp:
                    body = cast(IO[bytes], storage.stream(doc.object_store_url))
                    while chunk := body.read(1024 * 1024):
                        tmp.write(chunk)
                    if doc.file_size_kb is None:
                        doc.file_size_kb = round(tmp.tell() / 1024, 2)
                    tmp.seek(0)
                    uploaded = self.client.files.create(
                        file=(doc.fname, tmp), purpose="assistants"
                    )
            except Exception as err:
                logger.error(
                    "[OpenAIProvider.upload_files] Failed to upload file | doc_id=%s, error=%s",
                    doc.id,
                    str(err),
                    exc_info=True,
                )
                raise

            doc.file_id = {**(doc.file_id or {}), OPENAI_PROVIDER: uploaded.id}
            try:
                with Session(engine) as session:
                    document_crud = DocumentCrud(session, project_id)
                    db_doc = document_crud.read_one(doc.id)
                    db_doc.file_id = doc.file_id
                    db_doc.file_size_kb = doc.file_size_kb
                    document_crud.update(db_doc)
            except Exception as err:
                logger.error(
                    "[OpenAIProvider.upload_files] DB persistence failed, rolling back OpenAI file | "
                    "doc_id=%s, openai_file_id=%s, error=%s",
                    doc.id,
                    uploaded.id,
                    str(err),
                    exc_info=True,
                )
                try:
                    self.client.files.delete(uploaded.id)
                    logger.info(
                        "[OpenAIProvider.upload_files] Rolled back OpenAI file | "
                        "doc_id=%s, openai_file_id=%s",
                        doc.id,
                        uploaded.id,
                    )
                except Exception as delete_err:
                    logger.error(
                        "[OpenAIProvider.upload_files] Rollback failed, file is orphaned | "
                        "doc_id=%s, openai_file_id=%s, error=%s",
                        doc.id,
                        uploaded.id,
                        str(delete_err),
                    )
                doc.file_id.pop(OPENAI_PROVIDER, None)
                raise

    def create_vector_store(self) -> str:
        try:
            vector_store = OpenAIVectorStoreCrud(self.client).create()
            logger.info(
                "[OpenAIProvider.create_vector_store] Vector store created | vector_store_id=%s",
                vector_store.id,
            )
            return vector_store.id
        except Exception as e:
            logger.error(
                f"[OpenAIProvider.create_vector_store] Failed to create vector store: {str(e)}",
                exc_info=True,
            )
            raise

    def create(
        self,
        docs: List[Document],
        vector_store_id: str,
    ) -> Collection:
        try:
            if docs:
                OpenAIVectorStoreCrud(self.client).update(vector_store_id, docs)
                logger.info(
                    "[OpenAIProvider.create] Batch uploaded | vector_store_id=%s, doc_count=%d",
                    vector_store_id,
                    len(docs),
                )

            return Collection(  # pyright: ignore[reportCallIssue]
                knowledge_base_id=vector_store_id,
                knowledge_base_provider=get_service_name(ProviderType.openai),
            )

        except Exception as e:
            logger.error(
                f"[OpenAIProvider.create] Failed to create collection: {str(e)}",
                exc_info=True,
            )
            raise

    def delete(self, collection: Collection) -> None:
        try:
            OpenAIVectorStoreCrud(self.client).delete(collection.knowledge_base_id)
            logger.info(
                f"[OpenAIProvider.delete] Deleted vector store | vector_store_id={collection.knowledge_base_id}"
            )
        except Exception as e:
            logger.error(
                f"[OpenAIProvider.delete] Failed to delete vector store | "
                f"knowledge_base_id={collection.knowledge_base_id}, error={str(e)}",
                exc_info=True,
            )
            raise
