import json
import logging
import functools as ft
from io import BytesIO
from typing import Iterable

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from app.core.cloud import CloudStorage
from app.models import Document

logger = logging.getLogger(__name__)


def vs_ls(client: OpenAI, vector_store_id: str):
    kwargs = {}
    while True:
        page = client.vector_stores.files.list(
            vector_store_id=vector_store_id,
            **kwargs,
        )
        yield from page
        if not page.has_more:
            break
        kwargs["after"] = page.last_id


class BaseModelEncoder(json.JSONEncoder):
    @ft.singledispatchmethod
    def default(self, o):
        return super().default(o)

    @default.register
    def _(self, o: BaseModel):
        return o.model_dump()


class ResourceCleaner:
    def __init__(self, client):
        self.client = client

    def __str__(self):
        return type(self).__name__

    def __call__(self, resource, retries=1):
        logger.info(
            f"[ResourceCleaner.call] Starting resource cleanup | {{'cleaner_type': '{self}', 'resource': '{resource}', 'retries': {retries}}}"
        )
        for i in range(retries):
            try:
                self.clean(resource)
                logger.info(
                    f"[ResourceCleaner.call] Resource cleaned successfully | {{'cleaner_type': '{self}', 'resource': '{resource}'}}"
                )
                return
            except OpenAIError as err:
                logger.error(
                    f"[ResourceCleaner.call] OpenAI error during cleanup | {{'cleaner_type': '{self}', 'resource': '{resource}', 'error': '{str(err)}'}}",
                    exc_info=True,
                )

        logger.warning(
            f"[ResourceCleaner.call] Cleanup failure after retries | {{'cleaner_type': '{self}', 'resource': '{resource}'}}"
        )

    def clean(self, resource):
        raise NotImplementedError()


class AssistantCleaner(ResourceCleaner):
    def clean(self, resource):
        logger.info(
            f"[AssistantCleaner.clean] Deleting assistant | {{'assistant_id': '{resource}'}}"
        )
        self.client.beta.assistants.delete(resource)


class VectorStoreCleaner(ResourceCleaner):
    def clean(self, resource):
        logger.info(
            f"[VectorStoreCleaner.clean] Starting vector store cleanup | {{'vector_store_id': '{resource}'}}"
        )
        for i in vs_ls(self.client, resource):
            self.client.files.delete(i.id)
        logger.info(
            f"[VectorStoreCleaner.clean] Deleting vector store | {{'vector_store_id': '{resource}'}}"
        )
        self.client.vector_stores.delete(resource)


class OpenAICrud:
    def __init__(self, client):
        if client is None:
            logger.error("[OpenAICrud] OpenAI client is not configured")
            raise ValueError("OpenAI client is not configured")

        self.client = client


class OpenAIVectorStoreCrud(OpenAICrud):
    def create(self):
        logger.info(
            f"[OpenAIVectorStoreCrud.create] Creating vector store | {{'action': 'create'}}"
        )
        vector_store = self.client.vector_stores.create()
        logger.info(
            f"[OpenAIVectorStoreCrud.create] Vector store created | {{'vector_store_id': '{vector_store.id}'}}"
        )
        return vector_store

    def read(self, vector_store_id: str):
        logger.info(
            f"[OpenAIVectorStoreCrud.read] Reading files from vector store | {{'vector_store_id': '{vector_store_id}'}}"
        )
        yield from vs_ls(self.client, vector_store_id)

    def update(
        self,
        vector_store_id: str,
        storage: CloudStorage,
        documents: Iterable[Document],
    ):
        for docs in documents:
            files = []
            for d in docs:
                # Get file bytes and wrap in BytesIO for OpenAI API
                content = storage.get(d.object_store_url)
                f_obj = BytesIO(content)
                f_obj.name = d.fname
                files.append(f_obj)

            logger.info(
                f"[OpenAIVectorStoreCrud.update] Uploading files to vector store | {{'vector_store_id': '{vector_store_id}', 'file_count': {len(files)}}}"
            )
            req = self.client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=files,
            )
            logger.info(
                f"[OpenAIVectorStoreCrud.update] File upload completed | {{'vector_store_id': '{vector_store_id}', 'completed_files': {req.file_counts.completed}, 'total_files': {req.file_counts.total}}}"
            )
            if req.file_counts.completed != req.file_counts.total:
                error_msg = f"OpenAI document processing error: {req.file_counts.completed}/{req.file_counts.total} files completed"
                logger.error(
                    f"[OpenAIVectorStoreCrud.update] Document processing error | {{'vector_store_id': '{vector_store_id}', 'completed_files': {req.file_counts.completed}, 'total_files': {req.file_counts.total}}}"
                )
                raise InterruptedError(error_msg)

            yield from docs

    def update_batch(
        self,
        vector_store_id: str,
        storage: CloudStorage,
        docs: list[Document],
    ) -> tuple[list[Document], list[Document]]:
        """
        Upload a single batch of documents to the vector store with partial failure handling.

        Reuses provider_file_id if the document was already uploaded.
        Returns (succeeded, failed) — failed docs should be retried in the next batch.
        """
        succeeded: list[Document] = []
        failed: list[Document] = []

        reuse_docs = [doc for doc in docs if doc.provider_file_id]
        new_docs = [doc for doc in docs if not doc.provider_file_id]

        # Attach already-uploaded files in one batch call
        if reuse_docs:
            try:
                batch = self.client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=[],
                    file_ids=[doc.provider_file_id for doc in reuse_docs],
                )
                logger.info(
                    f"[OpenAIVectorStoreCrud.update_batch] Reuse batch complete | "
                    f"{{'vector_store_id': '{vector_store_id}', 'completed': {batch.file_counts.completed}, 'failed': {batch.file_counts.failed}}}"
                )
                if batch.file_counts.failed == 0:
                    succeeded.extend(reuse_docs)
                else:
                    # Can't identify which specific files failed — retry all of them
                    logger.warning(
                        f"[OpenAIVectorStoreCrud.update_batch] Reuse batch had failures, marking all for retry | "
                        f"{{'vector_store_id': '{vector_store_id}', 'failed_count': {batch.file_counts.failed}}}"
                    )
                    failed.extend(reuse_docs)
            except OpenAIError as err:
                logger.error(
                    f"[OpenAIVectorStoreCrud.update_batch] Reuse batch request failed | "
                    f"{{'vector_store_id': '{vector_store_id}', 'error': '{str(err)}'}}",
                    exc_info=True,
                )
                failed.extend(reuse_docs)

        # Upload new files individually so we can track provider_file_id per doc
        for idx, doc in enumerate(new_docs, start=1):
            logger.info(
                f"[OpenAIVectorStoreCrud.update_batch] Uploading file {idx}/{len(new_docs)} | "
                f"{{'vector_store_id': '{vector_store_id}', 'doc_id': '{doc.id}', 'fname': '{doc.fname}'}}"
            )
            try:
                content = storage.get(doc.object_store_url)
                f_obj = BytesIO(content)
                f_obj.name = doc.fname

                uploaded_file = self.client.files.create(
                    file=f_obj, purpose="assistants"
                )
                doc.provider_file_id = uploaded_file.id

                del content, f_obj

                self.client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=uploaded_file.id,
                )
                logger.info(
                    f"[OpenAIVectorStoreCrud.update_batch] File uploaded and attached | "
                    f"{{'vector_store_id': '{vector_store_id}', 'doc_id': '{doc.id}', 'provider_file_id': '{uploaded_file.id}'}}"
                )
                succeeded.append(doc)
            except Exception as err:
                logger.error(
                    f"[OpenAIVectorStoreCrud.update_batch] Failed to upload file | "
                    f"{{'vector_store_id': '{vector_store_id}', 'doc_id': '{doc.id}', 'error': '{str(err)}'}}",
                    exc_info=True,
                )
                failed.append(doc)

        logger.info(
            f"[OpenAIVectorStoreCrud.update_batch] Batch complete | "
            f"{{'vector_store_id': '{vector_store_id}', 'succeeded': {len(succeeded)}, 'failed': {len(failed)}}}"
        )
        return succeeded, failed

    def delete(self, vector_store_id: str, retries: int = 3):
        if retries < 1:
            try:
                raise ValueError("Retries must be greater-than 1")
            except ValueError as err:
                logger.error(
                    f"[OpenAIVectorStoreCrud.delete] Invalid retries value | {{'vector_store_id': '{vector_store_id}', 'retries': {retries}}}",
                    exc_info=True,
                )
                raise

        cleaner = VectorStoreCleaner(self.client)
        cleaner(vector_store_id)
        logger.info(
            f"[OpenAIVectorStoreCrud.delete] Vector store deleted | {{'vector_store_id': '{vector_store_id}'}}"
        )


class OpenAIAssistantCrud(OpenAICrud):
    def create(self, vector_store_id: str, **kwargs):
        logger.info(
            f"[OpenAIAssistantCrud.create] Creating assistant | {{'vector_store_id': '{vector_store_id}'}}"
        )
        assistant = self.client.beta.assistants.create(
            tools=[
                {
                    "type": "file_search",
                }
            ],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [
                        vector_store_id,
                    ],
                },
            },
            **kwargs,
        )
        logger.info(
            f"[OpenAIAssistantCrud.create] Assistant created | {{'assistant_id': '{assistant.id}', 'vector_store_id': '{vector_store_id}'}}"
        )
        return assistant

    def delete(self, assistant_id: str):
        logger.info(
            f"[OpenAIAssistantCrud.delete] Starting assistant deletion | {{'assistant_id': '{assistant_id}'}}"
        )
        assistant = self.client.beta.assistants.retrieve(assistant_id)
        vector_stores = assistant.tool_resources.file_search.vector_store_ids

        try:
            (vector_store_id,) = vector_stores
        except ValueError:
            if vector_stores:
                names = ", ".join(vector_stores)
                err = ValueError(f"Too many attached vector stores: {names}")
            else:
                err = ValueError("No vector stores found")

            logger.error(
                f"[OpenAIAssistantCrud.delete] Invalid vector store state | {{'assistant_id': '{assistant_id}', 'vector_stores': '{vector_stores}'}}",
                exc_info=True,
            )
            raise err

        v_crud = OpenAIVectorStoreCrud(self.client)
        v_crud.delete(vector_store_id)

        cleaner = AssistantCleaner(self.client)
        cleaner(assistant_id)
        logger.info(
            f"[OpenAIAssistantCrud.delete] Assistant deleted | {{'assistant_id': '{assistant_id}'}}"
        )
