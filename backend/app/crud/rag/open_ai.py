import json
import logging
import functools as ft
import time

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

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
                logger.warning(
                    f"[ResourceCleaner.call] OpenAI error during cleanup | {{'cleaner_type': '{self}', 'resource': '{resource}', 'error': '{str(err)}'}}",
                )

        logger.warning(
            f"[ResourceCleaner.call] Cleanup failure after retries | {{'cleaner_type': '{self}', 'resource': '{resource}'}}"
        )

    def clean(self, resource):
        raise NotImplementedError()


class VectorStoreCleaner(ResourceCleaner):
    def clean(self, resource):
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
        docs: list[Document],
    ) -> None:
        if not docs:
            return

        try:
            batch = self.client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=[],
                file_ids=[doc.openai_file_id for doc in docs],
            )
        except OpenAIError as err:
            logger.error(
                f"[OpenAIVectorStoreCrud.update] Batch attach failed | "
                f"{{'vector_store_id': '{vector_store_id}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise

        logger.info(
            f"[OpenAIVectorStoreCrud.update] Batch complete | "
            f"{{'vector_store_id': '{vector_store_id}', "
            f"'completed': {batch.file_counts.completed}, 'failed': {batch.file_counts.failed}}}"
        )
        if batch.file_counts.failed > 0:
            raise RuntimeError(
                f"Batch attach to vector store {vector_store_id!r} completed with "
                f"{batch.file_counts.failed} failed file(s) "
                f"({batch.file_counts.completed} succeeded)"
            )

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


class OpenAIFileCrud(OpenAICrud):
    def delete(self, file_id: str) -> None:
        logger.info(
            f"[OpenAIFileCrud.delete] Deleting OpenAI file | {{'file_id': '{file_id}'}}"
        )
        try:
            self.client.files.delete(file_id)
            logger.info(
                f"[OpenAIFileCrud.delete] OpenAI file deleted | {{'file_id': '{file_id}'}}"
            )
        except OpenAIError as err:
            logger.warning(
                f"[OpenAIFileCrud.delete] Failed to delete OpenAI file | {{'file_id': '{file_id}', 'error': '{str(err)}'}}"
            )
