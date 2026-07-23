import json
import logging
import functools as ft

import openai
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from app.models import Document, ProviderType

logger = logging.getLogger(__name__)

OPENAI_PROVIDER = ProviderType.openai.value


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

        logger.info(
            f"[OpenAIVectorStoreCrud.update] Uploading files to vector store | "
            f"{{'vector_store_id': '{vector_store_id}', 'file_count': {len(docs)}}}"
        )

        try:
            created = self.client.vector_stores.file_batches.create(
                vector_store_id=vector_store_id,
                file_ids=[doc.file_id[OPENAI_PROVIDER] for doc in docs],
            )
            # poll()'s return deserializes a vector-store body, so its .id is the vs_ id;
            # capture the real vsfb_ batch id from create() before polling.
            batch_id = created.id
            batch = self.client.vector_stores.file_batches.poll(
                batch_id, vector_store_id=vector_store_id
            )
        except openai.RateLimitError as e:
            error_message = (
                f"[OPENAI] Rate limit exceeded (code: {e.status_code}): "
                f"{e.message}. Try again in 1 minute. If issue persists, "
                f"contact Kaapi."
            )
            logger.warning(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}, file_count={len(docs)}",
                exc_info=True,
            )
            raise InterruptedError(error_message)
        except openai.AuthenticationError as e:
            error_message = (
                f"[OPENAI] Authentication failed (code: {e.status_code}): "
                f"{e.message}. Check your OpenAI API key is valid and "
                f"has not expired."
            )
            logger.warning(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}",
                exc_info=True,
            )
            raise InterruptedError(error_message)
        except openai.NotFoundError as e:
            error_message = (
                f"[OPENAI] Resource not found (code: {e.status_code}): "
                f"{e.message}. Verify the vector store ID exists and "
                f"hasn't been deleted."
            )
            logger.warning(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}",
                exc_info=True,
            )
            raise InterruptedError(error_message)
        except openai.BadRequestError as e:
            error_message = (
                f"[OPENAI] Bad request (code: {e.status_code}): "
                f"{e.message}. Review the file payload and metadata; the "
                f"request may be malformed."
            )
            logger.warning(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}, file_count={len(docs)}",
                exc_info=True,
            )
            raise InterruptedError(error_message)
        except openai.UnprocessableEntityError as e:
            error_message = (
                f"[OPENAI] Unprocessable entity (code: {e.status_code}): "
                f"{e.message}. The uploaded files may be in an "
                f"unsupported format or exceed size limits."
            )
            logger.warning(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}, file_count={len(docs)}",
                exc_info=True,
            )
            raise InterruptedError(error_message)
        except openai.InternalServerError as e:
            error_message = (
                f"[OPENAI] Server error (code: {e.status_code}): "
                f"{e.message}. This is usually transient — retry in a "
                f"few seconds. If issue persists, contact Kaapi."
            )
            logger.error(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}",
                exc_info=True,
            )
            raise InterruptedError(error_message)
        except openai.APITimeoutError as e:
            error_message = (
                f"[KAAPI] OpenAI request timed out (code: "
                f"{type(e).__name__}): {e}. Retry the upload, or split "
                f"the batch into smaller chunks."
            )
            logger.error(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}, file_count={len(docs)}",
                exc_info=True,
            )
            raise InterruptedError(error_message)
        except openai.OpenAIError as e:
            error_message = f"[OPENAI] SDK error: {e}. If this persists, contact Kaapi."
            logger.warning(
                f"[OpenAIVectorStoreCrud.update] {error_message} | "
                f"vector_store_id={vector_store_id}",
                exc_info=True,
            )
            raise InterruptedError(error_message)

        logger.info(
            f"[OpenAIVectorStoreCrud.update] Batch complete | "
            f"{{'vector_store_id': '{vector_store_id}', 'batch_id': '{batch_id}', "
            f"'completed': {batch.file_counts.completed}, 'failed': {batch.file_counts.failed}}}"
        )
        if batch.file_counts.failed > 0:
            try:
                failed_files = self.client.vector_stores.file_batches.list_files(
                    vector_store_id=vector_store_id,
                    batch_id=batch_id,
                    filter="failed",
                )
                doc_by_file_id = {d.file_id[OPENAI_PROVIDER]: d for d in docs}
                parts = []
                for f in failed_files:
                    d = doc_by_file_id.get(f.id)
                    label = d.fname if d else f.id
                    msg = f.last_error.message if f.last_error else "no error detail"
                    parts.append(f"{label}: {msg}")
                logger.error(
                    f"[OpenAIVectorStoreCrud.update] Files failed to index | "
                    f"{{'batch_id': '{batch_id}', 'failed_files': '{', '.join(parts)}'}}"
                )
                raise RuntimeError("; ".join(parts))
            except OpenAIError as err:
                logger.warning(
                    f"[OpenAIVectorStoreCrud.update] Could not fetch per-file errors | "
                    f"{{'batch_id': '{batch_id}', 'error': '{str(err)}'}}"
                )
                raise

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
