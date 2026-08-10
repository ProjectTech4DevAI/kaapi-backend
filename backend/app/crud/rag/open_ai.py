import json
import logging
import time
import functools as ft
from typing import Any, Iterator

import openai
from openai import OpenAI, OpenAIError
from openai.types import VectorStore
from openai.types.vector_stores import VectorStoreFile, VectorStoreFileBatch
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.models import Document, ProviderType

logger = logging.getLogger(__name__)

OPENAI_PROVIDER = ProviderType.openai.value

# Under the Celery soft time limit so a hung call can't eat the whole task window.
# SDK-level retries are off (registry.py: max_retries=0); tenacity is the sole
# retry layer, wrapping batch create+index.
OPENAI_TIMEOUT_SECONDS = 30

BATCH_POLL_INTERVAL_SECONDS = 2

# Retry batch create+index on any OpenAI/indexing failure, exponential backoff
# (~2s, 4s, 8s), all inside one Celery soft-time-limit window.
BATCH_INDEX_MAX_ATTEMPTS = 4
BATCH_RETRY_BACKOFF_BASE_SECONDS = 2


def _log_batch_retry(retry_state: RetryCallState) -> None:
    logger.warning(
        f"[OpenAIVectorStoreCrud._create_and_index_batch] Batch attempt failed, retrying | "
        f"attempt={retry_state.attempt_number}, "
        f"error={retry_state.outcome.exception() if retry_state.outcome else None}"
    )


def vs_ls(client: OpenAI, vector_store_id: str) -> Iterator[VectorStoreFile]:
    # SyncCursorPage auto-paginates on iteration (it has no `last_id`).
    yield from client.vector_stores.files.list(vector_store_id=vector_store_id)


def _provider_file_id(doc: Document) -> str:
    if not doc.file_id or OPENAI_PROVIDER not in doc.file_id:
        raise RuntimeError(
            f"Document {doc.id} has no OpenAI file id; upload it before indexing."
        )
    return doc.file_id[OPENAI_PROVIDER]


class BaseModelEncoder(json.JSONEncoder):
    @ft.singledispatchmethod
    def default(
        self, o: Any
    ) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().default(o)

    @default.register
    def _(self, o: BaseModel) -> Any:
        return o.model_dump()


class ResourceCleaner:
    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def __str__(self) -> str:
        return type(self).__name__

    def __call__(self, resource: str, retries: int = 1) -> None:
        logger.info(
            f"[ResourceCleaner.call] Starting resource cleanup | {{'cleaner_type': '{self}', 'resource': '{resource}', 'retries': {retries}}}"
        )
        for _ in range(retries):
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

    def clean(self, resource: str) -> None:
        raise NotImplementedError()


class VectorStoreCleaner(ResourceCleaner):
    def clean(self, resource: str) -> None:
        logger.info(
            f"[VectorStoreCleaner.clean] Deleting vector store | {{'vector_store_id': '{resource}'}}"
        )
        self.client.vector_stores.delete(resource)


class OpenAICrud:
    def __init__(self, client: OpenAI) -> None:
        if client is None:  # pyright: ignore[reportUnnecessaryComparison]
            logger.error("[OpenAICrud] OpenAI client is not configured")
            raise ValueError("OpenAI client is not configured")

        self.client = client


class OpenAIVectorStoreCrud(OpenAICrud):
    def create(self) -> VectorStore:
        logger.info(
            f"[OpenAIVectorStoreCrud.create] Creating vector store | {{'action': 'create'}}"
        )
        vector_store = self.client.vector_stores.create()
        logger.info(
            f"[OpenAIVectorStoreCrud.create] Vector store created | {{'vector_store_id': '{vector_store.id}'}}"
        )
        return vector_store

    def read(self, vector_store_id: str) -> Iterator[VectorStoreFile]:
        logger.info(
            f"[OpenAIVectorStoreCrud.read] Reading files from vector store | {{'vector_store_id': '{vector_store_id}'}}"
        )
        yield from vs_ls(self.client, vector_store_id)

    def _create_file_batch(self, vector_store_id: str, file_ids: list[str]) -> str:
        """Returns the vsfb_ id. poll()'s return deserializes a vector-store body,
        so its .id is the vs_ id - take the batch id from create()."""
        created = self.client.vector_stores.file_batches.create(
            vector_store_id=vector_store_id,
            file_ids=file_ids,
        )
        return created.id

    def _retrieve_file_batch(
        self, batch_id: str, vector_store_id: str
    ) -> VectorStoreFileBatch:
        return self.client.vector_stores.file_batches.retrieve(
            batch_id, vector_store_id=vector_store_id
        )

    def _poll_file_batch(
        self, batch_id: str, vector_store_id: str
    ) -> VectorStoreFileBatch:
        """Poll until indexing finishes; the Celery soft time limit is the deadline."""
        while True:
            batch = self._retrieve_file_batch(batch_id, vector_store_id)
            if batch.status != "in_progress":
                return batch
            time.sleep(BATCH_POLL_INTERVAL_SECONDS)

    def _raise_if_batch_incomplete(
        self,
        batch: VectorStoreFileBatch,
        batch_id: str,
        vector_store_id: str,
        docs: list[Document],
    ) -> None:
        """Raise on any indexing failure so the batch attempt is retried."""
        if batch.file_counts.failed > 0:
            try:
                failed_files = self.client.vector_stores.file_batches.list_files(
                    vector_store_id=vector_store_id,
                    batch_id=batch_id,
                    filter="failed",
                )
                doc_by_file_id = {_provider_file_id(d): d for d in docs}
                parts: list[str] = []
                for f in failed_files:
                    d = doc_by_file_id.get(f.id)
                    label = d.fname if d else f.id
                    msg = f.last_error.message if f.last_error else "no error detail"
                    parts.append(f"{label}: {msg}")
                logger.error(
                    f"[OpenAIVectorStoreCrud._raise_if_batch_incomplete] Files failed to index | "
                    f"{{'batch_id': '{batch_id}', 'failed_files': '{', '.join(parts)}'}}"
                )
                raise RuntimeError("; ".join(parts))
            except OpenAIError as err:
                logger.warning(
                    f"[OpenAIVectorStoreCrud._raise_if_batch_incomplete] Could not fetch per-file errors | "
                    f"{{'batch_id': '{batch_id}', 'error': '{str(err)}'}}"
                )
                raise

        # Only 'completed' is success; a 'cancelled'/'failed' batch with no per-file
        # failures slips past the failed-count check above.
        if batch.status != "completed":
            error_message = (
                f"[OPENAI] Vector store indexing did not complete "
                f"(status: {batch.status}). Retry the collection."
            )
            logger.error(
                f"[OpenAIVectorStoreCrud._raise_if_batch_incomplete] {error_message} | "
                f"vector_store_id={vector_store_id}, batch_id={batch_id}, "
                f"status={batch.status}"
            )
            raise RuntimeError(error_message)

    @retry(
        reraise=True,
        stop=stop_after_attempt(BATCH_INDEX_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=BATCH_RETRY_BACKOFF_BASE_SECONDS),
        retry=retry_if_exception_type((OpenAIError, RuntimeError)),
        before_sleep=_log_batch_retry,
    )
    def _create_and_index_batch(
        self, vector_store_id: str, docs: list[Document]
    ) -> tuple[VectorStoreFileBatch, str]:
        """Create the file batch, wait for indexing, verify it completed. Retried as
        a unit on any OpenAI/indexing failure; SoftTimeLimitExceeded is not retried
        (not an OpenAIError/RuntimeError) so it aborts the task inside the window."""
        batch_id = self._create_file_batch(
            vector_store_id, [_provider_file_id(doc) for doc in docs]
        )
        batch = self._poll_file_batch(batch_id, vector_store_id)
        self._raise_if_batch_incomplete(batch, batch_id, vector_store_id, docs)
        return batch, batch_id

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
            batch, batch_id = self._create_and_index_batch(vector_store_id, docs)
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

    def delete(self, vector_store_id: str, retries: int = 3) -> None:
        if retries < 1:
            logger.error(
                f"[OpenAIVectorStoreCrud.delete] Invalid retries value | {{'vector_store_id': '{vector_store_id}', 'retries': {retries}}}",
            )
            raise ValueError("Retries must be greater-than 1")

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
