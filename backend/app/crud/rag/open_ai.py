import json
import logging
import functools as ft
from io import BytesIO
from typing import Iterable

import openai
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
                logger.warning(
                    f"[ResourceCleaner.call] OpenAI error during cleanup | {{'cleaner_type': '{self}', 'resource': '{resource}', 'error': '{str(err)}'}}",
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
                f"[OpenAIVectorStoreCrud.update] Uploading files to vector store | "
                f"{{'vector_store_id': '{vector_store_id}', 'file_count': {len(files)}}}"
            )

            try:
                req = self.client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=files,
                )
            except openai.RateLimitError as e:
                raise InterruptedError(
                    f"OpenAI rate limit exceeded (code: {e.status_code}): "
                    f"{e.message}. Try again in 1 minute. If issue persists, "
                    f"contact Kaapi."
                )
            except openai.AuthenticationError as e:
                raise InterruptedError(
                    f"OpenAI authentication failed (code: {e.status_code}): "
                    f"{e.message}. Check your OpenAI API key is valid and "
                    f"has not expired."
                )
            except openai.NotFoundError as e:
                raise InterruptedError(
                    f"OpenAI resource not found (code: {e.status_code}): "
                    f"{e.message}. Verify the vector store ID exists and "
                    f"hasn't been deleted."
                )
            except openai.BadRequestError as e:
                raise InterruptedError(
                    f"OpenAI bad request (code: {e.status_code}): {e.message}. "
                    f"Review the file payload and metadata; the request may "
                    f"be malformed."
                )
            except openai.UnprocessableEntityError as e:
                raise InterruptedError(
                    f"OpenAI unprocessable entity (code: {e.status_code}): "
                    f"{e.message}. The uploaded files may be in an "
                    f"unsupported format or exceed size limits."
                )
            except openai.InternalServerError as e:
                raise InterruptedError(
                    f"OpenAI server error (code: {e.status_code}): {e.message}. "
                    f"This is usually transient — retry in a few seconds. If "
                    f"issue persists, contact Kaapi."
                )
            except openai.APITimeoutError as e:
                raise InterruptedError(
                    f"OpenAI request timed out: {e}. Retry the upload, or "
                    f"split the batch into smaller chunks."
                )
            except openai.OpenAIError as e:
                raise InterruptedError(
                    f"OpenAI error: {e}. If this persists, contact Kaapi."
                )

            logger.info(
                f"[OpenAIVectorStoreCrud.update] File upload completed | "
                f"{{'vector_store_id': '{vector_store_id}', "
                f"'completed_files': {req.file_counts.completed}, "
                f"'total_files': {req.file_counts.total}}}"
            )

            if req.file_counts.completed != req.file_counts.total:
                # Enrich the error string by listing each failed file's
                # `last_error.message` from OpenAI. Fall back to the
                # count-only message if the follow-up list_files call
                # itself fails.
                failed_summary = ""
                try:
                    page = self.client.vector_stores.file_batches.list_files(
                        batch_id=req.id,
                        vector_store_id=vector_store_id,
                        filter="failed",
                        limit=10,
                    )
                    parts = []
                    for f in page:
                        f_err = getattr(f, "last_error", None)
                        f_msg = (
                            getattr(f_err, "message", None) if f_err else None
                        ) or "Unknown error"
                        parts.append(f"{f.id} ({f_msg})")
                    failed_summary = ", ".join(parts)
                    if getattr(page, "has_more", False):
                        failed_summary = f"{failed_summary}, ..."
                    if len(failed_summary) > 600:
                        failed_summary = failed_summary[:597] + "..."
                except OpenAIError as list_err:
                    logger.warning(
                        f"[OpenAIVectorStoreCrud.update] Could not list failed "
                        f"files | {{'vector_store_id': '{vector_store_id}', "
                        f"'batch_id': '{req.id}', 'error': '{list_err}'}}"
                    )

                error_msg = (
                    f"OpenAI document processing error: "
                    f"{req.file_counts.completed}/{req.file_counts.total} "
                    f"files completed"
                )
                if failed_summary:
                    error_msg = f"{error_msg}. Failed files: {failed_summary}"
                logger.error(
                    f"[OpenAIVectorStoreCrud.update] Document processing error | "
                    f"{{'vector_store_id': '{vector_store_id}', "
                    f"'completed_files': {req.file_counts.completed}, "
                    f"'total_files': {req.file_counts.total}}}"
                )
                raise InterruptedError(error_msg)

            yield from docs

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
