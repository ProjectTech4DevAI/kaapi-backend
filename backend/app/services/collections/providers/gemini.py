import logging
import mimetypes
import tempfile
import time
from typing import List

from google import genai
from google.genai.types import File, FileState, UploadFileConfig
from sqlmodel import Session

from app.core.cloud.storage import CloudStorage
from app.core.db import engine
from app.crud import DocumentCrud
from app.crud.rag import GeminiFileSearchStoreCrud
from app.models import Collection, Document, ProviderType
from app.services.collections.helpers import get_service_name
from app.services.collections.providers import BaseProvider

logger = logging.getLogger(__name__)

GOOGLE_PROVIDER = ProviderType.google.value

STREAM_CHUNK_BYTES = 1024 * 1024

# Freshly uploaded Files API objects report PROCESSING until Gemini finishes
# ingesting them; only ACTIVE files can be imported into a File Search store.
FILE_ACTIVE_POLL_INTERVAL_SECONDS = 2
FILE_ACTIVE_POLL_TIMEOUT_SECONDS = 120


class GeminiAIStudioProvider(BaseProvider):
    """Google AI Studio collection provider backed by Gemini File Search stores."""

    def __init__(self, client: genai.Client) -> None:
        super().__init__(client)
        self.client = client

    def get_existing_file_id(self, doc: Document) -> str | None:
        """Return the cached Files API name only if it is still ACTIVE upstream.

        Files API objects expire after 48h, so the cache is validated against
        the API rather than trusted; any failure falls back to a re-upload.
        """
        cached = (doc.file_id or {}).get(GOOGLE_PROVIDER)
        if not cached:
            return None
        try:
            remote = self.client.files.get(name=cached)
            if remote.state == FileState.ACTIVE:
                logger.info(
                    "[GeminiAIStudioProvider.get_existing_file_id] Reusing cached file | "
                    "doc_id=%s, file_name=%s",
                    doc.id,
                    cached,
                )
                return cached
            logger.info(
                "[GeminiAIStudioProvider.get_existing_file_id] Cached file not ACTIVE (state=%s), re-uploading | "
                "doc_id=%s, file_name=%s",
                remote.state,
                doc.id,
                cached,
            )
        except Exception as err:
            # Expired (48h) or deleted upstream — expected cache miss, not an error.
            logger.info(
                "[GeminiAIStudioProvider.get_existing_file_id] Cached file invalid, re-uploading | "
                "doc_id=%s, file_name=%s, reason=%s",
                doc.id,
                cached,
                str(err),
            )
        return None

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
                # The temp file path has no extension, so the SDK cannot guess
                # the mimetype; derive it from the original filename instead.
                mime_type, _ = mimetypes.guess_type(doc.fname)
                if mime_type is None:
                    raise ValueError(
                        f"[KAAPI] Could not determine MIME type for '{doc.fname}'. "
                        f"Rename the document with a standard extension "
                        f"(e.g. .pdf, .txt, .json) and re-upload."
                    )
                with tempfile.NamedTemporaryFile() as tmp:
                    body = storage.stream(doc.object_store_url)
                    while chunk := body.read(STREAM_CHUNK_BYTES):
                        tmp.write(chunk)
                    if doc.file_size_kb is None:
                        doc.file_size_kb = round(tmp.tell() / 1024, 2)
                    tmp.seek(0)
                    uploaded = self.client.files.upload(
                        file=tmp.name,
                        config=UploadFileConfig(
                            display_name=doc.fname, mime_type=mime_type
                        ),
                    )
            except Exception as err:
                logger.error(
                    "[GeminiAIStudioProvider.upload_files] Failed to upload file | doc_id=%s, error=%s",
                    doc.id,
                    str(err),
                    exc_info=True,
                )
                raise

            self._wait_until_active(uploaded, doc)

            doc.file_id = {
                **(doc.file_id or {}),
                GOOGLE_PROVIDER: uploaded.name,
            }
            try:
                with Session(engine) as session:
                    document_crud = DocumentCrud(session, project_id)
                    db_doc = document_crud.read_one(doc.id)
                    db_doc.file_id = doc.file_id
                    db_doc.file_size_kb = doc.file_size_kb
                    document_crud.update(db_doc)
            except Exception as err:
                logger.error(
                    "[GeminiAIStudioProvider.upload_files] DB persistence failed, rolling back Gemini file | "
                    "doc_id=%s, file_name=%s, error=%s",
                    doc.id,
                    uploaded.name,
                    str(err),
                    exc_info=True,
                )
                try:
                    self.client.files.delete(name=uploaded.name)
                    logger.info(
                        "[GeminiAIStudioProvider.upload_files] Rolled back Gemini file | "
                        "doc_id=%s, file_name=%s",
                        doc.id,
                        uploaded.name,
                    )
                except Exception as delete_err:
                    logger.error(
                        "[GeminiAIStudioProvider.upload_files] Rollback failed, file is orphaned | "
                        "doc_id=%s, file_name=%s, error=%s",
                        doc.id,
                        uploaded.name,
                        str(delete_err),
                    )
                doc.file_id.pop(GOOGLE_PROVIDER, None)
                raise

    def _wait_until_active(self, uploaded: File, doc: Document) -> None:
        """Poll an uploaded Files API object until it becomes ACTIVE.

        Raises InterruptedError on FAILED state or timeout so the caller rolls
        back the upload rather than importing an unusable file.
        """
        deadline = time.monotonic() + FILE_ACTIVE_POLL_TIMEOUT_SECONDS
        while uploaded.state == FileState.PROCESSING:
            if time.monotonic() > deadline:
                error_message = (
                    f"[KAAPI] Timed out (code: file-active-timeout) after "
                    f"{FILE_ACTIVE_POLL_TIMEOUT_SECONDS}s waiting for Gemini to "
                    f"process uploaded file for doc {doc.id}. Retry the upload "
                    f"with a smaller file."
                )
                logger.error(
                    f"[GeminiAIStudioProvider._wait_until_active] {error_message} | "
                    f"doc_id={doc.id}, file_name={uploaded.name}"
                )
                raise InterruptedError(error_message)
            time.sleep(FILE_ACTIVE_POLL_INTERVAL_SECONDS)
            uploaded = self.client.files.get(name=uploaded.name)

        if uploaded.state == FileState.FAILED:
            error_message = (
                f"[GEMINI] File processing failed for doc {doc.id}: "
                f"{uploaded.error}. Verify the file is a supported, non-corrupt "
                f"format, then retry."
            )
            logger.error(
                f"[GeminiAIStudioProvider._wait_until_active] {error_message} | "
                f"doc_id={doc.id}, file_name={uploaded.name}"
            )
            raise InterruptedError(error_message)

    def create_vector_store(self) -> str:
        try:
            store_name = GeminiFileSearchStoreCrud(self.client).create()
            logger.info(
                "[GeminiAIStudioProvider.create_vector_store] File search store created | store_name=%s",
                store_name,
            )
            return store_name
        except Exception as e:
            logger.error(
                f"[GeminiAIStudioProvider.create_vector_store] Failed to create file search store: {str(e)}",
                exc_info=True,
            )
            raise

    def create(
        self,
        docs: List[Document],
        vector_store_id: str,
    ) -> Collection:
        try:
            store_crud = GeminiFileSearchStoreCrud(self.client)
            for doc in docs:
                store_crud.import_document(
                    vector_store_id, doc.file_id[GOOGLE_PROVIDER]
                )
            if docs:
                logger.info(
                    "[GeminiAIStudioProvider.create] Imported documents | store_name=%s, doc_count=%d",
                    vector_store_id,
                    len(docs),
                )

            return Collection(
                knowledge_base_id=vector_store_id,
                knowledge_base_provider=get_service_name(GOOGLE_PROVIDER),
            )

        except Exception as e:
            logger.error(
                f"[GeminiAIStudioProvider.create] Failed to create collection: {str(e)}",
                exc_info=True,
            )
            raise

    def delete(self, collection: Collection) -> None:
        try:
            GeminiFileSearchStoreCrud(self.client).delete(collection.knowledge_base_id)
            logger.info(
                f"[GeminiAIStudioProvider.delete] Deleted file search store | "
                f"store_name={collection.knowledge_base_id}"
            )
        except Exception as e:
            logger.error(
                f"[GeminiAIStudioProvider.delete] Failed to delete file search store | "
                f"knowledge_base_id={collection.knowledge_base_id}, error={str(e)}",
                exc_info=True,
            )
            raise
