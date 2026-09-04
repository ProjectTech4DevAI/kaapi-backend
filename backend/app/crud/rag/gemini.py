import logging
import time
from typing import NoReturn

from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

# Import into a File Search store is a long-running operation. Poll until it
# reports done, bounded so a stuck operation surfaces as an error instead of
# blocking the collection task indefinitely.
IMPORT_POLL_INTERVAL_SECONDS = 5
IMPORT_POLL_TIMEOUT_SECONDS = 300


def _raise_genai_error(
    err: errors.APIError,
    *,
    log_prefix: str,
    resource: str,
) -> NoReturn:
    """Translate a google-genai SDK error into a user-facing InterruptedError.

    Mirrors the source-tagged, fault-based pattern in
    OpenAIVectorStoreCrud.update and GoogleAIProvider.execute: 4xx is the
    caller's fault (warning), 5xx is provider-side (error).
    """
    if isinstance(err, errors.ClientError):
        code = err.code
        status = err.status or ""
        msg = err.message or str(err)
        if code == 429:
            error_message = (
                f"[GEMINI] Rate limit / quota exceeded (code: 429 {status}): "
                f"{msg}. Wait at least 1 minute and retry; if the issue "
                f"persists, request a quota increase from Google or contact Kaapi."
            )
        elif code == 403:
            error_message = (
                f"[GEMINI] Authentication / permission denied (code: 403 "
                f"{status}): {msg}. Verify the Gemini API key is valid, not "
                f"expired, and has access to the File Search API for this project."
            )
        elif code == 404:
            error_message = (
                f"[GEMINI] Resource not found (code: 404 {status}): {msg}. "
                f"Verify the File Search store and file names exist and have "
                f"not expired."
            )
        elif code == 400:
            error_message = (
                f"[GEMINI] Bad request (code: 400 {status}): {msg}. Review the "
                f"File Search store configuration and file references — the "
                f"request shape may be invalid."
            )
        else:
            error_message = (
                f"[GEMINI] Client error (code: {code} {status}): {msg}. Review "
                f"the request configuration; if the issue persists, contact Kaapi."
            )
        logger.warning(
            f"[{log_prefix}] {error_message} | resource={resource}",
            exc_info=True,
        )
    else:
        status = getattr(err, "status", "") or ""
        code = getattr(err, "code", "unknown")
        msg = getattr(err, "message", None) or str(err)
        error_message = (
            f"[GEMINI] Server error (code: {code} {status}): {msg}. This is "
            f"typically transient (Gemini overloaded or internal error) — retry "
            f"in a few seconds. If the issue persists, contact Kaapi."
        )
        logger.error(
            f"[{log_prefix}] {error_message} | resource={resource}",
            exc_info=True,
        )
    raise InterruptedError(error_message)


class GeminiCrud:
    def __init__(self, client: genai.Client) -> None:
        if client is None:  # pyright: ignore[reportUnnecessaryComparison]
            logger.error("[GeminiCrud] Gemini client is not configured")
            raise ValueError("Gemini client is not configured")

        self.client = client


class GeminiFileSearchStoreCrud(GeminiCrud):
    """CRUD for Google AI Studio (Gemini) File Search stores."""

    def create(self) -> str:
        logger.info("[GeminiFileSearchStoreCrud.create] Creating file search store")
        try:
            store = self.client.file_search_stores.create(
                config=types.CreateFileSearchStoreConfig()
            )
        except (errors.ClientError, errors.ServerError) as err:
            _raise_genai_error(
                err,
                log_prefix="GeminiFileSearchStoreCrud.create",
                resource="<new store>",
            )

        if store.name is None:
            raise RuntimeError(
                "[GEMINI] File search store created without a name; cannot use as knowledge_base_id"
            )
        logger.info(
            f"[GeminiFileSearchStoreCrud.create] File search store created | "
            f"store_name={store.name}"
        )
        return store.name

    def import_document(self, store_name: str, file_name: str) -> None:
        logger.info(
            f"[GeminiFileSearchStoreCrud.import_document] Importing file into store | "
            f"store_name={store_name}, file_name={file_name}"
        )
        try:
            operation = self.client.file_search_stores.import_file(
                file_search_store_name=store_name,
                file_name=file_name,
            )
        except (errors.ClientError, errors.ServerError) as err:
            _raise_genai_error(
                err,
                log_prefix="GeminiFileSearchStoreCrud.import_document",
                resource=store_name,
            )

        deadline = time.monotonic() + IMPORT_POLL_TIMEOUT_SECONDS
        while not operation.done:
            if time.monotonic() > deadline:
                error_message = (
                    f"[KAAPI] Timed out (code: import-timeout) after "
                    f"{IMPORT_POLL_TIMEOUT_SECONDS}s waiting for Gemini to import "
                    f"'{file_name}' into store '{store_name}'. Retry the import, "
                    f"or use fewer / smaller files per collection."
                )
                logger.error(
                    f"[GeminiFileSearchStoreCrud.import_document] {error_message}"
                )
                raise InterruptedError(error_message)

            time.sleep(IMPORT_POLL_INTERVAL_SECONDS)
            try:
                operation = self.client.operations.get(operation)
            except (errors.ClientError, errors.ServerError) as err:
                _raise_genai_error(
                    err,
                    log_prefix="GeminiFileSearchStoreCrud.import_document",
                    resource=store_name,
                )

        if operation.error:
            error_message = (
                f"[GEMINI] Import operation failed for '{file_name}' into store "
                f"'{store_name}': {operation.error}. Verify the file is a "
                f"supported format and has not expired, then retry."
            )
            logger.error(f"[GeminiFileSearchStoreCrud.import_document] {error_message}")
            raise InterruptedError(error_message)

        logger.info(
            f"[GeminiFileSearchStoreCrud.import_document] File imported | "
            f"store_name={store_name}, file_name={file_name}"
        )

    def delete(self, store_name: str) -> None:
        logger.info(
            f"[GeminiFileSearchStoreCrud.delete] Deleting file search store | "
            f"store_name={store_name}"
        )
        try:
            self.client.file_search_stores.delete(
                name=store_name,
                config=types.DeleteFileSearchStoreConfig(force=True),
            )
        except (errors.ClientError, errors.ServerError) as err:
            _raise_genai_error(
                err,
                log_prefix="GeminiFileSearchStoreCrud.delete",
                resource=store_name,
            )

        logger.info(
            f"[GeminiFileSearchStoreCrud.delete] File search store deleted | "
            f"store_name={store_name}"
        )
