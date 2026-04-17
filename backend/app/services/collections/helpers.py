import logging
import json
import ast
import re
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import select

from app.crud import CollectionCrud
from app.api.deps import SessionDep
from app.models import DocumentCollection, Collection, CollectionPublic, Document

if TYPE_CHECKING:
    from app.core.cloud.storage import CloudStorage


logger = logging.getLogger(__name__)

# Necessary Constants -
# Maximum individual document size (must be less than batch size)
MAX_DOC_SIZE_MB = 25  # 25 MB maximum per document

# Maximum batch size for uploading documents to vector store
# Derived from MAX_DOC_SIZE + buffer to ensure single docs always fit
MAX_BATCH_SIZE_KB = (MAX_DOC_SIZE_MB + 5) * 1024  # 30 MB in KB (25 + 5 MB buffer)
MAX_BATCH_COUNT = 200  # Maximum documents per batch


def get_service_name(provider: str) -> str:
    """Get the collection service name for a provider."""
    names = {
        "openai": "openai vector store",
        #   "bedrock": "bedrock knowledge base",
        #  "gemini": "gemini file search store",
    }
    return names.get(provider.lower(), "")


def extract_error_message(err: Exception) -> str:
    """Extract a concise, user-facing message from an exception, preferring `error.message`
    in JSON/dict bodies after stripping prefixes.Falls back to cleaned text and truncates to
    1000 characters."""
    err_str = str(err).strip()

    body = re.sub(r"^Error code:\s*\d+\s*-\s*", "", err_str)
    message = None
    try:
        payload = json.loads(body)
        if isinstance(payload, dict):
            message = payload.get("error", {}).get("message")
    except Exception:
        pass

    if message is None:
        try:
            payload = ast.literal_eval(body)
            if isinstance(payload, dict):
                message = payload.get("error", {}).get("message")
        except Exception:
            pass

    if not message:
        message = body

    return message.strip()[:1000]


def calculate_total_size_kb(documents: list[Document], storage: CloudStorage) -> float:
    """
    Sum document sizes in KB. Uses the stored file_size_kb if available.
    """
    total: float = 0
    for doc in documents:
        if doc.file_size_kb is not None:
            total += doc.file_size_kb
        else:
            logger.info(
                f"[calculate_total_size_kb] file_size_kb missing, fetching from storage | {{'doc_id': '{doc.id}', 'fname': '{doc.fname}'}}"
            )
            total += storage.get_file_size_kb(doc.object_store_url)
    return total


def batch_documents(documents: list[Document]) -> list[list[Document]]:
    """
    Batch documents dynamically based on size and count limits.

    Creates a new batch when either:
    - Total size reaches 30 MB (30,720 KB)
    - Document count reaches 200

    Args:
        documents: List of Document objects to batch

    Returns:
        List of document batches
    """

    docs_batches = []
    current_batch = []
    current_batch_size_kb = 0

    for doc in documents:
        doc_size_kb = doc.file_size_kb or 15 * 1024

        would_exceed_size = (current_batch_size_kb + doc_size_kb) > MAX_BATCH_SIZE_KB
        would_exceed_count = len(current_batch) >= MAX_BATCH_COUNT

        if current_batch and (would_exceed_size or would_exceed_count):
            docs_batches.append(current_batch)
            logger.info(
                f"[batch_documents] Batch completed | {{'batch_num': {len(docs_batches)}, 'doc_count': {len(current_batch)}, 'batch_size_mb': {round(current_batch_size_kb / 1024)}}}"
            )
            current_batch = []
            current_batch_size_kb = 0

        current_batch.append(doc)
        current_batch_size_kb += doc_size_kb

    if current_batch:
        docs_batches.append(current_batch)
        logger.info(
            f"[batch_documents] Final Batch completed | {{'batch_num': {len(docs_batches)}, 'doc_count': {len(current_batch)}, 'batch_size_mb': {round(current_batch_size_kb / 1024)}}}"
        )

    logger.info(
        f"[batch_documents] Batching complete | {{'total_batches': {len(docs_batches)}, 'total_documents': {len(documents)}}}"
    )

    return docs_batches


# Even though this function is used in the documents router, it's kept here for now since the assistant creation logic will
# eventually be removed from Kaapi. Once that happens, this function can be safely deleted -
def pick_service_for_documennt(session, doc_id: UUID, a_crud, v_crud):
    """
    Return the correct remote (v_crud or a_crud) for this document
    by inspecting an active linked Collection's llm_service_name.
    Defaults to a_crud if not vector store.
    """
    coll = session.exec(
        select(Collection)
        .join(DocumentCollection, DocumentCollection.collection_id == Collection.id)
        .where(
            DocumentCollection.document_id == doc_id,
            Collection.deleted_at.is_(None),
        )
        .limit(1)
    ).first()

    service = (
        (getattr(coll, "llm_service_name", "") or "").strip().lower() if coll else ""
    )
    return v_crud if service == get_service_name("openai") else a_crud


def ensure_unique_name(
    session: SessionDep,
    project_id: int,
    requested_name: str,
) -> str:
    """
    Ensure collection name is unique based on strategy.

    """
    existing = CollectionCrud(session, project_id).exists_by_name(requested_name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{requested_name}' already exists. Choose a different name.",
        )

    return requested_name


def to_collection_public(collection: Collection) -> CollectionPublic:
    """
    Convert a Collection DB model to CollectionPublic response model.

    Maps fields based on service type:
    - If llm_service_name is a vector store (matches get_service_name pattern),
      use knowledge_base_id/knowledge_base_provider
    - Otherwise (assistant), use llm_service_id/llm_service_name
    """
    is_vector_store = collection.llm_service_name == get_service_name(
        collection.provider
    )

    if is_vector_store:
        return CollectionPublic(
            id=collection.id,
            knowledge_base_id=collection.llm_service_id,
            knowledge_base_provider=collection.llm_service_name,
            project_id=collection.project_id,
            inserted_at=collection.inserted_at,
            updated_at=collection.updated_at,
            deleted_at=collection.deleted_at,
        )
    else:
        return CollectionPublic(
            id=collection.id,
            llm_service_id=collection.llm_service_id,
            llm_service_name=collection.llm_service_name,
            project_id=collection.project_id,
            inserted_at=collection.inserted_at,
            updated_at=collection.updated_at,
            deleted_at=collection.deleted_at,
        )
