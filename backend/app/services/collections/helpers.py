import logging
import json
import ast
import re

from fastapi import HTTPException

from app.crud import CollectionCrud
from app.api.deps import SessionDep
from app.models import Collection, CollectionPublic, Document, ProviderType


logger = logging.getLogger(__name__)

# Necessary Constants -
# Maximum individual document size (must be less than batch size)
MAX_DOC_SIZE_MB = 25  # 25 MB maximum per document

# Maximum batch size for uploading documents to vector store
MAX_BATCH_SIZE_KB = (MAX_DOC_SIZE_MB + 5) * 1024  # 30 MB in KB (25 + 5 MB buffer)
MAX_BATCH_COUNT = 200  # Maximum documents per batch


def get_service_name(provider: ProviderType) -> str:
    """Get the collection service name for a provider."""
    names: dict[str, str] = {
        ProviderType.openai.value: "openai vector store",
        ProviderType.google_aistudio.value: "gemini file search store",
        # ProviderType.bedrock.value: "bedrock knowledge base",
    }
    return names.get(provider.lower(), "")


def extract_error_message(err: Exception | str) -> str:
    """Extract a concise, user-facing message from an exception, preferring `error.message`
    in JSON/dict bodies after stripping prefixes.Falls back to cleaned text and truncates to
    1000 characters."""
    err_str = str(err).strip()
    body = re.sub(r"^Error code:\s*\d+\s*-\s*", "", err_str)

    message: str | None = None
    for parse in (json.loads, ast.literal_eval):
        try:
            payload = parse(body)
            candidate = payload["error"]["message"]
        except Exception:
            continue
        if isinstance(candidate, str):
            message = candidate
            break

    return (message or body).strip()[:1000]


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

    docs_batches: list[list[Document]] = []
    current_batch: list[Document] = []
    current_batch_size_kb = 0.0

    for doc in documents:
        doc_size_kb = doc.file_size_kb
        if doc_size_kb is None:
            raise TypeError(f"[batch_documents] Document {doc.id} missing file_size_kb")

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
    return CollectionPublic(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        knowledge_base_id=collection.knowledge_base_id,
        knowledge_base_provider=collection.knowledge_base_provider,
        project_id=collection.project_id,
        inserted_at=collection.inserted_at,
        updated_at=collection.updated_at,
        deleted_at=collection.deleted_at,
    )
