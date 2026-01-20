import logging
import json
import ast
import re
from uuid import UUID
from typing import List

from fastapi import HTTPException
from sqlmodel import select
from openai import OpenAIError

from app.crud import DocumentCrud, CollectionCrud
from app.api.deps import SessionDep
from app.models import DocumentCollection, Collection


logger = logging.getLogger(__name__)


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


def batch_documents(
    document_crud: DocumentCrud, documents: List[UUID], batch_size: int
):
    """Batch document IDs into chunks of size `batch_size`, load each via `DocumentCrud.read_each`,
    and return a list of document batches."""

    logger.info(
        f"[batch_documents] Starting batch iteration for documents | {{'batch_size': {batch_size}, 'total_documents': {len(documents)}}}"
    )
    docs_batches = []
    start, stop = 0, batch_size
    while True:
        view = documents[start:stop]
        if not view:
            break
        batch_docs = document_crud.read_each(view)
        docs_batches.append(batch_docs)
        start = stop
        stop += batch_size
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
