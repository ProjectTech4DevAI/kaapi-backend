"""Attachment utilities for assessment batch builds: URL normalization, gs:// resolution, provider parts."""

import logging
import re
from typing import Any, cast
from urllib.parse import urlparse

from sqlmodel import Session

from app.core.config import settings
from app.models.assessment import AssessmentAttachment
from app.models.llm.constants import KaapiProvider
from app.services.buckets.attachments import is_gcs_uri, resolve_attachments

logger = logging.getLogger(__name__)

_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def split_attachment_urls(value: str) -> list[str]:
    """Split comma/newline separated attachment URLs from a single dataset cell."""
    return [part.strip() for part in re.split(r"[\n,]+", value) if part.strip()]


def rewrite_gcs_attachment_urls(
    *,
    session: Session,
    rows: list[dict[str, str]],
    attachments: list[AssessmentAttachment],
    llm_provider: str,
    project_id: int,
    organization_id: int,
) -> list[dict[str, str]]:
    """Replace gs:// tokens in attachment cells with LLM-reachable URLs.

    Bulk-resolves every gs:// URI across the batch once, then rewrites each cell;
    non-gs values are left as-is.
    """
    gcs_uris = {
        attachment_url
        for row in rows
        for att in attachments
        for attachment_url in split_attachment_urls(row.get(att.column, ""))
        if is_gcs_uri(attachment_url)
    }
    if not gcs_uris:
        return rows

    resolved = resolve_attachments(
        session=session,
        source=list(gcs_uris),
        llm_provider=cast(KaapiProvider, llm_provider),
        project_id=project_id,
        organization_id=organization_id,
        expires_in=settings.MAX_SIGNED_URL_EXPIRY_SECONDS,
    )
    assert isinstance(resolved, dict)  # list input always yields a dict

    rewritten: list[dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        for att in attachments:
            value = row.get(att.column)
            if not value:
                continue
            new_row[att.column] = ", ".join(
                resolved.get(attachment_url, attachment_url)
                for attachment_url in split_attachment_urls(value)
            )
        rewritten.append(new_row)
    return rewritten


def to_direct_attachment_url(url: str, attachment_type: str) -> str:
    """Normalize share-page attachment URLs into provider-fetchable direct URLs.

    This currently handles common Google Drive share URL shapes. The file must
    still be publicly accessible to the model provider.
    """
    url = url.strip()
    file_id = None

    match = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url)
    if match:
        file_id = match.group(1)

    if not file_id:
        match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if match and (
            "drive.google.com" in url or "drive.usercontent.google.com" in url
        ):
            file_id = match.group(1)

    if not file_id:
        return url

    if attachment_type == "image":
        return f"https://lh3.googleusercontent.com/d/{file_id}"

    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _guess_image_mime_from_url(url: str) -> str | None:
    path = urlparse(url).path or ""
    for ext, mime in _IMAGE_MIME_BY_EXT.items():
        if path.lower().endswith(ext):
            return mime
    return None


def resolve_item_type(declared: str, type_override: str | None = None) -> str | None:
    """Resolve an attachment item as 'image' or 'pdf' from the user-declared type.

    A per-row ``type_override`` (for 'mixed' columns) wins, else the column's declared
    ``type``. Returns None when the type stays unresolved (e.g. a 'mixed' row whose
    value didn't map to a concrete type) so callers can skip rather than guess.
    """
    item_type = type_override or declared
    return item_type if item_type in ("image", "pdf") else None


def _normalize_type_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _split_type_values(value: str) -> list[str]:
    return [
        normalized
        for part in re.split(r"[\n,]+", value)
        if (normalized := _normalize_type_value(part))
    ]


def attachment_type_for_row(
    att: AssessmentAttachment, row: dict[str, str]
) -> str | None:
    """For a 'mixed' column, resolve this row's type from type_column + type_value_map.

    Returns 'image'/'pdf', or None to let normal detection (extension/declared) decide.
    """
    type_column = getattr(att, "type_column", None)
    type_value_map = getattr(att, "type_value_map", None)
    if att.type != "mixed" or not type_column or not type_value_map:
        return None

    normalized_map: dict[str, str] = {}
    for raw_values, mapped_type in type_value_map.items():
        if mapped_type not in ("image", "pdf"):
            continue
        for value in _split_type_values(raw_values):
            normalized_map[value] = mapped_type

    row_values = _split_type_values(row.get(type_column) or "")
    if not row_values:
        return None

    mapped_values = {
        normalized_map[value] for value in row_values if value in normalized_map
    }
    return mapped_values.pop() if len(mapped_values) == 1 else None


def resolve_attachment_values(
    value: str,
    att: AssessmentAttachment,
    type_override: str | None = None,
) -> list[dict[str, Any]]:
    """Convert one dataset cell into one or more OpenAI-style input objects (by URL)."""
    value = value.strip()
    if not value:
        return []

    item_type = resolve_item_type(att.type, type_override)
    if item_type is None:
        logger.warning(
            "[resolve_attachment_values] Unresolved type for column=%s — skipping",
            att.column,
        )
        return []
    resolved: list[dict[str, Any]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == "image":
            resolved.append({"type": "input_image", "image_url": url})
        else:
            resolved.append({"type": "input_file", "file_url": url})
    return resolved


def build_anthropic_attachment_parts(
    value: str,
    att: AssessmentAttachment,
    type_override: str | None = None,
) -> list[dict[str, Any]]:
    """Convert one dataset cell into one or more Anthropic content blocks (by URL)."""
    value = value.strip()
    if not value:
        return []

    item_type = resolve_item_type(att.type, type_override)
    if item_type is None:
        logger.warning(
            "[build_anthropic_attachment_parts] Unresolved type for column=%s — skipping",
            att.column,
        )
        return []
    blocks: list[dict[str, Any]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == "image":
            blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            blocks.append({"type": "document", "source": {"type": "url", "url": url}})
    return blocks


def build_gemini_attachment_parts(
    value: str,
    att: AssessmentAttachment,
    type_override: str | None = None,
) -> list[dict[str, Any]]:
    """Convert one dataset cell into one or more Gemini content parts (by URL).

    Mirrors the per-item type routing used for the L2 batch so the same
    image/pdf handling applies to prefilter (topic relevance) calls.
    """
    value = value.strip()
    if not value:
        return []

    item_type = resolve_item_type(att.type, type_override)
    if item_type is None:
        logger.warning(
            "[build_gemini_attachment_parts] Unresolved type for column=%s — skipping",
            att.column,
        )
        return []
    parts: list[dict[str, Any]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == "image":
            mime_type = _guess_image_mime_from_url(url) or "image/png"
            parts.append({"fileData": {"mimeType": mime_type, "fileUri": url}})
        else:
            parts.append({"fileData": {"mimeType": "application/pdf", "fileUri": url}})
    return parts
