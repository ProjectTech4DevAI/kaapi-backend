"""Attachment resolution utilities for assessment batch builds.

URL-only: dataset cells hold attachment URLs. Handles Google Drive URL
normalization and conversion of cell values into provider input objects.
Attachments are passed to providers by reference (URL), never inlined as base64,
to keep the batch build memory-light.
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse

from app.models.assessment import AssessmentAttachment

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


def resolve_item_type(declared: str, type_override: str | None = None) -> str:
    """Resolve an attachment item as 'image' or 'pdf' from the user-declared type.

    Trusts the user: a per-row ``type_override`` (for 'mixed' columns) wins, else the
    column's declared ``type``. Anything non-concrete falls back to 'image'.
    """
    item_type = type_override or declared
    return item_type if item_type in ("image", "pdf") else "image"


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
    resolved: list[dict[str, Any]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == "image":
            resolved.append({"type": "input_image", "image_url": url})
        else:
            resolved.append({"type": "input_file", "file_url": url})
    return resolved


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
    parts: list[dict[str, Any]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == "image":
            mime_type = _guess_image_mime_from_url(url) or "image/png"
            parts.append({"fileData": {"mimeType": mime_type, "fileUri": url}})
        else:
            parts.append({"fileData": {"mimeType": "application/pdf", "fileUri": url}})
    return parts
