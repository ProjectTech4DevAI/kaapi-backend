"""Attachment resolution utilities for assessment batch builds.

Handles MIME type detection, base64 decoding, Google Drive URL normalization,
data-URL parsing, and conversion of dataset cell values into provider input objects.
"""

import base64
import binascii
import re
from typing import Any
from urllib.parse import urlparse

from app.models.assessment import AssessmentAttachment

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


def split_data_url(value: str) -> tuple[str | None, str]:
    """Return (mime_type, base64_payload) for a data URL; otherwise (None, value)."""
    match = re.match(
        r"^data:([^;]+);base64,(.+)$",
        value.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, value.strip()
    return match.group(1).strip().lower(), match.group(2).strip()


def _guess_image_mime_from_url(url: str) -> str | None:
    path = urlparse(url).path or ""
    for ext, mime in _IMAGE_MIME_BY_EXT.items():
        if path.lower().endswith(ext):
            return mime
    return None


def _decode_base64_prefix(payload: str, max_chars: int = 256) -> bytes | None:
    compact = re.sub(r"\s+", "", payload)
    if not compact:
        return None
    sample = compact[:max_chars]
    padding = "=" * (-len(sample) % 4)
    try:
        return base64.b64decode(sample + padding, validate=False)
    except (binascii.Error, ValueError):
        return None


def _guess_image_mime_from_base64(payload: str) -> str | None:
    blob = _decode_base64_prefix(payload)
    if not blob:
        return None
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if blob.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if blob.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if blob.startswith(b"BM"):
        return "image/bmp"
    if len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    if blob.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return None


def resolve_image_mime_and_payload(
    value: str,
    format_type: str,
) -> tuple[str, str]:
    """Resolve image mime type and raw base64 payload (for base64 format)."""
    if format_type == "url":
        return _guess_image_mime_from_url(value) or "image/png", value

    data_url_mime, payload = split_data_url(value)
    if data_url_mime and data_url_mime.startswith("image/"):
        return data_url_mime, payload

    return _guess_image_mime_from_base64(payload) or "image/png", payload


def resolve_attachment_values(
    value: str,
    att: AssessmentAttachment,
) -> list[dict[str, Any]]:
    """Convert one dataset cell into one or more OpenAI-style input objects."""
    value = value.strip()
    if not value:
        return []

    if att.format == "url":
        values = split_attachment_urls(value)
    else:
        values = [value]

    resolved: list[dict[str, Any]] = []
    for item_value in values:
        normalized_value = (
            to_direct_attachment_url(item_value, att.type)
            if att.format == "url"
            else item_value
        )

        if att.type == "image":
            if att.format == "url":
                resolved.append({"type": "input_image", "image_url": normalized_value})
            else:
                mime_type, payload = resolve_image_mime_and_payload(
                    normalized_value,
                    "base64",
                )
                resolved.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{payload}",
                    }
                )
        elif att.type == "pdf":
            if att.format == "url":
                resolved.append(
                    {
                        "type": "input_file",
                        "file_url": normalized_value,
                    }
                )
            else:
                _, payload = split_data_url(normalized_value)
                resolved.append(
                    {
                        "type": "input_file",
                        "file_data": f"data:application/pdf;base64,{payload}",
                        "filename": "document.pdf",
                    }
                )

    return resolved
