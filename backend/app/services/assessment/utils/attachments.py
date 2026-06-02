"""Attachment resolution utilities for assessment batch builds.

Handles MIME type detection, base64 decoding, Google Drive URL normalization,
data-URL parsing, and conversion of dataset cell values into provider input objects.
"""

import base64
import binascii
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from app.models.assessment import AssessmentAttachment
from app.utils import validate_callback_url

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


def _image_mime_from_magic(blob: bytes) -> str | None:
    """Detect image mime type from leading magic bytes."""
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


def _guess_image_mime_from_base64(payload: str) -> str | None:
    blob = _decode_base64_prefix(payload)
    if not blob:
        return None
    return _image_mime_from_magic(blob)


def _type_from_magic(blob: bytes) -> str | None:
    """Detect 'image' or 'pdf' from leading magic bytes; None if neither."""
    if blob.startswith(b"%PDF"):
        return "pdf"
    if _image_mime_from_magic(blob):
        return "image"
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


def _drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from common share URL shapes."""
    match = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match and ("drive.google.com" in url or "drive.usercontent.google.com" in url):
        return match.group(1)
    return None


def _type_from_url_extension(url: str) -> str | None:
    """Detect 'image' or 'pdf' from a URL path extension; None if unknown."""
    path = (urlparse(url).path or "").lower()
    if path.endswith(".pdf"):
        return "pdf"
    if _guess_image_mime_from_url(url):
        return "image"
    return None


def _type_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    content_type = content_type.split(";")[0].strip().lower()
    if content_type == "application/pdf":
        return "pdf"
    if content_type.startswith("image/"):
        return "image"
    return None


_PROBE_MAX_REDIRECTS = 3


def _probe_url_type(url: str, num_bytes: int = 16) -> str | None:
    """Probe a remote URL's type: ranged byte sniff first, Content-Type fallback.
    Handles Google Drive URLs with the same logic as to_direct_attachment_url, since"""
    file_id = _drive_file_id(url)
    current = (
        f"https://drive.google.com/uc?export=download&id={file_id}" if file_id else url
    )

    try:
        for _ in range(_PROBE_MAX_REDIRECTS + 1):
            validate_callback_url(current)
            with requests.get(
                current,
                headers={"Range": f"bytes=0-{num_bytes - 1}"},
                timeout=10,
                stream=True,
                allow_redirects=False,
            ) as resp:
                location = resp.headers.get("Location")
                if resp.is_redirect and location:
                    current = urljoin(current, location)
                    continue
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=num_bytes):
                    magic_type = _type_from_magic(chunk)
                    if magic_type:
                        return magic_type
                    break
                return _type_from_content_type(resp.headers.get("Content-Type"))
        logger.warning(f"[_probe_url_type] Too many redirects probing {url}")
        return None
    except ValueError as e:
        logger.warning(f"[_probe_url_type] Blocked unsafe probe URL {url}: {e}")
        return None
    except requests.RequestException as e:
        logger.warning(f"[_probe_url_type] Probe failed for {url}: {e}")
        return None


def detect_item_type(
    value: str,
    format_type: str,
    fallback: str,
    cache: dict[str, str] | None = None,
) -> str:
    """Resolve a single attachment item as 'image' or 'pdf'.

    Order: data-URL/base64 magic (no network) -> URL extension -> remote probe
    (ranged byte sniff, then Content-Type) -> declared ``fallback`` type.
    ``fallback`` may be 'mixed'; when detection is inconclusive it resolves to
    'image'. Remote probe results are memoized in ``cache`` keyed by item value.
    """
    # 'mixed' is not a concrete output type; terminal default is image.
    safe_fallback = fallback if fallback in ("image", "pdf") else "image"

    if format_type != "url":
        data_url_mime, payload = split_data_url(value)
        if data_url_mime == "application/pdf":
            return "pdf"
        if data_url_mime and data_url_mime.startswith("image/"):
            return "image"
        blob = _decode_base64_prefix(payload)
        return (_type_from_magic(blob) if blob else None) or safe_fallback

    if cache is not None and value in cache:
        return cache[value]

    item_type = (
        _type_from_url_extension(value) or _probe_url_type(value) or safe_fallback
    )
    if cache is not None:
        cache[value] = item_type
    return item_type


def resolve_attachment_values(
    value: str,
    att: AssessmentAttachment,
    type_cache: dict[str, str] | None = None,
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
        item_type = detect_item_type(item_value, att.format, att.type, type_cache)
        normalized_value = (
            to_direct_attachment_url(item_value, item_type)
            if att.format == "url"
            else item_value
        )

        if item_type == "image":
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
        elif item_type == "pdf":
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


def build_gemini_attachment_parts(
    value: str,
    att: AssessmentAttachment,
    type_cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert one dataset cell into one or more Gemini content parts.

    Mirrors the per-item type detection used for the L2 batch so the same
    image/pdf routing applies to prefilter (topic relevance) calls.
    """
    value = value.strip()
    if not value:
        return []

    values = split_attachment_urls(value) if att.format == "url" else [value]

    parts: list[dict[str, Any]] = []
    for item_value in values:
        item_type = detect_item_type(item_value, att.format, att.type, type_cache)
        normalized_value = (
            to_direct_attachment_url(item_value, item_type)
            if att.format == "url"
            else item_value
        )

        if item_type == "image":
            mime_type, payload = resolve_image_mime_and_payload(
                normalized_value, att.format
            )
            if att.format == "url":
                parts.append(
                    {"fileData": {"mimeType": mime_type, "fileUri": normalized_value}}
                )
            else:
                parts.append({"inlineData": {"mimeType": mime_type, "data": payload}})
        elif item_type == "pdf":
            if att.format == "url":
                parts.append(
                    {
                        "fileData": {
                            "mimeType": "application/pdf",
                            "fileUri": normalized_value,
                        }
                    }
                )
            else:
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "application/pdf",
                            "data": split_data_url(normalized_value)[1],
                        }
                    }
                )

    return parts
