"""Attachment utilities for assessment batch builds: URL normalization, gs:// resolution, provider parts."""

import logging
import re
from typing import cast
from urllib.parse import urlparse

from pydantic import JsonValue
from sqlmodel import Session

from app.core.config import settings
from app.models.assessment import AssessmentAttachment
from app.models.config.assessment_blob import (
    AUDIO_COLUMN_TYPE,
    IMAGE_COLUMN_TYPE,
    VIDEO_COLUMN_TYPE,
)
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

_VIDEO_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".mov": "video/mov",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".avi": "video/avi",
    ".wmv": "video/wmv",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".3gp": "video/3gpp",
}

_AUDIO_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

_PDF_MIME = "application/pdf"
_DEFAULT_IMAGE_MIME = "image/png"
_DEFAULT_VIDEO_MIME = "video/mp4"
_DEFAULT_AUDIO_MIME = "audio/mp3"

YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com")


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
            rewritten_urls: list[str] = []
            for attachment_url in split_attachment_urls(value):
                target = resolved.get(attachment_url)
                if target is None and is_gcs_uri(attachment_url):
                    logger.warning(
                        "[rewrite_gcs_attachment_urls] Unresolved gs:// attachment | column=%s",
                        att.column,
                    )
                rewritten_urls.append(target or attachment_url)
            new_row[att.column] = ", ".join(rewritten_urls)
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


def _resolve_video_mime_from_url(url: str) -> str | None:
    path = urlparse(url).path or ""
    for ext, mime in _VIDEO_MIME_BY_EXT.items():
        if path.lower().endswith(ext):
            return mime
    return None


def _resolve_audio_mime_from_url(url: str) -> str | None:
    path = urlparse(url).path or ""
    for ext, mime in _AUDIO_MIME_BY_EXT.items():
        if path.lower().endswith(ext):
            return mime
    return None


def resolve_attachment_values(
    value: str,
    att: AssessmentAttachment,
) -> list[dict[str, JsonValue]]:
    """Resolve one dataset cell into OpenAI-supported content params (by URL)."""
    value = value.strip()
    if not value:
        return []

    item_type = att.type
    # Openai doesn't support video or audio attachments (Responses API)
    if item_type in (VIDEO_COLUMN_TYPE, AUDIO_COLUMN_TYPE):
        return []
    resolved: list[dict[str, JsonValue]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == IMAGE_COLUMN_TYPE:
            resolved.append({"type": "input_image", "image_url": url})
        else:
            resolved.append({"type": "input_file", "file_url": url})
    return resolved


def build_anthropic_attachment_parts(
    value: str,
    att: AssessmentAttachment,
) -> list[dict[str, JsonValue]]:
    """Convert one dataset cell into one or more Anthropic content blocks (by URL)."""
    value = value.strip()
    if not value:
        return []

    item_type = att.type
    # Anthropic doesn't support video or audio attachments
    if item_type in (VIDEO_COLUMN_TYPE, AUDIO_COLUMN_TYPE):
        return []
    blocks: list[dict[str, JsonValue]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == IMAGE_COLUMN_TYPE:
            blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            blocks.append({"type": "document", "source": {"type": "url", "url": url}})
    return blocks


def build_gemini_video_part(
    url: str, video_part_config: dict[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """One Gemini ``fileData`` video part; a YouTube url must carry no mimeType."""
    file_data: dict[str, JsonValue] = {"fileUri": url}
    if (urlparse(url).hostname or "").lower() not in YOUTUBE_HOSTS:
        file_data["mimeType"] = _resolve_video_mime_from_url(url) or _DEFAULT_VIDEO_MIME
    return {"fileData": file_data, **(video_part_config or {})}


def build_gemini_attachment_parts(
    value: str,
    att: AssessmentAttachment,
    video_part_config: dict[str, JsonValue] | None = None,
) -> list[dict[str, JsonValue]]:
    """Convert one dataset cell into one or more Gemini content parts (by URL).

    Mirrors the per-item type routing used for the L2 batch so the same
    image/pdf/video/audio handling applies to prefilter (topic relevance) calls.
    ``video_part_config`` comes from the Google mapper, already wire-shaped.
    """
    value = value.strip()
    if not value:
        return []

    item_type = att.type
    parts: list[dict[str, JsonValue]] = []
    for item_value in split_attachment_urls(value):
        url = to_direct_attachment_url(item_value, item_type)
        if item_type == IMAGE_COLUMN_TYPE:
            mime_type = _guess_image_mime_from_url(url) or _DEFAULT_IMAGE_MIME
            parts.append({"fileData": {"mimeType": mime_type, "fileUri": url}})
        elif item_type == VIDEO_COLUMN_TYPE:
            parts.append(build_gemini_video_part(url, video_part_config))
        elif item_type == AUDIO_COLUMN_TYPE:
            mime_type = _resolve_audio_mime_from_url(url) or _DEFAULT_AUDIO_MIME
            parts.append({"fileData": {"mimeType": mime_type, "fileUri": url}})
        else:
            parts.append({"fileData": {"mimeType": _PDF_MIME, "fileUri": url}})
    return parts
