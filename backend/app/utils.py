from __future__ import annotations

import base64
import functools as ft
import hashlib
import hmac
import ipaddress
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import requests
import socket

from typing import Any, Dict, Generic, Optional, TypeVar
from urllib.parse import urlparse

import emails
from jinja2 import Template
from fastapi import HTTPException
from langfuse import Langfuse
import openai
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel
from sqlmodel import Session

from app.core import security
from app.core.audio_utils import AudioRef
from app.core.config import settings
from app.crud.credentials import get_provider_credential, get_tracing_credential
from app.models.llm.request import (
    TextInput,
    AudioInput,
    ImageInput,
    PDFInput,
    ImageContent,
    PDFContent,
)
from app.services.llm.providers.base import ContentPart, MultiModalInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB


class ValidationErrorDetail(BaseModel):
    field: str
    message: str


class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    errors: Optional[list[ValidationErrorDetail]] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def success_response(
        cls, data: T, metadata: Optional[Dict[str, Any]] = None
    ) -> "APIResponse[T]":
        return cls(success=True, data=data, error=None, metadata=metadata)

    @classmethod
    def failure_response(
        cls,
        error: str | list,
        data: Optional[T] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "APIResponse[None]":
        if isinstance(error, list):  # to handle cases when error is a list of errors
            structured_errors = []
            for err in error:
                loc = err.get("loc", ())
                parts = [str(p) for p in loc if p != "body"]
                field = ".".join(parts) if parts else "unknown"

                # Strip Pydantic error type prefixes to get better error message
                msg = str(err.get("msg", ""))
                prefixes = ["Value error, ", "Type error, ", "Assertion error, "]
                for prefix in prefixes:
                    if msg.startswith(prefix):
                        msg = msg[len(prefix) :]
                        break

                structured_errors.append(
                    ValidationErrorDetail(field=str(field), message=msg)
                )

            return cls(
                success=False,
                data=data,
                error="Validation failed",
                errors=structured_errors,
                metadata=metadata,
            )

        else:
            return cls(success=False, data=data, error=error, metadata=metadata)


@dataclass
class EmailData:
    html_content: str
    subject: str


def render_email_template(*, template_name: str, context: dict[str, Any]) -> str:
    template_str = (
        Path(__file__).parent / "email-templates" / "build" / template_name
    ).read_text()
    html_content = Template(template_str).render(context)
    return html_content


def send_email(
    *,
    email_to: str,
    subject: str = "",
    html_content: str = "",
) -> None:
    assert settings.emails_enabled, "no provided configuration for email variables"
    message = emails.Message(
        subject=subject,
        html=html_content,
        mail_from=(settings.EMAILS_FROM_NAME, settings.EMAILS_FROM_EMAIL),
    )
    smtp_options = {"host": settings.SMTP_HOST, "port": settings.SMTP_PORT}
    if settings.SMTP_TLS:
        smtp_options["tls"] = True
    elif settings.SMTP_SSL:
        smtp_options["ssl"] = True
    if settings.SMTP_USER:
        smtp_options["user"] = settings.SMTP_USER
    if settings.SMTP_PASSWORD:
        smtp_options["password"] = settings.SMTP_PASSWORD
    response = message.send(to=email_to, smtp=smtp_options)
    logger.info(f"send email result: {response}")


def generate_test_email(email_to: str) -> EmailData:
    project_name = settings.PROJECT_NAME
    subject = f"{project_name} - Test email"
    html_content = render_email_template(
        template_name="test_email.html",
        context={"project_name": settings.PROJECT_NAME, "email": email_to},
    )
    return EmailData(html_content=html_content, subject=subject)


def generate_reset_password_email(email_to: str, email: str, token: str) -> EmailData:
    project_name = settings.PROJECT_NAME
    subject = f"{project_name} - Password recovery for user {email}"
    link = f"{settings.FRONTEND_HOST}/reset-password?token={token}"
    html_content = render_email_template(
        template_name="reset_password.html",
        context={
            "project_name": settings.PROJECT_NAME,
            "username": email,
            "email": email_to,
            "valid_hours": settings.EMAIL_RESET_TOKEN_EXPIRE_HOURS,
            "link": link,
        },
    )
    return EmailData(html_content=html_content, subject=subject)


def generate_new_account_email(
    email_to: str, username: str, password: str
) -> EmailData:
    project_name = settings.PROJECT_NAME
    subject = f"{project_name} - New account for user {username}"
    html_content = render_email_template(
        template_name="new_account.html",
        context={
            "project_name": settings.PROJECT_NAME,
            "username": username,
            "password": password,
            "email": email_to,
            "link": settings.FRONTEND_HOST,
        },
    )
    return EmailData(html_content=html_content, subject=subject)


def generate_invite_email(
    *,
    email_to: str,
    project_name: str,
    organization_name: str,
    invite_token: str,
) -> EmailData:
    app_name = settings.PROJECT_NAME
    subject = f"{app_name} - You've been invited to {project_name}"
    link = f"{settings.FRONTEND_HOST}/invite?token={invite_token}"
    html_content = render_email_template(
        template_name="invite_user.html",
        context={
            "app_name": app_name,
            "project_name": project_name,
            "organization_name": organization_name,
            "link": link,
            "valid_days": settings.INVITE_TOKEN_EXPIRE_HOURS // 24,
        },
    )
    return EmailData(html_content=html_content, subject=subject)


def generate_eval_completion_email(
    *,
    run_name: str,
    project_name: str,
    status: str,
    completed_at: str,
    link: str,
    error_message: str | None = None,
) -> EmailData:
    app_name = settings.PROJECT_NAME
    status_label = "Completed" if status == "completed" else "Failed"
    subject = f"{app_name} - Evaluation {status_label}: {run_name}"
    html_content = render_email_template(
        template_name="eval_completion.html",
        context={
            "app_name": app_name,
            "run_name": run_name,
            "project_name": project_name,
            "status_label": status_label,
            "completed_at": completed_at,
            "link": link,
            "error_message": error_message,
        },
    )
    return EmailData(html_content=html_content, subject=subject)


def generate_magic_link_email(*, email_to: str, magic_link_token: str) -> EmailData:
    app_name = settings.PROJECT_NAME
    subject = f"{app_name} - Sign in to your account"
    link = f"{settings.FRONTEND_HOST}/verify?token={magic_link_token}"
    html_content = render_email_template(
        template_name="magic_link_login.html",
        context={
            "app_name": app_name,
            "email": email_to,
            "link": link,
            "valid_minutes": settings.MAGIC_LINK_TOKEN_EXPIRE_MINUTES,
        },
    )
    return EmailData(html_content=html_content, subject=subject)


def generate_password_reset_token(email: str) -> str:
    return security.encode_jwt_token(
        subject=email,
        token_type="password_reset",
        expires_delta=timedelta(hours=settings.EMAIL_RESET_TOKEN_EXPIRE_HOURS),
    )


def verify_password_reset_token(token: str) -> str | None:
    payload = security.decode_jwt_token(token, expected_type="password_reset")
    return str(payload["sub"]) if payload and "sub" in payload else None


def mask_string(value: str, mask_char: str = "*") -> str:
    if not value:
        return ""

    length = len(value)
    num_mask = length // 2
    start = (length - num_mask) // 2
    end = start + num_mask

    return value[:start] + (mask_char * num_mask) + value[end:]


def get_openai_client(session: Session, org_id: int, project_id: int) -> OpenAI:
    """
    Fetch OpenAI credentials for the current org/project and return a configured client.
    """
    credentials = get_provider_credential(
        session=session,
        org_id=org_id,
        provider="openai",
        project_id=project_id,
    )

    if not credentials or "api_key" not in credentials:
        logger.warning(
            f"[get_openai_client] OpenAI credentials not found. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="OpenAI credentials not configured for this organization/project.",
        )

    try:
        return OpenAI(api_key=credentials["api_key"])
    except Exception as e:
        logger.warning(
            f"[get_openai_client] Failed to configure OpenAI client. | project_id: {project_id} | error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure OpenAI client: {str(e)}",
        )


def get_anthropic_client(session: Session, org_id: int, project_id: int) -> Anthropic:
    """
    Fetch Anthropic credentials for the current org/project and return a configured client.
    """
    credentials = get_provider_credential(
        session=session,
        org_id=org_id,
        provider="anthropic",
        project_id=project_id,
    )

    if not credentials or "api_key" not in credentials:
        logger.warning(
            f"[get_anthropic_client] Anthropic credentials not found. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="Anthropic credentials not configured for this organization/project.",
        )

    try:
        return Anthropic(api_key=credentials["api_key"])
    except Exception as e:
        logger.error(
            f"[get_anthropic_client] Failed to configure Anthropic client. | project_id: {project_id} | error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure Anthropic client: {str(e)}",
        )


def _build_langfuse_client(credentials: dict[str, Any]) -> Langfuse:
    return Langfuse(
        public_key=credentials["public_key"],
        secret_key=credentials["secret_key"],
        host=credentials["host"],
        timeout=60,
    )


def get_langfuse_client(session: Session, org_id: int, project_id: int) -> Langfuse:
    """
    Fetch Langfuse credentials for the current org/project and return a configured client.
    """
    credentials = get_provider_credential(
        session=session,
        org_id=org_id,
        provider="langfuse",
        project_id=project_id,
    )

    if not credentials or not all(
        key in credentials for key in ["public_key", "secret_key", "host"]
    ):
        logger.warning(
            f"[get_langfuse_client] Langfuse credentials not found or incomplete. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="Langfuse credentials not configured for this organization/project.",
        )

    try:
        return _build_langfuse_client(credentials)
    except Exception as e:
        logger.warning(
            f"[get_langfuse_client] Failed to configure Langfuse client. | project_id: {project_id} | error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure Langfuse client: {str(e)}",
        )


def get_tracing_client(
    session: Session, org_id: int, project_id: int
) -> Langfuse | None:
    """Return the Langfuse client when the project opted into tracing, else None
    (never raises), so evaluations degrade to cosine-only instead of failing."""
    credentials = get_tracing_credential(
        session=session,
        org_id=org_id,
        project_id=project_id,
    )

    if not credentials or not all(
        key in credentials for key in ["public_key", "secret_key", "host"]
    ):
        logger.info(
            f"[get_tracing_client] Tracing off or credentials missing; "
            f"skipping Langfuse | project_id: {project_id}"
        )
        return None

    try:
        return _build_langfuse_client(credentials)
    except Exception as e:
        logger.warning(
            f"[get_tracing_client] Failed to configure Langfuse client; "
            f"continuing without tracing | project_id: {project_id} | "
            f"error: {str(e)}",
            exc_info=True,
        )
        return None


def handle_openai_error(e: openai.OpenAIError) -> str:
    if hasattr(e, "body") and isinstance(e.body, dict) and "message" in e.body:
        return e.body["message"]
    elif hasattr(e, "message"):
        return e.message
    elif hasattr(e, "response") and hasattr(e.response, "json"):
        try:
            error_data = e.response.json()
            if isinstance(error_data, dict) and "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict) and "message" in error_info:
                    return error_info["message"]
        except:
            pass
    return str(e)


def _is_private_ip(ip: str) -> tuple[bool, str]:
    """Check if an IP address is private, localhost, or reserved."""
    try:
        ip_obj = ipaddress.ip_address(ip)

        checks = [
            (ip_obj.is_loopback, "loopback/localhost"),
            (ip_obj.is_link_local, "link-local"),
            (ip_obj.is_multicast, "multicast"),
            (ip_obj.is_private, "private"),
            (ip_obj.is_reserved, "reserved"),
        ]

        for is_blocked, reason in checks:
            if is_blocked:
                return (True, reason)

        return (False, "")

    except ValueError:
        return (False, "")


def validate_callback_url(url: str) -> None:
    """
    Validate callback URL to prevent SSRF attacks.

    Blocks:
    - Non-HTTPS URLs
    - Private IP addresses (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Localhost/loopback addresses (127.0.0.0/8, ::1)
    - Link-local addresses (169.254.0.0/16)
    - Cloud metadata endpoints (169.254.169.254)
    - Reserved IP ranges

    Args:
        url: The callback URL to validate

    Raises:
        ValueError: If URL is not allowed
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme != "https":
            raise ValueError(
                f"Only HTTPS URLs are allowed for callbacks. Got: {parsed.scheme}"
            )

        if not parsed.hostname:
            raise ValueError("URL must have a valid hostname")

        addr_info = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or 443,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )

        for info in addr_info:
            ip_address = info[4][0]
            is_blocked, reason = _is_private_ip(ip_address)
            if is_blocked:
                raise ValueError(
                    f"Callback URL resolves to {reason} IP address: {ip_address}. "
                    f"This IP type is not allowed for callbacks."
                )

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error validating callback URL: {str(e)}") from e


def sign_webhook_payload(
    secret: str, raw_body: bytes, timestamp_ms: int | None = None
) -> tuple[str, int]:
    """
    Generate an HMAC-SHA256 signature for a webhook payload.

    Signing string format: "<timestamp_ms>.<raw_body>"
    The receiver must reconstruct the exact same signing string to verify.

    Args:
        secret: Shared HMAC secret (pre-registered by the receiver).
        raw_body: Exact bytes that will be sent in the HTTP body.
        timestamp_ms: Unix timestamp in milliseconds. Generated if not provided.

    Returns:
        (hex_signature, timestamp_ms)
    """
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)

    signing_string = f"{timestamp_ms}.".encode() + raw_body
    signature = hmac.new(
        secret.encode(),
        signing_string,
        hashlib.sha256,
    ).hexdigest()
    return signature, timestamp_ms


def require_organization_for_project(
    project_id: int | None,
    organization_id: int | None,
) -> None:
    """Raise 400 if project_id is provided without organization_id."""
    if project_id is not None and organization_id is None:
        raise HTTPException(
            status_code=400,
            detail="organization_id is required when project_id is set",
        )


def get_webhook_secret(
    project_id: int | None, organization_id: int | None
) -> str | None:
    """Look up the configured webhook signing secret for this project, or None."""
    if project_id is None or organization_id is None:
        return None
    # Imported lazily: app.core.db pulls in app.crud, which imports app.utils,
    # so a top-level import here would deadlock module initialization.
    from app.core.db import engine

    with Session(engine) as session:
        creds = get_provider_credential(
            session=session,
            org_id=organization_id,
            project_id=project_id,
            provider="webhook_secret",
        )
    return creds.get("webhook_secret") if isinstance(creds, dict) else None


def send_callback(
    callback_url: str,
    data: dict[str, Any],
    webhook_secret: str | None = None,
) -> bool:
    """
    Send results to the callback URL (synchronously) with SSRF protection.

    Security features:
    - HTTPS-only enforcement
    - Private IP blocking (RFC 1918)
    - Localhost/loopback blocking
    - Cloud metadata endpoint blocking
    - DNS rebinding protection
    - Redirect following disabled
    - Strict timeouts
    - Optional HMAC-SHA256 signing when webhook_secret is provided

    Args:
        callback_url: The HTTPS URL to send the callback to
        data: The JSON data to send in the POST request
        webhook_secret: If provided, sign the request with HMAC-SHA256 and
            attach X-Webhook-Signature / X-Webhook-Timestamp headers.

    Returns:
        bool: True if callback succeeded, False otherwise
    """
    try:
        validate_callback_url(str(callback_url))
    except ValueError as ve:
        logger.warning(f"[send_callback] Invalid callback URL: {ve}", exc_info=True)
        return False
    try:
        raw_body = json.dumps(data, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}

        if webhook_secret:
            signature, timestamp_ms = sign_webhook_payload(webhook_secret, raw_body)
            headers["X-Webhook-Signature"] = signature
            headers["X-Webhook-Timestamp"] = str(timestamp_ms)
        with requests.Session() as session:
            session.trust_env = False  # Ignores environment proxies and other implicit settings for SSRF safety

            response = session.post(
                callback_url,
                data=raw_body,
                headers=headers,
                timeout=(
                    settings.CALLBACK_CONNECT_TIMEOUT,
                    settings.CALLBACK_READ_TIMEOUT,
                ),
                allow_redirects=False,
            )

            response.raise_for_status()

            logger.info(f"[send_callback] Callback sent successfully to {callback_url}")
            return True

    except requests.RequestException as e:
        logger.error(f"[send_callback] Callback failed: {str(e)}", exc_info=True)
        return False


@ft.singledispatch
def load_description(filename: Path) -> str:
    if not filename.exists():
        this = Path(__file__)
        filename = this.parent.joinpath("api", "docs", filename)

    return filename.read_text()


@load_description.register
def _(filename: str) -> str:
    return load_description(Path(filename))


# Input resolver functions moved from app.services.llm.input_resolver
def get_file_extension(mime_type: str) -> str:
    """Map MIME type to file extension."""
    mime_to_ext = {
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/webm": ".webm",
        "audio/mp4": ".mp4",
        "audio/m4a": ".m4a",
    }
    return mime_to_ext.get(mime_type, ".audio")


def resolve_audio_base64(
    data: str, mime_type: str
) -> tuple["AudioRef | None", str | None]:
    """Decode base64 audio into an in-memory AudioRef."""
    try:
        audio_bytes = base64.b64decode(data)
    except Exception as e:
        return None, f"Invalid base64 audio data: {str(e)}"
    return AudioRef(bytes_=audio_bytes, mime_type=mime_type), None


def download_audio_bytes(url: str) -> tuple[bytes | None, str | None]:
    """Download audio from a public URL. Returns (bytes, error)."""
    try:
        validate_callback_url(str(url))
    except ValueError as e:
        logger.error(
            f"[download_audio_bytes] Invalid public url URL, only supports HTTPS prefixed URLs.: {e}",
            exc_info=True,
        )
        return None, f"[download_audio_bytes] Invalid public URL: {e}"

    try:
        with requests.get(url, timeout=30, stream=True) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("audio/"):
                logger.error(
                    f"[download_audio_bytes] Unexpected Content-Type: {content_type}"
                )
                return None, f"Unexpected Content-Type: {content_type}"

            length = resp.headers.get("Content-Length")
            if length and int(length) > MAX_AUDIO_SIZE:
                logger.error(
                    f"[download_audio_bytes] File too large: {length} bytes. Upto 50 MB audio files are allowed."
                )
                return None, f"File too large : {length} bytes."
            chunks = []
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > MAX_AUDIO_SIZE:
                    logger.error(
                        f"[download_audio_bytes] File size exceeded max size of 50MB during download."
                    )
                    return None, "File exceeded max size during download."
                chunks.append(chunk)

            return b"".join(chunks), None
    except requests.exceptions.Timeout:
        return None, f"Timed out downloading audio from URL: {url}"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code} downloading audio from URL: {url}"
    except Exception as e:
        return None, f"Failed to download audio from URL: {str(e)}"


def resolve_audio_url(url: str, mime_type: str) -> tuple["AudioRef | None", str | None]:
    """Download audio from a public URL into an in-memory AudioRef."""
    audio_bytes, error = download_audio_bytes(url)
    if error or not audio_bytes:
        return None, error
    return AudioRef(bytes_=audio_bytes, mime_type=mime_type), None


def resolve_image_content(image_input: ImageInput) -> list[ImageContent]:
    contents = (
        image_input.content
        if isinstance(image_input.content, list)
        else [image_input.content]
    )
    for c in contents:
        if not c.mime_type:
            c.mime_type = "image/png"
    return contents


def resolve_pdf_content(pdf_input: PDFInput) -> list[PDFContent]:
    contents = (
        pdf_input.content
        if isinstance(pdf_input.content, list)
        else [pdf_input.content]
    )
    for c in contents:
        if not c.mime_type:
            c.mime_type = "application/pdf"
    return contents


def resolve_input(
    query_input,
) -> tuple[
    "str | AudioRef | list[ImageContent] | list[PDFContent] | MultiModalInput | None",
    str | None,
]:
    """Resolve query input to provider-ready format.

    Returns:
        - TextInput: (str, None)
        - AudioInput: (AudioRef, None)
        - ImageInput: (list[ImageContent], None)
        - PDFInput: (list[PDFContent], None)
        - list[QueryInput]: (MultiModalInput, None)
        - Error: (None, error_message)
    """

    try:
        if isinstance(query_input, TextInput):
            return query_input.content.value, None

        elif isinstance(query_input, AudioInput):
            mime_type = query_input.content.mime_type or "audio/wav"
            if query_input.content.format == "url":
                return resolve_audio_url(query_input.content.value, mime_type)
            return resolve_audio_base64(query_input.content.value, mime_type)

        elif isinstance(query_input, ImageInput):
            return resolve_image_content(query_input), None

        elif isinstance(query_input, PDFInput):
            return resolve_pdf_content(query_input), None

        elif isinstance(query_input, list):
            parts: list[ContentPart] = []
            for item in query_input:
                if isinstance(item, TextInput):
                    parts.append(item.content)
                elif isinstance(item, ImageInput):
                    parts.extend(resolve_image_content(item))
                elif isinstance(item, PDFInput):
                    parts.extend(resolve_pdf_content(item))
                elif isinstance(item, AudioInput):
                    return (
                        None,
                        "Audio input is not supported in multimodal. Please use completion type 'stt' for audio processing.",
                    )
                else:
                    return (
                        None,
                        "Unsupported input type in multimodal list. Multimodal only supports text, image, and pdf inputs.",
                    )
            return MultiModalInput(parts=parts), None

        else:
            return None, f"Unknown input type: {type(query_input)}"

    except Exception as e:
        logger.warning(f"[resolve_input] Failed to resolve input: {e}", exc_info=True)
        return None, f"Failed to resolve input: {str(e)}"


def cleanup_temp_file(file_path: str) -> None:
    """Clean up a temporary file if it exists."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"[cleanup_temp_file] Failed to delete temp file: {e}")
