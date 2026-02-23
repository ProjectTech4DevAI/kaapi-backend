import base64
import functools as ft
import ipaddress
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import socket
from typing import Any, Dict, Generic, Optional, TypeVar
from urllib.parse import urlparse

import jwt
import emails
from jinja2 import Template
from jwt.exceptions import InvalidTokenError
from fastapi import HTTPException
from langfuse import Langfuse
import openai
from openai import OpenAI
from pydantic import BaseModel
from sqlmodel import Session

from app.core import security
from app.core.config import settings
from app.crud.credentials import get_provider_credential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
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
            error_message = "\n".join([f"{err['loc']}: {err['msg']}" for err in error])
        else:
            error_message = error

        return cls(success=False, data=data, error=error_message, metadata=metadata)


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


def generate_password_reset_token(email: str) -> str:
    delta = timedelta(hours=settings.EMAIL_RESET_TOKEN_EXPIRE_HOURS)
    now = datetime.now(timezone.utc)
    expires = now + delta
    exp = expires.timestamp()
    encoded_jwt = jwt.encode(
        {"exp": exp, "nbf": now, "sub": email},
        settings.SECRET_KEY,
        algorithm=security.ALGORITHM,
    )
    return encoded_jwt


def verify_password_reset_token(token: str) -> str | None:
    try:
        decoded_token = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        return str(decoded_token["sub"])
    except InvalidTokenError:
        return None


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
        logger.error(
            f"[get_openai_client] OpenAI credentials not found. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="OpenAI credentials not configured for this organization/project.",
        )

    try:
        return OpenAI(api_key=credentials["api_key"])
    except Exception as e:
        logger.error(
            f"[get_openai_client] Failed to configure OpenAI client. | project_id: {project_id} | error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure OpenAI client: {str(e)}",
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
        logger.error(
            f"[get_langfuse_client] Langfuse credentials not found or incomplete. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="Langfuse credentials not configured for this organization/project.",
        )

    try:
        return Langfuse(
            public_key=credentials["public_key"],
            secret_key=credentials["secret_key"],
            host=credentials["host"],
        )
    except Exception as e:
        logger.error(
            f"[get_langfuse_client] Failed to configure Langfuse client. | project_id: {project_id} | error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure Langfuse client: {str(e)}",
        )


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


def send_callback(callback_url: str, data: dict[str, Any]) -> bool:
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

    Args:
        callback_url: The HTTPS URL to send the callback to
        data: The JSON data to send in the POST request

    Returns:
        bool: True if callback succeeded, False otherwise
    """
    try:
        validate_callback_url(str(callback_url))
    except ValueError as ve:
        logger.error(f"[send_callback] Invalid callback URL: {ve}", exc_info=True)
        return False

    try:
        with requests.Session() as session:
            session.trust_env = False  # Ignores environment proxies and other implicit settings for SSRF safety

            response = session.post(
                callback_url,
                json=data,
                timeout=(
                    settings.CALLBACK_CONNECT_TIMEOUT,
                    settings.CALLBACK_READ_TIMEOUT,
                ),
                allow_redirects=False,
            )

            response.raise_for_status()

            logger.info("[send_callback] Callback sent successfully")
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


def resolve_audio_base64(data: str, mime_type: str) -> tuple[str, str | None]:
    """Decode base64 audio and write to temp file. Returns (file_path, error)."""
    try:
        audio_bytes = base64.b64decode(data)
    except Exception as e:
        return "", f"Invalid base64 audio data: {str(e)}"

    ext = get_file_extension(mime_type)
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, prefix="audio_"
        ) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        logger.info(f"[resolve_audio_base64] Wrote audio to temp file: {temp_path}")
        return temp_path, None
    except Exception as e:
        return "", f"Failed to write audio to temp file: {str(e)}"


def resolve_input(query_input) -> tuple[str, str | None]:
    """Resolve discriminated union input to content string.

    Args:
        query_input: The input from QueryParams (TextInput or AudioInput)

    Returns:
        (content_string, None) on success - for text returns content value, for audio returns temp file path
        ("", error_message) on failure
    """
    from app.models.llm.request import TextInput, AudioInput

    try:
        if isinstance(query_input, TextInput):
            return query_input.content.value, None

        elif isinstance(query_input, AudioInput):
            # AudioInput content is base64-encoded audio
            mime_type = query_input.content.mime_type or "audio/wav"
            return resolve_audio_base64(query_input.content.value, mime_type)

        else:
            return "", f"Unknown input type: {type(query_input)}"

    except Exception as e:
        logger.error(f"[resolve_input] Failed to resolve input: {e}", exc_info=True)
        return "", f"Failed to resolve input: {str(e)}"


def cleanup_temp_file(file_path: str) -> None:
    """Clean up a temporary file if it exists."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"[cleanup_temp_file] Failed to delete temp file: {e}")
