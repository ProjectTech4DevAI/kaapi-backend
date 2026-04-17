import logging
from datetime import timedelta

import jwt as pyjwt
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlmodel import Session

from app.core import security
from app.core.config import settings
from app.models import (
    GoogleAuthResponse,
    InviteTokenPayload,
    Token,
    TokenPayload,
    User,
    UserPublic,
)
from app.utils import APIResponse

logger = logging.getLogger(__name__)


def create_token_pair(
    user_id: int,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> tuple[str, str]:
    """Create an access token and refresh token pair."""
    access_token = security.create_access_token(
        user_id,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        organization_id=organization_id,
        project_id=project_id,
    )
    refresh_token = security.create_refresh_token(
        user_id,
        expires_delta=timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES),
        organization_id=organization_id,
        project_id=project_id,
    )
    return access_token, refresh_token


def set_auth_cookies(
    response: JSONResponse,
    access_token: str,
    refresh_token: str,
) -> None:
    """Set access_token and refresh_token as HTTP-only cookies on the response."""
    is_secure = settings.ENVIRONMENT in ("staging", "production")

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=is_secure,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=is_secure,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def clear_auth_cookies(response: JSONResponse) -> None:
    """Clear access_token and refresh_token cookies from the response."""
    is_secure = settings.ENVIRONMENT in ("staging", "production")

    response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=is_secure,
        samesite="lax",
        path="/",
    )
    response.delete_cookie(
        key="refresh_token",
        httponly=True,
        secure=is_secure,
        samesite="lax",
        path="/",
    )


def build_google_auth_response(
    user: User,
    google_profile: dict,
    available_projects: list[dict],
    organization_id: int | None = None,
    project_id: int | None = None,
    requires_project_selection: bool = False,
) -> JSONResponse:
    """Create JWT token pair, build Google auth response, and set cookies."""
    access_token, refresh_token = create_token_pair(
        user.id,
        organization_id=organization_id,
        project_id=project_id,
    )

    response_data = GoogleAuthResponse(
        access_token=access_token,
        user=UserPublic.model_validate(user),
        google_profile=google_profile,
        requires_project_selection=requires_project_selection,
        available_projects=available_projects,
    )

    api_response = APIResponse.success_response(data=response_data)
    response = JSONResponse(content=api_response.model_dump())
    set_auth_cookies(response, access_token, refresh_token)
    return response


def build_token_response(
    user_id: int,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> JSONResponse:
    """Create JWT token pair, build token response, and set cookies."""
    access_token, refresh_token = create_token_pair(
        user_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    api_response = APIResponse.success_response(data=Token(access_token=access_token))
    response = JSONResponse(content=api_response.model_dump())
    set_auth_cookies(response, access_token, refresh_token)
    return response


def validate_refresh_token(
    session: Session, refresh_token_value: str
) -> tuple[User, TokenPayload]:
    """
    Validate a refresh token and return the user and token data.

    Raises HTTPException on invalid/expired token or inactive user.
    """
    try:
        payload = pyjwt.decode(
            refresh_token_value,
            settings.SECRET_KEY,
            algorithms=[security.ALGORITHM],
        )
        token_data = TokenPayload(**payload)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please login again.",
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if token_data.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    user = session.get(User, token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")

    return user, token_data


def generate_email_token(
    email: str,
    token_type: str,
    expires_delta: timedelta,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> str:
    """Generate a JWT email token (invite or magic_link).

    Args:
        email: User's email address (stored as 'sub' claim)
        token_type: Token type identifier (e.g. "invite", "magic_link")
        expires_delta: Token expiration duration
        organization_id: Optional org ID to embed (for invite tokens)
        project_id: Optional project ID to embed (for invite tokens)
    """
    return security.encode_jwt_token(
        subject=email,
        token_type=token_type,
        expires_delta=expires_delta,
        extra_claims={"org_id": organization_id, "project_id": project_id},
    )


def verify_email_token(token: str, expected_type: str) -> dict | None:
    """Verify a JWT email token and return its claims as a dict, or None if invalid."""
    payload = security.decode_jwt_token(token, expected_type=expected_type)
    if not payload or "sub" not in payload:
        return None
    return {
        "email": payload["sub"],
        "organization_id": payload.get("org_id"),
        "project_id": payload.get("project_id"),
    }


def generate_invite_token(email: str, organization_id: int, project_id: int) -> str:
    """Generate a JWT invitation token for a user (expires in INVITE_TOKEN_EXPIRE_HOURS)."""
    return generate_email_token(
        email=email,
        token_type="invite",
        expires_delta=timedelta(hours=settings.INVITE_TOKEN_EXPIRE_HOURS),
        organization_id=organization_id,
        project_id=project_id,
    )


def verify_invite_token(token: str) -> InviteTokenPayload | None:
    """Verify an invitation token and return its payload, or None if invalid."""
    claims = verify_email_token(token, expected_type="invite")
    if (
        not claims
        or claims.get("organization_id") is None
        or claims.get("project_id") is None
    ):
        return None
    return InviteTokenPayload(
        email=claims["email"],
        organization_id=claims["organization_id"],
        project_id=claims["project_id"],
    )


def generate_magic_link_token(email: str) -> str:
    """Generate a short-lived magic link login token (expires in MAGIC_LINK_TOKEN_EXPIRE_MINUTES)."""
    return generate_email_token(
        email=email,
        token_type="magic_link",
        expires_delta=timedelta(minutes=settings.MAGIC_LINK_TOKEN_EXPIRE_MINUTES),
    )


def verify_magic_link_token(token: str) -> str | None:
    """Verify a magic link token. Returns email string or None."""
    result = verify_email_token(token, expected_type="magic_link")
    return result["email"] if result else None
