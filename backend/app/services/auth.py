import logging
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlmodel import Session

from app.core import security
from app.core.config import settings
from app.models import (
    GoogleAuthResponse,
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


def generate_invite_token(
    email: str,
    organization_id: int,
    project_id: int,
) -> str:
    """Generate a JWT invitation token for a user."""
    delta = timedelta(hours=settings.INVITE_TOKEN_EXPIRE_HOURS)
    now = datetime.now(timezone.utc)
    expires = now + delta
    to_encode = {
        "exp": expires.timestamp(),
        "nbf": now,
        "sub": email,
        "org_id": organization_id,
        "project_id": project_id,
        "type": "invite",
    }
    return pyjwt.encode(to_encode, settings.SECRET_KEY, algorithm=security.ALGORITHM)


def verify_invite_token(token: str) -> dict | None:
    """
    Verify an invitation token and return the payload.

    Returns dict with email, org_id, project_id or None if invalid.
    """
    try:
        payload = pyjwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        if payload.get("type") != "invite":
            return None
        return {
            "email": payload["sub"],
            "organization_id": payload["org_id"],
            "project_id": payload["project_id"],
        }
    except (InvalidTokenError, KeyError):
        return None
