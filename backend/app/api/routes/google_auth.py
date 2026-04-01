import logging
from datetime import timedelta

import jwt as pyjwt
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlmodel import Session, and_, select

from app.api.deps import AuthContextDep, SessionDep
from app.core import security
from app.core.config import settings
from app.crud import get_user_by_email
from app.models import (
    APIKey,
    GoogleAuthRequest,
    GoogleAuthResponse,
    Organization,
    Project,
    SelectProjectRequest,
    Token,
    TokenPayload,
    User,
    UserProject,
    UserPublic,
)
from app.utils import load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _get_user_projects(session: Session, user_id: int) -> list[dict]:
    """Query distinct org/project pairs for a user from both UserProject and APIKey tables."""
    # Query from UserProject table
    from_user_project = (
        select(Organization.id, Organization.name, Project.id, Project.name)
        .select_from(UserProject)
        .join(Organization, Organization.id == UserProject.organization_id)
        .join(Project, Project.id == UserProject.project_id)
        .where(
            and_(
                UserProject.user_id == user_id,
                Organization.is_active.is_(True),
                Project.is_active.is_(True),
            )
        )
    )

    # Query from APIKey table (backward compatibility)
    from_api_key = (
        select(Organization.id, Organization.name, Project.id, Project.name)
        .select_from(APIKey)
        .join(Organization, Organization.id == APIKey.organization_id)
        .join(Project, Project.id == APIKey.project_id)
        .where(
            and_(
                APIKey.user_id == user_id,
                APIKey.is_deleted.is_(False),
                Organization.is_active.is_(True),
                Project.is_active.is_(True),
            )
        )
    )

    # Union both queries and deduplicate
    combined = from_user_project.union(from_api_key)
    results = session.exec(combined).all()

    return [
        {
            "organization_id": org_id,
            "organization_name": org_name,
            "project_id": proj_id,
            "project_name": proj_name,
        }
        for org_id, org_name, proj_id, proj_name in results
    ]


def _set_auth_cookies(
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
        path=f"{settings.API_V1_STR}/auth",
    )


def _create_token_pair(
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


def _create_token_and_response(
    user,
    google_profile: dict,
    available_projects: list[dict],
    organization_id: int | None = None,
    project_id: int | None = None,
    requires_project_selection: bool = False,
) -> JSONResponse:
    """Create JWT token pair, build response, and set cookies."""
    access_token, refresh_token = _create_token_pair(
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

    response = JSONResponse(content=response_data.model_dump())
    _set_auth_cookies(response, access_token, refresh_token)
    return response


@router.post(
    "/google",
    description=load_description("auth/google.md"),
    response_model=GoogleAuthResponse,
)
def google_auth(session: SessionDep, body: GoogleAuthRequest) -> JSONResponse:
    """Authenticate a user via Google OAuth ID token."""

    if not settings.GOOGLE_CLIENT_ID:
        logger.error("[google_auth] GOOGLE_CLIENT_ID is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google authentication is not configured",
        )

    # Verify the Google ID token
    try:
        idinfo = id_token.verify_oauth2_token(
            body.token,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        logger.warning(f"[google_auth] Invalid Google token: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired Google token",
        )

    # Ensure the email is verified by Google
    if not idinfo.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google email is not verified",
        )

    email: str = idinfo["email"]

    # Look up user by email
    user = get_user_by_email(session=session, email=email)
    if not user:
        logger.info(f"[google_auth] No account found for email: {email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No account found for this Google email. Please Contact Support to add your account.",
        )

    # Activate user on first Google login
    if not user.is_active:
        user.is_active = True
        session.add(user)
        session.commit()
        session.refresh(user)
        logger.info(f"[google_auth] User activated on first login | user_id: {user.id}")

    google_profile = {
        "email": idinfo.get("email"),
        "name": idinfo.get("name"),
        "picture": idinfo.get("picture"),
        "given_name": idinfo.get("given_name"),
        "family_name": idinfo.get("family_name"),
    }

    # Query user's org/project access
    available_projects = _get_user_projects(session, user.id)

    if len(available_projects) == 1:
        # Auto-select the only org/project
        proj = available_projects[0]
        logger.info(
            f"[google_auth] User authenticated via Google (auto-selected project) | user_id: {user.id}"
        )
        return _create_token_and_response(
            user=user,
            google_profile=google_profile,
            available_projects=available_projects,
            organization_id=proj["organization_id"],
            project_id=proj["project_id"],
        )
    elif len(available_projects) > 1:
        # Multiple projects — return token without org/project, frontend must select
        logger.info(
            f"[google_auth] User authenticated via Google (requires project selection) | user_id: {user.id}"
        )
        return _create_token_and_response(
            user=user,
            google_profile=google_profile,
            available_projects=available_projects,
            requires_project_selection=True,
        )
    else:
        # No projects — return token with user only
        logger.info(
            f"[google_auth] User authenticated via Google (no projects) | user_id: {user.id}"
        )
        return _create_token_and_response(
            user=user,
            google_profile=google_profile,
            available_projects=[],
        )


@router.post(
    "/select-project",
    response_model=Token,
)
def select_project(
    session: SessionDep,
    auth_context: AuthContextDep,
    body: SelectProjectRequest,
) -> JSONResponse:
    """Select a project and get a new JWT token with org/project embedded."""

    user = auth_context.user

    # Verify the user has access to this project via an active API key
    available_projects = _get_user_projects(session, user.id)
    matching = [p for p in available_projects if p["project_id"] == body.project_id]

    if not matching:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this project",
        )

    proj = matching[0]

    access_token, refresh_token = _create_token_pair(
        user.id,
        organization_id=proj["organization_id"],
        project_id=proj["project_id"],
    )

    response = JSONResponse(content=Token(access_token=access_token).model_dump())
    _set_auth_cookies(response, access_token, refresh_token)

    logger.info(
        f"[select_project] Project selected | user_id: {user.id}, project_id: {body.project_id}"
    )
    return response


@router.post(
    "/refresh",
    response_model=Token,
)
def refresh_access_token(request: Request, session: SessionDep) -> JSONResponse:
    """Use a refresh token to get a new access token without re-authenticating."""

    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found",
        )

    # Decode and validate the refresh token
    try:
        payload = pyjwt.decode(
            refresh_token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
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

    # Ensure this is a refresh token, not an access token
    if token_data.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    # Validate the user still exists and is active
    user = session.get(User, token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")

    # Issue a new access token with the same org/project claims
    access_token, new_refresh_token = _create_token_pair(
        user.id,
        organization_id=token_data.org_id,
        project_id=token_data.project_id,
    )

    response = JSONResponse(content=Token(access_token=access_token).model_dump())
    _set_auth_cookies(response, access_token, new_refresh_token)

    logger.info(f"[refresh_access_token] Token refreshed | user_id: {user.id}")
    return response


@router.post("/logout")
def logout() -> JSONResponse:
    """Clear auth cookies to log the user out."""
    response = JSONResponse(content={"message": "Logged out successfully"})

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
        path=f"{settings.API_V1_STR}/auth",
    )

    return response
