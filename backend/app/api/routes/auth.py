import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.api.deps import AuthContextDep, SessionDep
from app.core.config import settings
from app.crud import get_user_by_email
from app.crud.auth import get_user_accessible_projects
from app.crud.notification import (
    create_pending_notification,
    mark_notification_failed,
    mark_notification_sent,
)
from app.crud.organization import validate_organization
from app.crud.project import validate_project
from app.crud.user_project import get_user_projects
from app.models import (
    GoogleAuthRequest,
    GoogleAuthResponse,
    MagicLinkRequest,
    Message,
    NotificationEntityType,
    NotificationProvider,
    NotificationType,
    SelectProjectRequest,
    Token,
)
from app.services.auth import (
    build_google_auth_response,
    build_token_response,
    clear_auth_cookies,
    generate_magic_link_token,
    validate_refresh_token,
    verify_invite_token,
    verify_magic_link_token,
)
from app.utils import (
    APIResponse,
    generate_magic_link_email,
    load_description,
    mask_string,
    send_email,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/google",
    description=load_description("auth/google.md"),
    response_model=APIResponse[GoogleAuthResponse],
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

    if not idinfo.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google email is not verified",
        )

    email: str = idinfo["email"]

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

    available_projects = get_user_accessible_projects(session=session, user_id=user.id)

    if len(available_projects) == 1:
        proj = available_projects[0]
        logger.info(
            f"[google_auth] User authenticated via Google (auto-selected project) | user_id: {user.id}"
        )
        return build_google_auth_response(
            user=user,
            google_profile=google_profile,
            available_projects=available_projects,
            organization_id=proj["organization_id"],
            project_id=proj["project_id"],
        )
    elif len(available_projects) > 1:
        logger.info(
            f"[google_auth] User authenticated via Google (requires project selection) | user_id: {user.id}"
        )
        return build_google_auth_response(
            user=user,
            google_profile=google_profile,
            available_projects=available_projects,
            requires_project_selection=True,
        )
    else:
        logger.info(
            f"[google_auth] User authenticated via Google (no projects) | user_id: {user.id}"
        )
        return build_google_auth_response(
            user=user,
            google_profile=google_profile,
            available_projects=[],
        )


@router.post(
    "/select-project",
    response_model=APIResponse[Token],
)
def select_project(
    session: SessionDep,
    auth_context: AuthContextDep,
    body: SelectProjectRequest,
) -> JSONResponse:
    """Select a project and get a new JWT token with org/project embedded."""

    user = auth_context.user

    available_projects = get_user_accessible_projects(session=session, user_id=user.id)
    matching = [p for p in available_projects if p["project_id"] == body.project_id]

    if not matching:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this project",
        )

    proj = matching[0]
    response = build_token_response(
        user_id=user.id,
        organization_id=proj["organization_id"],
        project_id=proj["project_id"],
    )

    logger.info(
        f"[select_project] Project selected | user_id: {user.id}, project_id: {body.project_id}"
    )
    return response


@router.post(
    "/refresh",
    response_model=APIResponse[Token],
)
def refresh_access_token(request: Request, session: SessionDep) -> JSONResponse:
    """Use a refresh token to get a new access token without re-authenticating."""

    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found",
        )

    user, token_data = validate_refresh_token(session, refresh_token)

    response = build_token_response(
        user_id=user.id,
        organization_id=token_data.org_id,
        project_id=token_data.project_id,
    )

    logger.info(f"[refresh_access_token] Token refreshed | user_id: {user.id}")
    return response


@router.post(
    "/logout",
    response_model=APIResponse[Message],
)
def logout() -> JSONResponse:
    """Clear auth cookies to log the user out."""
    api_response = APIResponse.success_response(
        data=Message(message="Logged out successfully")
    )
    response = JSONResponse(content=api_response.model_dump())
    clear_auth_cookies(response)
    return response


@router.get(
    "/invite/verify",
    description=load_description("auth/invite_verify.md"),
    response_model=APIResponse[Token],
)
def verify_invitation(session: SessionDep, token: str) -> JSONResponse:
    """Verify an invitation token, activate the user, and log them in."""

    invite_payload = verify_invite_token(token)
    if not invite_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired invitation link",
        )

    # Verify the org/project referenced by the invite still exist and are active.
    # Raises 404 from validate_* if missing.
    validate_organization(session=session, org_id=invite_payload.organization_id)
    validate_project(session=session, project_id=invite_payload.project_id)

    user = get_user_by_email(session=session, email=invite_payload.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User account not found. Please contact support.",
        )

    # Activate user if not already active
    if not user.is_active:
        user.is_active = True
        session.add(user)
        session.commit()
        session.refresh(user)
        logger.info(
            f"[verify_invitation] User activated via invite | user_id: {user.id}"
        )

    response = build_token_response(
        user_id=user.id,
        organization_id=invite_payload.organization_id,
        project_id=invite_payload.project_id,
    )

    logger.info(
        f"[verify_invitation] Invitation verified | user_id: {user.id}, project_id: {invite_payload.project_id}"
    )
    return response


@router.post(
    "/magic-link",
    description=load_description("auth/magic_link.md"),
    response_model=APIResponse[Message],
)
def request_magic_link(session: SessionDep, body: MagicLinkRequest) -> Any:
    """Send a magic link login email to the user."""

    user = get_user_by_email(session=session, email=body.email)
    if not user:
        logger.info(
            f"[request_magic_link] Magic link requested for non-existent email: {mask_string(body.email)}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found for this email.",
        )

    token = generate_magic_link_token(email=body.email)

    if not settings.emails_enabled:
        logger.warning("[request_magic_link] Email sending is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email service is not configured",
        )

    email_data = generate_magic_link_email(
        email_to=body.email,
        magic_link_token=token,
    )
    user_project_link = next(
        iter(get_user_projects(session=session, user_id=user.id)), None
    )
    notification = create_pending_notification(
        session=session,
        notification_type=NotificationType.MAGIC_LINK_LOGIN.value,
        provider=NotificationProvider.EMAIL.value,
        recipient_user_id=user.id,
        entity_type=NotificationEntityType.USER.value,
        entity_id=user.id,
        project_id=user_project_link.project_id if user_project_link else None,
        subject=email_data.subject,
        body_template="magic_link_login_v1",
        payload={
            "email": body.email,
            "valid_minutes": settings.MAGIC_LINK_TOKEN_EXPIRE_MINUTES,
        },
    )
    session.commit()
    session.refresh(notification)

    try:
        send_email(
            email_to=body.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
        mark_notification_sent(session=session, notification=notification)
        session.commit()
        logger.info(
            f"[request_magic_link] Magic link email sent | "
            f"email: {mask_string(body.email)}, notification_id: {notification.id}"
        )
    except Exception as e:
        mark_notification_failed(
            session=session, notification=notification, reason=str(e)
        )
        session.commit()
        logger.error(
            f"[request_magic_link] Failed to send magic link email | "
            f"email: {mask_string(body.email)}, notification_id: {notification.id}, error: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send login email. Please try again later.",
        )

    return APIResponse.success_response(
        data=Message(message="If an account exists, a login link has been sent.")
    )


@router.get(
    "/magic-link/verify",
    description=load_description("auth/magic_link_verify.md"),
    response_model=APIResponse[Token],
)
def verify_magic_link(session: SessionDep, token: str) -> JSONResponse:
    """Verify a magic link token and log the user in."""

    email = verify_magic_link_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired login link. Please request a new one.",
        )

    user = get_user_by_email(session=session, email=email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User account not found",
        )

    # Activate user if not already active
    if not user.is_active:
        user.is_active = True
        session.add(user)
        session.commit()
        session.refresh(user)
        logger.info(
            f"[verify_magic_link] User activated via magic link | user_id: {user.id}"
        )

    # Get user's projects to embed in token
    available_projects = get_user_accessible_projects(session=session, user_id=user.id)

    organization_id = None
    project_id = None
    if len(available_projects) == 1:
        organization_id = available_projects[0]["organization_id"]
        project_id = available_projects[0]["project_id"]

    response = build_token_response(
        user_id=user.id,
        organization_id=organization_id,
        project_id=project_id,
    )

    logger.info(
        f"[verify_magic_link] User logged in via magic link | user_id: {user.id}"
    )
    return response
