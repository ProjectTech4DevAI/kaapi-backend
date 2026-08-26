from collections.abc import Generator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from opentelemetry import trace
from pydantic import ValidationError
from sqlmodel import Session

from sqlmodel import and_, select

from app.core import security
from app.core.config import settings
from app.core.db import engine
from app.core.security import api_key_manager
from app.core.telemetry import set_request_log_context
from app.crud.organization import validate_organization
from app.crud.project import validate_project
from app.models import (
    APIKey,
    AuthContext,
    Organization,
    Project,
    TokenPayload,
    User,
    UserProject,
)


reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token", auto_error=False
)


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
SessionDep = Annotated[Session, Depends(get_db)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]


def _set_tenant_span_attributes(auth_context: AuthContext) -> None:
    """Tag the active OTel span and log context with tenant info after auth.

    Sets org/project on:
    - OTel span   → Sentry traces filterable by tenant
    - log context → every log record in this request carries org_id/project_id
    - Sentry scope → tags on all events for this request
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("user.id", str(auth_context.user.id))
        if auth_context.organization:
            span.set_attribute("tenant.org_id", auth_context.organization.id)
        if auth_context.project:
            span.set_attribute("tenant.project_id", auth_context.project.id)

    set_request_log_context(
        org_id=auth_context.organization.id if auth_context.organization else None,
        project_id=auth_context.project.id if auth_context.project else None,
        user_id=auth_context.user.id,
    )


def _authenticate_with_jwt(session: Session, token: str) -> AuthContext:
    """Validate a JWT token and return the authenticated user context."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except (InvalidTokenError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )

    # Reject refresh tokens — they should only be used at /auth/refresh
    if token_data.type == "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh tokens cannot be used for API access",
        )

    user = session.get(User, token_data.sub)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User access has been revoked",
        )

    organization: Organization | None = None
    project: Project | None = None

    if token_data.org_id:
        organization = validate_organization(session=session, org_id=token_data.org_id)
    if token_data.project_id:
        project = validate_project(session=session, project_id=token_data.project_id)

    # Verify user still has access to this project
    if project:
        has_access = session.exec(
            select(UserProject.id)
            .where(
                and_(
                    UserProject.user_id == user.id,
                    UserProject.project_id == project.id,
                )
            )
            .limit(1)
        ).first()

        if not has_access:
            # Fallback: check APIKey table for backward compatibility
            has_api_key = session.exec(
                select(APIKey.id)
                .where(
                    and_(
                        APIKey.user_id == user.id,
                        APIKey.project_id == project.id,
                        APIKey.deleted_at.is_(None),
                    )
                )
                .limit(1)
            ).first()

            if not has_api_key:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User access to this project has been revoked",
                )

    return AuthContext(user=user, organization=organization, project=project)


def get_auth_context(
    request: Request,
    session: SessionDep,
    token: TokenDep,
    api_key: Annotated[str, Depends(api_key_header)],
) -> AuthContext:
    """
    Verify valid authentication (API Key, JWT token, or cookie) and return authenticated user context.
    Returns AuthContext with user info, project_id, and organization_id.
    Authorization logic should be handled in routes.

    Authentication priority:
    1. X-API-KEY header
    2. Authorization: Bearer <token> header
    3. access_token cookie
    """
    # 1. Try X-API-KEY header
    if api_key:
        auth_context = api_key_manager.verify(session, api_key)
        if auth_context:
            if not auth_context.user.is_active:
                raise HTTPException(status_code=403, detail="Inactive user")

            if not auth_context.organization.is_active:
                raise HTTPException(status_code=403, detail="Inactive Organization")

            if not auth_context.project.is_active:
                raise HTTPException(status_code=403, detail="Inactive Project")

            _set_tenant_span_attributes(auth_context)
            return auth_context

    # 2. Try Authorization: Bearer <token> header
    if token:
        auth_context = _authenticate_with_jwt(session, token)
        _set_tenant_span_attributes(auth_context)
        return auth_context

    # 3. Try access_token cookie
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        auth_context = _authenticate_with_jwt(session, cookie_token)
        _set_tenant_span_attributes(auth_context)
        return auth_context

    raise HTTPException(status_code=401, detail="Invalid Authorization format")


AuthContextDep = Annotated[AuthContext, Depends(get_auth_context)]
