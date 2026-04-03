from collections.abc import Generator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from opentelemetry import trace
from pydantic import ValidationError
from sqlmodel import Session, select

from app.core import security
from app.core.config import settings
from app.core.db import engine
from app.core.security import api_key_manager
from app.crud.organization import validate_organization
from app.models import (
    AuthContext,
    TokenPayload,
    User,
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
    """Enrich the active OTEL span with tenant context for observability."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.set_attribute("user.id", str(auth_context.user.id))
    span.set_attribute("user.email", auth_context.user.email)
    if auth_context.organization:
        span.set_attribute("tenant.org_id", auth_context.organization.id)
        span.set_attribute("tenant.org_name", auth_context.organization.name)
    if auth_context.project:
        span.set_attribute("tenant.project_id", auth_context.project.id)
        span.set_attribute("tenant.project_name", auth_context.project.name)


def get_auth_context(
    session: SessionDep,
    token: TokenDep,
    api_key: Annotated[str, Depends(api_key_header)],
) -> AuthContext:
    """
    Verify valid authentication (API Key or JWT token) and return authenticated user context.
    Returns AuthContext with user info, project_id, and organization_id.
    Authorization logic should be handled in routes.
    """
    if api_key:
        auth_context = api_key_manager.verify(session, api_key)
        if not auth_context:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        if not auth_context.user.is_active:
            raise HTTPException(status_code=403, detail="Inactive user")

        if not auth_context.organization.is_active:
            raise HTTPException(status_code=403, detail="Inactive Organization")

        if not auth_context.project.is_active:
            raise HTTPException(status_code=403, detail="Inactive Project")

        _set_tenant_span_attributes(auth_context)
        return auth_context

    elif token:
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
            )
            token_data = TokenPayload(**payload)
        except (InvalidTokenError, ValidationError):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )

        user = session.get(User, token_data.sub)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Inactive user")

        auth_context = AuthContext(
            user=user,
        )
        _set_tenant_span_attributes(auth_context)
        return auth_context

    else:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")


AuthContextDep = Annotated[AuthContext, Depends(get_auth_context)]
