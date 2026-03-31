from collections.abc import Generator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from pydantic import ValidationError
from sqlmodel import Session

from app.core import security
from app.core.config import settings
from app.core.db import engine
from app.core.security import api_key_manager
from app.crud.organization import validate_organization
from app.crud.project import validate_project
from app.models import (
    AuthContext,
    Organization,
    Project,
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
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")

    organization: Organization | None = None
    project: Project | None = None

    if token_data.org_id:
        organization = validate_organization(session=session, org_id=token_data.org_id)
    if token_data.project_id:
        project = validate_project(session=session, project_id=token_data.project_id)

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

            return auth_context

    # 2. Try Authorization: Bearer <token> header
    if token:
        return _authenticate_with_jwt(session, token)

    # 3. Try access_token cookie
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return _authenticate_with_jwt(session, cookie_token)

    raise HTTPException(status_code=401, detail="Invalid Authorization format")


AuthContextDep = Annotated[AuthContext, Depends(get_auth_context)]
