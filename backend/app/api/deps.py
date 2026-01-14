from collections.abc import Generator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
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
        return auth_context

    else:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")


AuthContextDep = Annotated[AuthContext, Depends(get_auth_context)]
