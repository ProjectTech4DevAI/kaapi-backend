import logging
from typing import Any

from fastapi import APIRouter, Depends
from sqlmodel import func, select

from app.api.deps import (
    AuthContextDep,
    SessionDep,
)
from app.api.permissions import Permission, require_permission
from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.crud import create_user, get_user_by_email, update_user
from app.models import (
    Message,
    UpdatePassword,
    User,
    UserCreate,
    UserPublic,
    UserRegister,
    UsersPublic,
    UserUpdate,
    UserUpdateMe,
)
from app.utils import generate_new_account_email, send_email
from app.core.exception_handlers import HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=UsersPublic,
    include_in_schema=False,
)
def read_users(session: SessionDep, skip: int = 0, limit: int = 100) -> Any:
    count = session.exec(select(func.count()).select_from(User)).one()
    users = session.exec(select(User).offset(skip).limit(limit)).all()
    return UsersPublic(data=users, count=count)


@router.post(
    "/",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=UserPublic,
    include_in_schema=False,
)
def create_user_endpoint(*, session: SessionDep, user_in: UserCreate) -> Any:
    if get_user_by_email(session=session, email=user_in.email):
        logger.error(
            f"[create_user_endpoint] Attempting to create user with existing email | email: {user_in.email}"
        )
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )

    user = create_user(session=session, user_create=user_in)

    if settings.emails_enabled and user_in.email:
        email_data = generate_new_account_email(
            email_to=user_in.email, username=user_in.email, password=user_in.password
        )
        send_email(
            email_to=user_in.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
    return user


@router.patch("/me", response_model=UserPublic)
def update_user_me(
    *, session: SessionDep, user_in: UserUpdateMe, current_user_dep: AuthContextDep
) -> Any:
    current_user = current_user_dep.user
    if user_in.email:
        existing_user = get_user_by_email(session=session, email=user_in.email)
        if existing_user and existing_user.id != current_user.id:
            logger.error(
                f"[update_user_me] Attempting to update user with existing email | email: {user_in.email}, user_id: {current_user.id}"
            )
            raise HTTPException(
                status_code=409, detail="User with this email already exists"
            )

    current_user.sqlmodel_update(user_in.model_dump(exclude_unset=True))
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    logger.info(f"[update_user_me] User updated | user_id: {current_user.id}")
    return current_user


@router.patch("/me/password", response_model=Message)
def update_password_me(
    *, session: SessionDep, body: UpdatePassword, current_user_dep: AuthContextDep
) -> Any:
    current_user = current_user_dep.user
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect password")

    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=400,
            detail="New password cannot be the same as the current one",
        )

    current_user.hashed_password = get_password_hash(body.new_password)
    session.add(current_user)
    session.commit()
    logger.info(f"[update_password_me] Password updated | user_id: {current_user.id}")
    return Message(message="Password updated successfully")


@router.get("/me", response_model=UserPublic)
def read_user_me(current_user_dep: AuthContextDep) -> Any:
    return current_user_dep.user


@router.delete("/me", response_model=Message)
def delete_user_me(session: SessionDep, current_user_dep: AuthContextDep) -> Any:
    current_user = current_user_dep.user
    if current_user.is_superuser:
        logger.error(
            f"[delete_user_me] Attempting to delete superuser account by itself | user_id: {current_user.id}"
        )
        raise HTTPException(
            status_code=403, detail="Super users are not allowed to delete themselves"
        )
    session.delete(current_user)
    session.commit()
    logger.info(f"[delete_user_me] User deleted | user_id: {current_user.id}")
    return Message(message="User deleted successfully")


@router.post(
    "/signup",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=UserPublic,
)
def register_user(session: SessionDep, user_in: UserRegister) -> Any:
    """
    This endpoint allows the registration of a new user and is accessible only by a superuser.
    """
    if get_user_by_email(session=session, email=user_in.email):
        logger.error(
            f"[register_user] Attempting to create user with existing email | email: {user_in.email}"
        )
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system",
        )

    user_create = UserCreate.model_validate(user_in)
    return create_user(session=session, user_create=user_create)


@router.get("/{user_id}", response_model=UserPublic, include_in_schema=False)
def read_user_by_id(
    user_id: int, session: SessionDep, current_user: AuthContextDep
) -> Any:
    user = session.get(User, user_id)
    if user == current_user.user:
        return user

    if not current_user.user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="The user doesn't have enough privileges",
        )

    return user


@router.patch(
    "/{user_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=UserPublic,
    include_in_schema=False,
)
def update_user_endpoint(
    *,
    session: SessionDep,
    user_id: int,
    user_in: UserUpdate,
) -> Any:
    db_user = session.get(User, user_id)
    if not db_user:
        logger.error(f"[update_user_endpoint] User not found | user_id: {user_id}")
        raise HTTPException(
            status_code=404,
            detail="The user with this id does not exist in the system",
        )

    if user_in.email:
        existing_user = get_user_by_email(session=session, email=user_in.email)
        if existing_user and existing_user.id != user_id:
            logger.error(
                f"[update_user_endpoint] Attempting to update user with existing email | email: {user_in.email}, user_id: {user_id}"
            )
            raise HTTPException(
                status_code=409, detail="User with this email already exists"
            )

    return update_user(session=session, db_user=db_user, user_in=user_in)


@router.delete(
    "/{user_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    include_in_schema=False,
)
def delete_user(
    session: SessionDep, current_user: AuthContextDep, user_id: int
) -> Message:
    user = session.get(User, user_id)
    if not user:
        logger.error(f"[delete_user] User not found | user_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    if user == current_user.user:
        logger.error(
            f"[delete_user] Attempting to delete self by superuser | user_id: {current_user.user.id}"
        )
        raise HTTPException(
            status_code=403, detail="Super users are not allowed to delete themselves"
        )

    session.delete(user)
    session.commit()
    logger.info(f"[delete_user] User deleted | user_id: {user.id}")
    return Message(message="User deleted successfully")
