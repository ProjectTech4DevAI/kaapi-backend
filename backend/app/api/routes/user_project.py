import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.user_project import (
    add_user_to_project,
    get_users_by_project,
    remove_user_from_project,
)
from app.models import (
    AddUsersToProjectRequest,
    Message,
    UserProjectPublic,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user-projects", tags=["User Projects"])


@router.get(
    "/",
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    description=load_description("user_project/list.md"),
    response_model=APIResponse[list[UserProjectPublic]],
)
def list_project_users(
    session: SessionDep,
    auth_context: AuthContextDep,
) -> Any:
    """List all users in the current project."""
    users = get_users_by_project(session=session, project_id=auth_context.project_.id)
    return APIResponse.success_response(data=users)


@router.post(
    "/",
    dependencies=[
        Depends(require_permission(Permission.SUPERUSER)),
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
    ],
    description=load_description("user_project/add.md"),
    response_model=APIResponse[list[UserProjectPublic]],
    status_code=status.HTTP_201_CREATED,
)
def add_project_users(
    session: SessionDep,
    auth_context: AuthContextDep,
    body: AddUsersToProjectRequest,
) -> Any:
    """Add one or more users to the current project by email."""
    results = []

    for entry in body.users:
        user, user_project, created = add_user_to_project(
            session=session,
            email=str(entry.email),
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            full_name=entry.full_name,
        )
        results.append(
            UserProjectPublic(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                inserted_at=user_project.inserted_at,
            )
        )

    session.commit()

    logger.info(
        f"[add_project_users] Users added to project | "
        f"project_id: {auth_context.project_.id}, count: {len(body.users)}"
    )

    return APIResponse.success_response(data=results)


@router.delete(
    "/{user_id}",
    dependencies=[
        Depends(require_permission(Permission.SUPERUSER)),
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
    ],
    description=load_description("user_project/delete.md"),
    response_model=APIResponse[Message],
)
def delete_project_user(
    session: SessionDep,
    auth_context: AuthContextDep,
    user_id: int,
) -> Any:
    """Remove a user from the current project."""
    if user_id == auth_context.user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot remove yourself from the project",
        )

    removed = remove_user_from_project(
        session=session,
        user_id=user_id,
        project_id=auth_context.project_.id,
    )

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this project",
        )

    session.commit()

    logger.info(
        f"[delete_project_user] User removed from project | "
        f"user_id: {user_id}, project_id: {auth_context.project_.id}"
    )

    return APIResponse.success_response(
        data=Message(message="User removed from project successfully")
    )
