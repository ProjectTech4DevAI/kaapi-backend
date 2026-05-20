import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.config import settings
from app.crud import get_user_by_email
from app.crud.notification import (
    create_pending_notification,
    mark_notification_failed,
    mark_notification_sent,
)
from app.crud.organization import get_organization_by_id, validate_organization
from app.crud.project import get_project_by_id, validate_project
from app.crud.user_project import (
    add_user_to_project,
    get_users_by_project,
    remove_user_from_project,
)
from app.models import (
    AddUsersToProjectRequest,
    Message,
    NotificationEntityType,
    NotificationProvider,
    NotificationType,
    UserProjectPublic,
)
from app.services.auth import generate_invite_token
from app.utils import (
    APIResponse,
    generate_invite_email,
    load_description,
    mask_string,
    send_email,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user-projects", tags=["User Projects"])


@router.get(
    "",
    description=load_description("user_project/list.md"),
    response_model=APIResponse[list[UserProjectPublic]],
)
def list_project_users(
    session: SessionDep,
    auth_context: AuthContextDep,
    project_id: int,
) -> Any:
    """List all users in a project."""
    users = get_users_by_project(session=session, project_id=project_id)
    return APIResponse.success_response(data=users)


@router.post(
    "",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    description=load_description("user_project/add.md"),
    response_model=APIResponse[list[UserProjectPublic]],
    status_code=status.HTTP_201_CREATED,
)
def add_project_users(
    session: SessionDep,
    body: AddUsersToProjectRequest,
) -> Any:
    """Add one or more users to a project by email."""
    # Validate org and project exist and are active before issuing any invites.
    validate_organization(session=session, org_id=body.organization_id)
    validate_project(session=session, project_id=body.project_id)

    same_project_emails: list[str] = []
    different_project_emails: list[str] = []

    for entry in body.users:
        _, add_status = add_user_to_project(
            session=session,
            email=str(entry.email),
            organization_id=body.organization_id,
            project_id=body.project_id,
            full_name=entry.full_name,
        )
        if add_status == "same_project":
            same_project_emails.append(str(entry.email))
        elif add_status == "different_project":
            different_project_emails.append(str(entry.email))

    if same_project_emails or different_project_emails:
        session.rollback()
        errors = []
        if same_project_emails:
            errors.append(
                f"Already added to this project: {', '.join(same_project_emails)}"
            )
        if different_project_emails:
            errors.append(
                f"Already assigned to another project: {', '.join(different_project_emails)}"
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="; ".join(errors),
        )

    session.commit()

    # Send invitation emails
    organization = get_organization_by_id(session=session, org_id=body.organization_id)
    project = get_project_by_id(session=session, project_id=body.project_id)

    if settings.emails_enabled and organization and project:
        for entry in body.users:
            email_str = str(entry.email)
            invited_user = get_user_by_email(session=session, email=email_str)
            if not invited_user:
                logger.error(
                    f"[add_project_users] Inviting user row missing after add | email: {mask_string(email_str)}"
                )
                continue

            invite_token = generate_invite_token(
                email=email_str,
                organization_id=body.organization_id,
                project_id=body.project_id,
            )
            email_data = generate_invite_email(
                email_to=email_str,
                project_name=project.name,
                organization_name=organization.name,
                invite_token=invite_token,
            )
            notification = create_pending_notification(
                session=session,
                notification_type=NotificationType.INVITE_USER.value,
                provider=NotificationProvider.EMAIL.value,
                recipient_user_id=invited_user.id,
                entity_type=NotificationEntityType.USER.value,
                entity_id=invited_user.id,
                project_id=body.project_id,
                subject=email_data.subject,
                body_template="invite_user_v1",
                payload={
                    "email": email_str,
                    "project_name": project.name,
                    "organization_name": organization.name,
                    "valid_days": settings.INVITE_TOKEN_EXPIRE_HOURS // 24,
                },
            )
            session.commit()
            session.refresh(notification)

            try:
                send_email(
                    email_to=email_str,
                    subject=email_data.subject,
                    html_content=email_data.html_content,
                )
                mark_notification_sent(session=session, notification=notification)
                session.commit()
                logger.info(
                    f"[add_project_users] Invitation email sent | "
                    f"email: {mask_string(email_str)}, notification_id: {notification.id}"
                )
            except Exception as e:
                mark_notification_failed(
                    session=session, notification=notification, reason=str(e)
                )
                session.commit()
                logger.error(
                    f"[add_project_users] Failed to send invitation email | "
                    f"email: {mask_string(email_str)}, notification_id: {notification.id}, error: {e}"
                )

    # Re-fetch all users for this project to return the full list
    results = get_users_by_project(session=session, project_id=body.project_id)

    logger.info(
        f"[add_project_users] Users added to project | "
        f"project_id: {body.project_id}, count: {len(body.users)}"
    )

    return APIResponse.success_response(data=results)


@router.delete(
    "/{user_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    description=load_description("user_project/delete.md"),
    response_model=APIResponse[Message],
)
def delete_project_user(
    session: SessionDep,
    auth_context: AuthContextDep,
    user_id: int,
    project_id: int,
) -> Any:
    """Remove a user from a project."""
    if user_id == auth_context.user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot remove yourself from the project",
        )

    removed = remove_user_from_project(
        session=session,
        user_id=user_id,
        project_id=project_id,
    )

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this project",
        )

    session.commit()

    logger.info(
        f"[delete_project_user] User removed from project | "
        f"user_id: {user_id}, project_id: {project_id}"
    )

    return APIResponse.success_response(
        data=Message(message="User removed from project successfully")
    )
