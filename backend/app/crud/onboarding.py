import logging
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.core.security import encrypt_credentials, get_password_hash
from app.crud import (
    api_key_manager,
    get_organization_by_name,
    get_project_by_name,
    get_user_by_email,
)
from app.crud.user_project import add_user_to_project
from app.models import (
    APIKey,
    Credential,
    OnboardingRequest,
    OnboardingResponse,
    Organization,
    OrganizationCreate,
    Project,
    ProjectCreate,
    User,
    UserCreate,
)

logger = logging.getLogger(__name__)


def onboard_project(
    session: Session, onboard_in: OnboardingRequest
) -> OnboardingResponse:
    """
    Create or link resources for onboarding.

    - Organization:
    - Create new if `organization_name` does not exist.
    - Otherwise, attach project to existing organization.

    - Project:
    - Create if `project_name` does not exist in org.
    - If already exists, return 409 Conflict.

    - User:
    - Create and link if `email` does not exist.
    - If exists, attach to project.

    - OpenAI API Key (optional):
    - If provided, encrypted and stored as project credentials.
    - If omitted, project is created without OpenAI credentials.
    """
    existing_organization = get_organization_by_name(
        session=session, name=onboard_in.organization_name
    )
    if existing_organization:
        organization = existing_organization
    else:
        org_create = OrganizationCreate(name=onboard_in.organization_name)
        organization = Organization.model_validate(org_create)
        session.add(organization)
        session.flush()

    project = get_project_by_name(
        session=session,
        project_name=onboard_in.project_name,
        organization_id=organization.id,
    )
    if project:
        raise HTTPException(
            status_code=409,
            detail=f"Project already exists for organization '{organization.name}'",
        )

    project_create = ProjectCreate(
        name=onboard_in.project_name, organization_id=organization.id
    )
    project = Project.model_validate(project_create)
    session.add(project)
    session.flush()

    user = get_user_by_email(session=session, email=onboard_in.email)
    if not user:
        user_create = UserCreate(
            email=onboard_in.email,
            full_name=onboard_in.user_name,
            password=onboard_in.password,
        )
        user = User.model_validate(
            user_create,
            update={"hashed_password": get_password_hash(user_create.password)},
        )
        session.add(user)
        session.flush()

    _, mapping_status = add_user_to_project(
        session=session,
        email=user.email,
        organization_id=organization.id,
        project_id=project.id,
        full_name=onboard_in.user_name,
    )
    if mapping_status == "different_project":
        raise HTTPException(
            status_code=409,
            detail=(f"User '{user.email}' is already associated with another project"),
        )

    raw_key, key_prefix, key_hash = api_key_manager.generate()

    api_key = APIKey(
        key_prefix=key_prefix,
        key_hash=key_hash,
        user_id=user.id,
        organization_id=project.organization_id,
        project_id=project.id,
    )

    session.add(api_key)

    created_credentials: list[Credential] = []

    if onboard_in.credentials:
        for item in onboard_in.credentials:
            (provider_str,) = item.keys()
            values = item[provider_str]

            encrypted_credentials = encrypt_credentials(values)

            cred_row = Credential(
                organization_id=organization.id,
                project_id=project.id,
                is_active=True,
                provider=provider_str,
                credential=encrypted_credentials,
            )
            session.add(cred_row)

            created_credentials.append(cred_row)

    # Pre-checks are read-then-write; a concurrent onboard only collides here.
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        logger.warning(
            f"[onboard_project] Conflicting concurrent onboarding | "
            f"org={onboard_in.organization_name}, project={onboard_in.project_name} | {e}"
        )
        raise HTTPException(
            status_code=409,
            detail=f"Project '{onboard_in.project_name}' already exists for "
            f"organization '{onboard_in.organization_name}'",
        )

    cred_ids = [c.id for c in created_credentials]

    logger.info(
        "[onboard_project] Onboarding completed successfully. "
        f"org_id={organization.id}, project_id={project.id}, user_id={user.id}, "
        f"cred_ids={cred_ids}"
    )

    return OnboardingResponse(
        organization_id=organization.id,
        organization_name=organization.name,
        project_id=project.id,
        project_name=project.name,
        user_id=user.id,
        user_email=user.email,
        api_key=raw_key,
    )
