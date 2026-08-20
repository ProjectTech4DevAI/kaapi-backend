import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select


from app.crud.onboarding import onboard_project
from app.crud import (
    get_organization_by_name,
    get_project_by_name,
    get_user_by_email,
    get_organization_by_id,
)
from app.crud.user_project import add_user_to_project
from app.models import (
    OnboardingRequest,
    OnboardingResponse,
    Organization,
    Project,
    User,
    APIKey,
    Credential,
)
from app.tests.utils.utils import random_lower_string, random_email
from app.tests.utils.test_data import create_test_organization, create_test_project
from app.tests.utils.user import create_random_user


def test_onboard_project_new_organization_project_user(db: Session) -> None:
    """Test onboarding with new organization, project, and user."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()
    user_name = "Test User Onboard"
    openai_key = f"sk-{random_lower_string()}"

    onboard_request = OnboardingRequest(
        organization_name=org_name,
        project_name=project_name,
        email=email,
        password=password,
        user_name=user_name,
        credentials=[{"openai": {"api_key": openai_key}}],
    )

    response = onboard_project(session=db, onboard_in=onboard_request)

    assert isinstance(response, OnboardingResponse)
    assert response.organization_name == org_name
    assert response.project_name == project_name
    assert response.user_email == email
    assert response.api_key is not None
    assert len(response.api_key) > 0

    org = get_organization_by_name(session=db, name=org_name)
    assert org is not None
    assert org.id == response.organization_id
    assert org.name == org_name

    project = get_project_by_name(
        session=db, project_name=project_name, organization_id=org.id
    )
    assert project is not None
    assert project.id == response.project_id
    assert project.name == project_name
    assert project.organization_id == org.id

    user = get_user_by_email(session=db, email=email)
    assert user is not None
    assert user.id == response.user_id
    assert user.email == email
    assert user.full_name == user_name

    api_key = db.exec(
        select(APIKey).where(
            APIKey.user_id == user.id,
            APIKey.project_id == project.id,
            APIKey.organization_id == org.id,
        )
    ).first()

    assert api_key is not None

    credential = db.exec(
        select(Credential).where(
            Credential.organization_id == org.id,
            Credential.project_id == project.id,
            Credential.provider == "openai",
            Credential.is_active.is_(True),
        )
    ).first()
    assert credential is not None


def test_onboard_project_existing_organization(db: Session) -> None:
    """Test onboarding with existing organization but new project and user."""
    # Create existing organization
    existing_org = create_test_organization(db)

    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()
    user_name = "Test User Onboard"

    onboard_request = OnboardingRequest(
        organization_name=existing_org.name,
        project_name=project_name,
        email=email,
        password=password,
        user_name=user_name,
    )

    response = onboard_project(session=db, onboard_in=onboard_request)

    assert response.organization_id == existing_org.id
    assert response.organization_name == existing_org.name

    project = get_project_by_name(
        session=db, project_name=project_name, organization_id=existing_org.id
    )
    assert project is not None
    assert project.organization_id == existing_org.id


def test_onboard_project_existing_user(db: Session) -> None:
    """Test onboarding with existing user but new organization and project."""

    existing_user = create_random_user(db)

    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_request = OnboardingRequest(
        organization_name=org_name,
        project_name=project_name,
        email=existing_user.email,
        password=random_lower_string(),  # This should be ignored for existing user
        user_name="New Name",  # This should be ignored for existing user
    )

    response = onboard_project(session=db, onboard_in=onboard_request)

    # Assert user was reused
    assert response.user_id == existing_user.id
    assert response.user_email == existing_user.email

    # Verify new organization and project were created
    org = get_organization_by_name(session=db, name=org_name)
    assert org is not None
    project = get_project_by_name(
        session=db, project_name=project_name, organization_id=org.id
    )
    assert project is not None


def test_onboard_project_user_already_in_another_project(db: Session) -> None:
    """Onboarding must 409 when the user is already mapped to a different project."""
    existing_project = create_test_project(db)
    existing_user = create_random_user(db)

    add_user_to_project(
        session=db,
        email=existing_user.email,
        organization_id=existing_project.organization_id,
        project_id=existing_project.id,
    )
    db.commit()

    onboard_request = OnboardingRequest(
        organization_name="TestOrgOnboardDifferent",
        project_name="TestProjectOnboardDifferent",
        email=existing_user.email,
        password=random_lower_string(),
    )

    with pytest.raises(HTTPException) as exc_info:
        onboard_project(session=db, onboard_in=onboard_request)

    assert exc_info.value.status_code == 409
    assert "already associated with another project" in str(exc_info.value.detail)


def test_onboard_project_duplicate_project_name(db: Session) -> None:
    """Test that onboarding fails when project name already exists in organization."""
    # Create existing project
    existing_project = create_test_project(db)

    org = get_organization_by_id(session=db, org_id=existing_project.organization_id)
    email = random_email()
    password = random_lower_string()

    onboard_request = OnboardingRequest(
        organization_name=org.name,
        project_name=existing_project.name,
        email=email,
        password=password,
    )

    with pytest.raises(HTTPException) as exc_info:
        onboard_project(session=db, onboard_in=onboard_request)

    assert exc_info.value.status_code == 409
    assert "Project already exists" in str(exc_info.value.detail)


def test_onboard_project_with_auto_generated_defaults(db: Session) -> None:
    """Test onboarding with minimal input using auto-generated defaults."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    # Only provide required fields
    onboard_request = OnboardingRequest(
        organization_name=org_name,
        project_name=project_name,
        # email, password, user_name will be auto-generated
    )

    response = onboard_project(session=db, onboard_in=onboard_request)

    assert response.user_email is not None
    assert "@kaapi.org" in response.user_email

    user = get_user_by_email(session=db, email=response.user_email)
    assert user is not None
    assert user.full_name == f"{project_name} User"


def test_onboard_project_api_key_generation(db: Session) -> None:
    """Test that API key is properly generated and encrypted."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()

    onboard_request = OnboardingRequest(
        organization_name=org_name,
        project_name=project_name,
        email=email,
    )

    response = onboard_project(session=db, onboard_in=onboard_request)

    assert response.api_key is not None
    assert len(response.api_key) > 10  # Should be a reasonable length

    # Verify API key record exists in database
    user = get_user_by_email(session=db, email=email)
    org = get_organization_by_name(session=db, name=org_name)
    project = get_project_by_name(
        session=db, project_name=project_name, organization_id=org.id
    )

    api_key_record = db.exec(
        select(APIKey).where(
            APIKey.user_id == user.id,
            APIKey.project_id == project.id,
            APIKey.organization_id == org.id,
            APIKey.deleted_at.is_(None),
        )
    ).first()
    assert api_key_record is not None


def test_onboard_project_response_data_integrity(db: Session) -> None:
    """Test that all response data matches what was created in the database."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()
    user_name = "Test User Onboard"
    openai_key = f"sk-{random_lower_string()}"

    onboard_request = OnboardingRequest(
        organization_name=org_name,
        project_name=project_name,
        email=email,
        password=password,
        user_name=user_name,
        credential=[{"openai": {"api_key": openai_key}}],
    )

    response = onboard_project(session=db, onboard_in=onboard_request)

    # Fetch actual records from database
    org = db.get(Organization, response.organization_id)
    project = db.get(Project, response.project_id)
    user = db.get(User, response.user_id)

    # Verify all response data matches database records
    assert org.name == response.organization_name
    assert project.name == response.project_name
    assert project.organization_id == response.organization_id
    assert user.email == response.user_email


def test_onboard_kms_failure_raises_502(db: Session, monkeypatch) -> None:
    """A KMS outage while encrypting credentials is upstream, not our bug."""
    monkeypatch.setattr(
        "app.crud.onboarding.encrypt_credentials",
        lambda _: (_ for _ in ()).throw(
            HTTPException(status_code=502, detail="kms down")
        ),
    )

    with pytest.raises(HTTPException) as exc:
        onboard_project(
            session=db,
            onboard_in=OnboardingRequest(
                organization_name=random_lower_string(),
                project_name=random_lower_string(),
                email=random_email(),
                user_name=random_lower_string(),
                password=random_lower_string(),
                credentials=[{"openai": {"api_key": "sk-test"}}],
            ),
        )

    assert exc.value.status_code == 502


def test_onboard_concurrent_commit_conflict_raises_409(
    db: Session, monkeypatch
) -> None:
    """The pre-checks are racy; a collision at commit is a 409, not a 500."""

    def _raise_integrity_error() -> None:
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    monkeypatch.setattr(db, "commit", _raise_integrity_error)

    with pytest.raises(HTTPException) as exc:
        onboard_project(
            session=db,
            onboard_in=OnboardingRequest(
                organization_name=random_lower_string(),
                project_name=random_lower_string(),
                email=random_email(),
                user_name=random_lower_string(),
                password=random_lower_string(),
            ),
        )

    assert exc.value.status_code == 409
