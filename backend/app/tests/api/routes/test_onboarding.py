from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.tests.utils.utils import random_email, random_lower_string
from app.tests.utils.test_data import create_test_organization


def test_onboard_project_new_organization_project_user(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding with new organization, project, and user."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()
    user_name = "Test User Onboard"
    openai_key = f"sk-{random_lower_string()}"
    langfuse_secret_key = f"sk-lf-{random_lower_string()}"
    langfuse_public_key = f"pk-lf-{random_lower_string()}"
    langfuse_host = "https://cloud.langfuse.com"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "email": email,
        "password": password,
        "user_name": user_name,
        "credentials": [
            {"openai": {"api_key": openai_key}},
            {
                "langfuse": {
                    "secret_key": langfuse_secret_key,
                    "public_key": langfuse_public_key,
                    "host": langfuse_host,
                }
            },
        ],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()

    # Check the response structure
    assert "data" in response_data
    assert "success" in response_data
    assert response_data["success"] is True

    data = response_data["data"]
    assert data["organization_name"] == org_name
    assert data["project_name"] == project_name
    assert data["user_email"] == email
    assert "api_key" in data
    assert len(data["api_key"]) > 0
    assert "organization_id" in data
    assert "project_id" in data
    assert "user_id" in data


def test_onboard_project_existing_organization(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding with existing organization but new project and user."""
    # Create existing organization
    existing_org = create_test_organization(db)

    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()
    user_name = "Test User Onboard"

    onboard_data = {
        "organization_name": existing_org.name,
        "project_name": project_name,
        "email": email,
        "password": password,
        "user_name": user_name,
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()

    data = response_data["data"]
    assert data["organization_id"] == existing_org.id
    assert data["organization_name"] == existing_org.name
    assert data["project_name"] == project_name
    assert data["user_email"] == email


def test_onboard_project_duplicate_project_in_organization(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when project already exists in the organization."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "email": email,
        "password": password,
    }

    # First request should succeed
    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )
    assert response.status_code == 201

    # Second request with same org and project should fail
    email2 = random_email()
    onboard_data["email"] = email2

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 409
    error_response = response.json()
    assert "error" in error_response
    assert "Project already exists" in error_response["error"]


def test_onboard_project_with_auto_generated_defaults(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding with minimal input using auto-generated defaults."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    # Only provide required fields
    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        # email, password, user_name will be auto-generated
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()

    data = response_data["data"]
    assert data["organization_name"] == org_name
    assert data["project_name"] == project_name
    assert data["user_email"] is not None
    assert "@kaapi.org" in data["user_email"]
    assert "api_key" in data
    assert len(data["api_key"]) > 0


def test_onboard_project_invalid_provider(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when an unsupported provider is specified."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "email": email,
        "password": password,
        "user_name": "User",
        "credentials": [{"totally_not_a_provider": {"foo": "bar"}}],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any("Unsupported provider" in e["message"] for e in error_response["errors"])


def test_onboard_project_non_dict_values_in_credential(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when credential value for a provider is not an object/dict."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "email": email,
        "password": password,
        "user_name": "User",
        "credentials": [{"openai": "sk-should-be-inside-object"}],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "must be an object/dict" in e["message"] for e in error_response["errors"]
    )


def test_onboard_project_missing_required_fields_for_openai(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when OpenAI credential is missing required fields."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "email": email,
        "password": password,
        "user_name": "User",
        "credentials": [{"openai": {}}],  # missing api_key
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "Missing required fields for openai" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_project_missing_required_fields_for_langfuse(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when Langfuse credential is missing required fields."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "email": email,
        "password": password,
        "user_name": "User",
        "credentials": [
            {"langfuse": {"secret_key": "sk-only"}}
        ],  # missing public_key/host
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "Missing required fields for langfuse" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_project_aggregates_multiple_credential_errors(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding reports credential validation errors (fails on first error)."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    email = random_email()
    password = random_lower_string()

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "email": email,
        "password": password,
        "user_name": "User",
        "credentials": [
            {"notreal": {"x": "y"}},
            {"openai": "should-be-dict"},
        ],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    # Validation fails on the first error (unsupported provider)
    assert any("Unsupported provider" in e["message"] for e in error_response["errors"])


def test_onboard_project_credentials_not_a_list(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when credentials is not a list."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": {"openai": {"api_key": "sk-test"}},  # Should be a list
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    # Pydantic catches this before custom validator - returns type error
    assert any(
        "Input should be a valid list" in e["message"] for e in error_response["errors"]
    )


def test_onboard_project_credentials_string_instead_of_list(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when credentials is a string instead of list."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": "sk-test-key",  # Should be a list
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    # Pydantic catches this before custom validator - returns type error
    assert any(
        "Input should be a valid list" in e["message"] for e in error_response["errors"]
    )


def test_onboard_project_credential_item_not_a_dict(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when a credential item is not a dict."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": ["sk-test-key"],  # Items should be dicts
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    # Pydantic catches this before custom validator - returns type error
    assert any(
        "Input should be a valid dictionary" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_project_credential_item_with_multiple_provider_keys(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when credential item has multiple provider keys."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [
            {
                "openai": {"api_key": "sk-test"},
                "langfuse": {
                    "secret_key": "sk-lf",
                    "public_key": "pk-lf",
                    "host": "https://cloud.langfuse.com",
                },
            }
        ],  # Should have exactly one provider key per dict
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "must have exactly one provider key" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_project_credential_item_empty_dict(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when credential item is an empty dict."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [{}],  # Empty dict - no provider key
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "must have exactly one provider key" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_project_credentials_empty_list(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding succeeds with empty credentials list (treated same as None)."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [],  # Empty list is valid
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    # Should not have metadata about credentials
    assert response_data.get("metadata") is None


def test_onboard_project_with_google_credentials(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding with Google AI Studio credentials."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    google_api_key = f"AIza{random_lower_string()}"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [{"google-aistudio": {"api_key": google_api_key}}],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    assert (
        response_data["metadata"]["note"]
        == "Given credential(s) have been saved for this project."
    )


def test_onboard_project_with_sarvamai_credentials(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding with SarvamAI credentials."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    sarvamai_api_key = f"sarvam-{random_lower_string()}"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [{"sarvamai": {"api_key": sarvamai_api_key}}],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    assert (
        response_data["metadata"]["note"]
        == "Given credential(s) have been saved for this project."
    )


def test_onboard_project_with_elevenlabs_credentials(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding with ElevenLabs credentials."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"
    elevenlabs_api_key = f"el-{random_lower_string()}"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [{"elevenlabs": {"api_key": elevenlabs_api_key}}],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    assert (
        response_data["metadata"]["note"]
        == "Given credential(s) have been saved for this project."
    )


def test_onboard_project_with_all_supported_providers(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding with credentials for all supported providers."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [
            {"openai": {"api_key": f"sk-{random_lower_string()}"}},
            {
                "langfuse": {
                    "secret_key": f"sk-lf-{random_lower_string()}",
                    "public_key": f"pk-lf-{random_lower_string()}",
                    "host": "https://cloud.langfuse.com",
                }
            },
            {"google-aistudio": {"api_key": f"AIza{random_lower_string()}"}},
            {"sarvamai": {"api_key": f"sarvam-{random_lower_string()}"}},
            {"elevenlabs": {"api_key": f"el-{random_lower_string()}"}},
        ],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    assert (
        response_data["metadata"]["note"]
        == "Given credential(s) have been saved for this project."
    )


def test_onboard_project_google_missing_api_key(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when Google AI Studio credential is missing api_key."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [{"google-aistudio": {}}],  # missing api_key
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "Missing required fields for google-aistudio" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_project_sarvamai_missing_api_key(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when SarvamAI credential is missing api_key."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [{"sarvamai": {}}],  # missing api_key
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "Missing required fields for sarvamai" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_project_elevenlabs_missing_api_key(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """Test onboarding fails when ElevenLabs credential is missing api_key."""
    org_name = "TestOrgOnboard"
    project_name = "TestProjectOnboard"

    onboard_data = {
        "organization_name": org_name,
        "project_name": project_name,
        "credentials": [{"elevenlabs": {}}],  # missing api_key
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert error_response["errors"]
    assert any(
        "Missing required fields for elevenlabs" in e["message"]
        for e in error_response["errors"]
    )


# ---------------------------------------------------------------------------
# v2 onboarding (/api/v2/onboard)
# ---------------------------------------------------------------------------


def _google_gcp_credential() -> dict:
    return {
        "google-gcp": {
            "api_key": f"AQ.{random_lower_string()}",
            "project_id": "test-gcp-project",
            "location": "us-central1",
            "sa_key": {
                "type": "service_account",
                "project_id": "test-gcp-project",
                "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----\n",
                "client_email": "svc@test-gcp-project.iam.gserviceaccount.com",
            },
            "gcs_bucket": "test-audio-bucket",
        }
    }


def test_onboard_v2_with_google_gcp_credential(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """v2 accepts google-gcp and reports the saved-credentials note."""
    onboard_data = {
        "organization_name": f"OrgV2{random_lower_string()[:8]}",
        "project_name": f"ProjV2{random_lower_string()[:8]}",
        "credentials": [_google_gcp_credential()],
    }

    response = client.post(
        f"{settings.API_V2_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"]["api_key"]
    assert (
        response_data["metadata"]["note"]
        == "Given credential(s) have been saved for this project."
    )


def test_onboard_v2_without_credentials_has_no_metadata(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    onboard_data = {
        "organization_name": f"OrgV2{random_lower_string()[:8]}",
        "project_name": f"ProjV2{random_lower_string()[:8]}",
    }

    response = client.post(
        f"{settings.API_V2_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["metadata"] is None


def test_onboard_v2_rejects_vanilla_google(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """v2 requires an explicit Gemini backend; vanilla google is 422."""
    onboard_data = {
        "organization_name": f"OrgV2{random_lower_string()[:8]}",
        "project_name": f"ProjV2{random_lower_string()[:8]}",
        "credentials": [{"google": {"api_key": f"AIza{random_lower_string()}"}}],
    }

    response = client.post(
        f"{settings.API_V2_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert any(
        "not accepted on v2 onboarding" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_v1_rejects_google_gcp(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """google-gcp is v2-only; v1 must refuse it."""
    onboard_data = {
        "organization_name": f"OrgV1{random_lower_string()[:8]}",
        "project_name": f"ProjV1{random_lower_string()[:8]}",
        "credentials": [_google_gcp_credential()],
    }

    response = client.post(
        f"{settings.API_V1_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
    error_response = response.json()
    assert any(
        "not accepted on v1 onboarding" in e["message"]
        for e in error_response["errors"]
    )


def test_onboard_v2_payload_over_32kb_returns_413(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """The size check runs as a dependency, so 413 wins even though the
    oversized payload would also fail body validation."""
    onboard_data = {
        "organization_name": f"OrgV2{random_lower_string()[:8]}",
        "project_name": f"ProjV2{random_lower_string()[:8]}",
        "credentials": [{"openai": {"api_key": "sk-" + "x" * (33 * 1024)}}],
    }

    response = client.post(
        f"{settings.API_V2_STR}/onboard",
        json=onboard_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 413
    assert "exceeds the 32768 byte limit" in response.json()["error"]


def test_onboard_v2_payload_under_limit_passes_size_check(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """A small body sails through the dependency and hits normal validation."""
    response = client.post(
        f"{settings.API_V2_STR}/onboard",
        json={"organization_name": "x"},  # missing project_name
        headers=superuser_token_headers,
    )

    assert response.status_code == 422
