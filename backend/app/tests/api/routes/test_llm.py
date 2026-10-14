import pytest
from uuid import uuid4
from unittest.mock import patch

from sqlmodel import Session
from fastapi.testclient import TestClient

from app.crud import JobCrud
from app.models import Job, JobStatus, JobUpdate
from app.models.llm.response import LLMCallResponse
from app.models.llm.request import (
    LLMCallConfig,
    ConfigBlob,
    NativeCompletionConfig,
    QueryParams,
    build_kaapi_completion_config,
)
from app.models.llm import LLMCallRequest
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.llm import (
    create_llm_job,
    create_llm_call_with_response,
    create_llm_call_with_audio_uri_response,
)


@pytest.fixture
def llm_job(db: Session) -> Job:
    return create_llm_job(db)


@pytest.fixture
def llm_response_in_db(
    db: Session, llm_job: Job, user_api_key: TestAuthContext
) -> LLMCallResponse:
    return create_llm_call_with_response(
        db,
        job_id=llm_job.id,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )


def test_llm_call_success(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    """Test successful LLM call with mocked start_llm_job."""
    with patch("app.services.llm.jobs.start_llm_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="What is the capital of France?"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=NativeCompletionConfig(
                        provider="openai-native",
                        type="text",
                        params={
                            "model": "gpt-4",
                            "temperature": 0.7,
                        },
                    )
                )
            ),
            callback_url="https://example.com/callback",
        )

        response = client.post(
            "api/v1/llm/call",
            json=payload.model_dump(mode="json"),
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is True
        assert "response is being generated" in response_data["data"]["message"]

        mock_start_job.assert_called_once()


def test_llm_call_with_kaapi_config(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    """Test LLM call with Kaapi abstracted config."""
    with patch("app.services.llm.jobs.start_llm_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="Explain quantum computing"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=build_kaapi_completion_config(
                        provider="openai",
                        type="text",
                        params={
                            "model": "gpt-4o",
                            "instructions": "You are a physics expert",
                            "temperature": 0.5,
                        },
                    )
                )
            ),
        )

        response = client.post(
            "api/v1/llm/call",
            json=payload.model_dump(mode="json"),
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        mock_start_job.assert_called_once()


def test_llm_call_with_native_config(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    """Test LLM call with native OpenAI config (pass-through mode)."""
    with patch("app.services.llm.jobs.start_llm_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="Native API call test"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=NativeCompletionConfig(
                        provider="openai-native",
                        type="text",
                        params={
                            "model": "gpt-4",
                            "temperature": 0.9,
                            "max_tokens": 500,
                            "top_p": 1.0,
                        },
                    )
                )
            ),
        )

        response = client.post(
            "api/v1/llm/call",
            json=payload.model_dump(mode="json"),
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        mock_start_job.assert_called_once()


def test_llm_call_missing_config(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    """Test LLM call with missing config fails validation."""
    payload = {
        "query": {"input": "Test query"},
        # Missing config field
    }

    response = client.post(
        "api/v1/llm/call",
        json=payload,
        headers=user_api_key_header,
    )

    assert response.status_code == 422


def test_llm_call_invalid_provider(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    """Test LLM call with invalid provider type."""
    payload = {
        "query": {"input": "Test query"},
        "config": {
            "blob": {
                "completion": {
                    "provider": "invalid-provider",
                    "params": {"model": "gpt-4"},
                }
            }
        },
    }

    response = client.post(
        "api/v1/llm/call",
        json=payload,
        headers=user_api_key_header,
    )

    assert response.status_code == 422


def test_llm_call_success_with_guardrails(
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """Test successful LLM call when guardrails are enabled (no validators)."""

    with patch("app.services.llm.jobs.start_llm_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="What is the capital of France?"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=NativeCompletionConfig(
                        provider="openai-native",
                        type="text",
                        params={
                            "model": "gpt-4o",
                            "temperature": 0.7,
                        },
                    )
                )
            ),
            callback_url="https://example.com/callback",
        )

        response = client.post(
            "/api/v1/llm/call",
            json=payload.model_dump(mode="json"),
            headers=user_api_key_header,
        )

        assert response.status_code == 200

        body = response.json()
        assert body["success"] is True
        assert "response is being generated" in body["data"]["message"]

        mock_start_job.assert_called_once()


def test_llm_call_guardrails_bypassed_still_succeeds(
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """If guardrails service is unavailable (bypassed), request should still succeed."""

    with patch("app.services.llm.jobs.start_llm_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="What is the capital of France?"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=NativeCompletionConfig(
                        provider="openai-native",
                        type="text",
                        params={
                            "model": "gpt-4",
                            "temperature": 0.7,
                        },
                    )
                )
            ),
            callback_url="https://example.com/callback",
        )

        response = client.post(
            "/api/v1/llm/call",
            json=payload.model_dump(mode="json"),
            headers=user_api_key_header,
        )

        assert response.status_code == 200

        body = response.json()
        assert body["success"] is True
        assert "response is being generated" in body["data"]["message"]

        mock_start_job.assert_called_once()


def test_get_llm_call_pending(
    client: TestClient,
    user_api_key_header: dict[str, str],
    llm_job,
) -> None:
    """Job in PENDING state returns status with no llm_response."""
    response = client.get(
        f"/api/v1/llm/call/{llm_job.id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["job_id"] == str(llm_job.id)
    assert body["data"]["status"] == "PENDING"
    assert body["data"]["llm_response"] is None


def test_get_llm_call_success(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
    llm_response_in_db: LLMCallResponse,
) -> None:
    """Job in SUCCESS state returns full llm_response with usage."""

    JobCrud(db).update(llm_response_in_db.job_id, JobUpdate(status=JobStatus.SUCCESS))

    response = client.get(
        f"/api/v1/llm/call/{llm_response_in_db.job_id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["status"] == "SUCCESS"
    assert data["llm_response"] is not None
    assert data["llm_response"]["response"]["provider_response_id"] == "resp_abc123"
    assert data["llm_response"]["response"]["provider"] == "openai"
    assert data["llm_response"]["usage"]["input_tokens"] == 10
    assert data["llm_response"]["usage"]["output_tokens"] == 5
    assert data["llm_response"]["usage"]["total_tokens"] == 15


def test_get_llm_call_failed(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
    llm_job,
) -> None:
    JobCrud(db).update(
        llm_job.id,
        JobUpdate(status=JobStatus.FAILED, error_message="Provider timeout"),
    )

    response = client.get(
        f"/api/v1/llm/call/{llm_job.id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "FAILED"
    assert body["data"]["error_message"] == "Provider timeout"
    assert body["data"]["llm_response"] is None


def test_get_llm_call_not_found(
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """Non-existent job_id returns 404."""

    response = client.get(
        f"/api/v1/llm/call/{uuid4()}",
        headers=user_api_key_header,
    )

    assert response.status_code == 404


@pytest.fixture
def llm_audio_uri_job(db: Session, user_api_key: TestAuthContext) -> Job:
    job = create_llm_job(db)
    create_llm_call_with_audio_uri_response(
        db,
        job_id=job.id,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
    )
    return job


def test_get_llm_call_audio_uri_swapped_to_presigned_url(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
    llm_audio_uri_job: Job,
) -> None:
    """Audio content stored with format='uri' is served as format='url' with a presigned URL."""
    presigned = (
        "https://s3.amazonaws.com/kaapi-bucket/audio/output.wav?X-Amz-Signature=abc"
    )
    JobCrud(db).update(llm_audio_uri_job.id, JobUpdate(status=JobStatus.SUCCESS))

    with patch("app.api.routes.llm.get_cloud_storage") as mock_storage:
        mock_storage.return_value.get_signed_url.return_value = presigned

        response = client.get(
            f"/api/v1/llm/call/{llm_audio_uri_job.id}",
            headers=user_api_key_header,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    audio = body["data"]["llm_response"]["response"]["output"]["content"]
    assert audio["format"] == "url"
    assert audio["value"] == presigned


def test_get_llm_call_audio_uri_presigned_failure_returns_empty_value(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
    llm_audio_uri_job: Job,
) -> None:
    """When presigned URL generation fails, format is still 'url' and value is empty string."""
    JobCrud(db).update(llm_audio_uri_job.id, JobUpdate(status=JobStatus.SUCCESS))

    with patch("app.api.routes.llm.get_cloud_storage") as mock_storage:
        mock_storage.return_value.get_signed_url.side_effect = Exception(
            "S3 unavailable"
        )

        response = client.get(
            f"/api/v1/llm/call/{llm_audio_uri_job.id}",
            headers=user_api_key_header,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    audio = body["data"]["llm_response"]["response"]["output"]["content"]
    assert audio["format"] == "url"
    assert audio["value"] == ""
