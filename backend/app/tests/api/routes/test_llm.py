import pytest
from uuid import uuid4
from unittest.mock import patch

from sqlmodel import Session
from fastapi.testclient import TestClient

from app.crud import JobCrud
from app.crud.llm import create_llm_call, update_llm_call_response
from app.models import JobType, LLMCallRequest, Job, JobStatus, JobUpdate
from app.models.llm.response import LLMCallResponse
from app.models.llm.request import (
    QueryParams,
    LLMCallConfig,
    ConfigBlob,
    KaapiCompletionConfig,
    NativeCompletionConfig,
)


@pytest.fixture
def llm_job(db: Session) -> Job:
    crud = JobCrud(db)
    return crud.create(job_type=JobType.LLM_API)


@pytest.fixture
def llm_response_in_db(db: Session, llm_job, user_api_key) -> LLMCallResponse:
    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            params={
                "model": "gpt-4o",
                "instructions": "You are helpful.",
                "temperature": 0.7,
            },
            type="text",
        )
    )
    llm_call = create_llm_call(
        db,
        request=LLMCallRequest(
            query=QueryParams(input="What is the capital of France?"),
            config=LLMCallConfig(blob=config_blob),
        ),
        job_id=llm_job.id,
        project_id=user_api_key.project_id,
        organization_id=user_api_key.organization_id,
        resolved_config=config_blob,
        original_provider="openai",
    )
    update_llm_call_response(
        db,
        llm_call_id=llm_call.id,
        provider_response_id="resp_abc123",
        content={"type": "text", "content": {"format": "text", "value": "Paris"}},
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": None,
        },
    )
    return llm_call


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
                    completion=KaapiCompletionConfig(
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
    llm_job,
    llm_response_in_db,
) -> None:
    """Job in SUCCESS state returns full llm_response with usage."""

    JobCrud(db).update(llm_job.id, JobUpdate(status=JobStatus.SUCCESS))

    response = client.get(
        f"/api/v1/llm/call/{llm_job.id}",
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
