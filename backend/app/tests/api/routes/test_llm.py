from unittest.mock import patch

from fastapi.testclient import TestClient

from app.models import LLMCallRequest
from app.models.llm.request import (
    QueryParams,
    LLMCallConfig,
    CompletionConfig,
    ConfigBlob,
    KaapiLLMParams,
    KaapiCompletionConfig,
    NativeCompletionConfig,
)


def test_llm_call_success(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    """Test successful LLM call with mocked start_high_priority_job."""
    with patch("app.services.llm.jobs.start_high_priority_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="What is the capital of France?"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=NativeCompletionConfig(
                        provider="openai-native",
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
    with patch("app.services.llm.jobs.start_high_priority_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="Explain quantum computing"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=KaapiCompletionConfig(
                        provider="openai",
                        params=KaapiLLMParams(
                            model="gpt-4o",
                            instructions="You are a physics expert",
                            temperature=0.5,
                        ),
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
    with patch("app.services.llm.jobs.start_high_priority_job") as mock_start_job:
        mock_start_job.return_value = "test-task-id"

        payload = LLMCallRequest(
            query=QueryParams(input="Native API call test"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=NativeCompletionConfig(
                        provider="openai-native",
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
