from unittest.mock import patch
from fastapi.testclient import TestClient

from app.models import ResponsesAPIRequest


def test_responses_async_success(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    with patch("app.api.routes.responses.start_job") as mock_start_job:
        payload = ResponsesAPIRequest(
            assistant_id="assistant_123",
            question="What is the capital of France?",
            callback_url="http://example.com/callback",
            response_id="response_123",
            extra_field="extra_value",
        )

        response = client.post(
            "api/v1/responses", json=payload.model_dump(), headers=user_api_key_header
        )

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is True
        assert response_data["data"]["status"] == "processing"
        assert "Your request is being processed" in response_data["data"]["message"]
        assert response_data["data"]["extra_field"] == "extra_value"

        mock_start_job.assert_called_once()
