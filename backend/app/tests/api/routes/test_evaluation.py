import io
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.util import now
from app.crud.evaluations.batch import (
    build_google_evaluation_jsonl,
    build_openai_evaluation_jsonl,
    start_evaluation_batch,
)
from app.crud.evaluations.core import create_evaluation_run
from app.models import (
    BatchJob,
    EvaluationDataset,
    EvaluationRun,
    Organization,
    Project,
)
from app.models.batch_job import BatchJobType
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_config, create_test_evaluation_dataset


# Helper function to create CSV file-like object
def create_csv_file(content: str) -> tuple[str, io.BytesIO]:
    """Create a CSV file-like object for testing."""
    file_obj = io.BytesIO(content.encode("utf-8"))
    return ("test.csv", file_obj)


@pytest.fixture
def valid_csv_content() -> str:
    """Valid CSV content with question and answer columns."""
    return """question,answer
"Who is known as the strongest jujutsu sorcerer?","Satoru Gojo"
"What is the name of Gojo's Domain Expansion?","Infinite Void"
"Who is known as the King of Curses?","Ryomen Sukuna"
"""


@pytest.fixture
def invalid_csv_missing_columns() -> str:
    """CSV content missing required columns."""
    return """query,response
"Who is known as the strongest jujutsu sorcerer?","Satoru Gojo"
"""


@pytest.fixture
def csv_with_empty_rows() -> str:
    """CSV content with some empty rows."""
    return """question,answer
"Who is known as the strongest jujutsu sorcerer?","Satoru Gojo"
"","4"
"Who wrote Romeo and Juliet?",""
"Valid question","Valid answer"
"""


class TestDatasetUploadValidation:
    """Test CSV validation and parsing."""

    def test_upload_dataset_valid_csv(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
        db: Session,
    ) -> None:
        """Test uploading a valid CSV file."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_langfuse_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"

            mock_get_langfuse_client.return_value = Mock()

            mock_langfuse_upload.return_value = ("test_dataset_id", 9)

            filename, file_obj = create_csv_file(valid_csv_content)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset",
                    "description": "Test dataset description",
                    "duplication_factor": 3,
                },
                headers=user_api_key_header,
            )

            assert response.status_code == 200, response.text
            response_data = response.json()
            assert response_data["success"] is True
            data = response_data["data"]

            assert data["dataset_name"] == "test_dataset"
            assert data["original_items"] == 3
            assert data["total_items"] == 9  # 3 items * 3 duplication
            assert data["duplication_factor"] == 3
            assert data["langfuse_dataset_id"] == "test_dataset_id"
            assert data["object_store_url"] == "s3://bucket/datasets/test_dataset.csv"
            assert "dataset_id" in data

            # Verify object store upload was called
            mock_store_upload.assert_called_once()

            mock_langfuse_upload.assert_called_once()

    def test_upload_dataset_missing_columns(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        invalid_csv_missing_columns: str,
    ) -> None:
        """Test uploading CSV with missing required columns."""
        filename, file_obj = create_csv_file(invalid_csv_missing_columns)

        # The CSV validation happens before any mocked functions are called
        # so this test checks the actual validation logic
        response = client.post(
            "/api/v1/evaluations/datasets",
            files={"file": (filename, file_obj, "text/csv")},
            data={
                "dataset_name": "test_dataset",
                "duplication_factor": 5,
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("message", str(response_data))
        )
        assert "question" in error_str.lower() or "answer" in error_str.lower()

    def test_upload_dataset_empty_rows(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        csv_with_empty_rows: str,
    ) -> None:
        """Test uploading CSV with empty rows (should skip them)."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_langfuse_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"
            mock_get_langfuse_client.return_value = Mock()
            mock_langfuse_upload.return_value = ("test_dataset_id", 4)

            filename, file_obj = create_csv_file(csv_with_empty_rows)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset",
                    "duplication_factor": 2,
                },
                headers=user_api_key_header,
            )

            assert response.status_code == 200, response.text
            response_data = response.json()
            assert response_data["success"] is True
            data = response_data["data"]

            # Should only have 2 valid items (first and last rows)
            assert data["original_items"] == 2
            assert data["total_items"] == 4  # 2 items * 2 duplication


class TestDatasetUploadDuplication:
    """Test duplication logic."""

    def test_upload_with_default_duplication(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
    ) -> None:
        """Test uploading with default duplication factor (1)."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_langfuse_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"
            mock_get_langfuse_client.return_value = Mock()
            mock_langfuse_upload.return_value = ("test_dataset_id", 3)

            filename, file_obj = create_csv_file(valid_csv_content)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset",
                    # duplication_factor not provided, would default to 1
                },
                headers=user_api_key_header,
            )

            assert response.status_code == 200, response.text
            response_data = response.json()
            assert response_data["success"] is True
            data = response_data["data"]

            assert data["duplication_factor"] == 1
            assert data["original_items"] == 3
            assert data["total_items"] == 3

    def test_upload_with_custom_duplication(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
    ) -> None:
        """Test uploading with custom duplication factor."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_langfuse_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"
            mock_get_langfuse_client.return_value = Mock()
            mock_langfuse_upload.return_value = ("test_dataset_id", 12)

            filename, file_obj = create_csv_file(valid_csv_content)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset",
                    "duplication_factor": 4,
                },
                headers=user_api_key_header,
            )

            assert response.status_code == 200, response.text
            response_data = response.json()
            assert response_data["success"] is True
            data = response_data["data"]

            assert data["duplication_factor"] == 4
            assert data["original_items"] == 3
            assert data["total_items"] == 12  # 3 items * 4 duplication

    def test_upload_with_description(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
        db: Session,
    ) -> None:
        """Test uploading with a description."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_langfuse_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"
            mock_get_langfuse_client.return_value = Mock()
            mock_langfuse_upload.return_value = ("test_dataset_id", 9)

            filename, file_obj = create_csv_file(valid_csv_content)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset_with_description",
                    "description": "This is a test dataset for evaluation",
                    "duplication_factor": 3,
                },
                headers=user_api_key_header,
            )

            assert response.status_code == 200, response.text
            response_data = response.json()
            assert response_data["success"] is True
            data = response_data["data"]

            # Verify the description is stored
            dataset = db.exec(
                select(EvaluationDataset).where(
                    EvaluationDataset.id == data["dataset_id"]
                )
            ).first()

            assert dataset is not None
            assert dataset.description == "This is a test dataset for evaluation"

    def test_upload_with_duplication_factor_below_minimum(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
    ) -> None:
        """Test uploading with duplication factor below minimum (0)."""
        filename, file_obj = create_csv_file(valid_csv_content)

        response = client.post(
            "/api/v1/evaluations/datasets",
            files={"file": (filename, file_obj, "text/csv")},
            data={
                "dataset_name": "test_dataset",
                "duplication_factor": 0,
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422
        response_data = response.json()
        assert response_data["errors"]
        assert any(
            "greater than or equal to 1" in e["message"]
            for e in response_data["errors"]
        )

    def test_upload_with_duplication_factor_above_maximum(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
    ) -> None:
        """Test uploading with duplication factor above maximum (6)."""
        filename, file_obj = create_csv_file(valid_csv_content)

        response = client.post(
            "/api/v1/evaluations/datasets",
            files={"file": (filename, file_obj, "text/csv")},
            data={
                "dataset_name": "test_dataset",
                "duplication_factor": 6,
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422
        response_data = response.json()
        assert response_data["errors"]
        assert any(
            "less than or equal to 5" in e["message"] for e in response_data["errors"]
        )

    def test_upload_with_duplication_factor_boundary_minimum(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
    ) -> None:
        """Test uploading with duplication factor at minimum boundary (1)."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_langfuse_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"
            mock_get_langfuse_client.return_value = Mock()
            mock_langfuse_upload.return_value = ("test_dataset_id", 3)

            filename, file_obj = create_csv_file(valid_csv_content)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset",
                    "duplication_factor": 1,
                },
                headers=user_api_key_header,
            )

            assert response.status_code == 200, response.text
            response_data = response.json()
            assert response_data["success"] is True
            data = response_data["data"]

            assert data["duplication_factor"] == 1
            assert data["original_items"] == 3
            assert data["total_items"] == 3


class TestDatasetUploadErrors:
    """Test error handling."""

    def test_upload_succeeds_without_langfuse_when_tracing_off(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
    ) -> None:
        """With tracing opt-out (get_tracing_client returns None), the dataset is
        created from object store only; no Langfuse upload, no error."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_tracing_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"
            mock_get_tracing_client.return_value = None

            filename, file_obj = create_csv_file(valid_csv_content)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset",
                    "duplication_factor": 5,
                },
                headers=user_api_key_header,
            )

            assert response.status_code == 200, response.text
            data = response.json()["data"]
            assert data["langfuse_dataset_id"] is None
            assert data["object_store_url"] == "s3://bucket/datasets/test_dataset.csv"
            mock_langfuse_upload.assert_not_called()

    def test_upload_returns_500_when_langfuse_upload_raises(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        valid_csv_content: str,
    ) -> None:
        """Tracing enabled but the Langfuse upload fails → 500."""
        with (
            patch("app.core.cloud.get_cloud_storage") as _mock_storage,
            patch(
                "app.services.evaluations.dataset.upload_csv_to_object_store"
            ) as mock_store_upload,
            patch(
                "app.services.evaluations.dataset.get_tracing_client"
            ) as mock_get_tracing_client,
            patch(
                "app.services.evaluations.dataset.upload_dataset_to_langfuse"
            ) as mock_langfuse_upload,
        ):
            mock_store_upload.return_value = "s3://bucket/datasets/test_dataset.csv"
            mock_get_tracing_client.return_value = Mock()
            mock_langfuse_upload.side_effect = Exception("Langfuse down")

            filename, file_obj = create_csv_file(valid_csv_content)

            response = client.post(
                "/api/v1/evaluations/datasets",
                files={"file": (filename, file_obj, "text/csv")},
                data={
                    "dataset_name": "test_dataset",
                    "duplication_factor": 3,
                },
                headers=user_api_key_header,
            )

        assert response.status_code == 500, response.text
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("error", str(response_data))
        )
        assert "Failed to upload dataset to Langfuse" in str(error_str)

    def test_upload_invalid_csv_format(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        """Test uploading invalid CSV format."""
        invalid_csv = "not,a,valid\ncsv format here!!!"
        filename, file_obj = create_csv_file(invalid_csv)

        response = client.post(
            "/api/v1/evaluations/datasets",
            files={"file": (filename, file_obj, "text/csv")},
            data={
                "dataset_name": "test_dataset",
                "duplication_factor": 5,
            },
            headers=user_api_key_header,
        )

        # Should fail validation - check error contains expected message
        assert response.status_code == 422
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("message", str(response_data))
        )
        assert (
            "question" in error_str.lower()
            or "answer" in error_str.lower()
            or "invalid" in error_str.lower()
        )

    def test_upload_without_authentication(self, client, valid_csv_content):
        """Test uploading without authentication."""
        filename, file_obj = create_csv_file(valid_csv_content)

        response = client.post(
            "/api/v1/evaluations/datasets",
            files={"file": (filename, file_obj, "text/csv")},
            data={
                "dataset_name": "test_dataset",
                "duplication_factor": 5,
            },
        )

        assert response.status_code == 401  # Unauthorized


class TestBatchEvaluation:
    """Test batch evaluation endpoint using OpenAI Batch API."""

    @pytest.fixture
    def sample_evaluation_config(self) -> dict[str, Any]:
        """Sample evaluation configuration."""
        return {
            "model": "gpt-4o",
            "temperature": 0.2,
            "instructions": "You are a helpful assistant",
        }

    def test_start_batch_evaluation_invalid_dataset_id(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test batch evaluation fails with invalid dataset_id."""
        # Create a valid config to use
        config = create_test_config(db, project_id=user_api_key.project_id)

        # Try to start evaluation with non-existent dataset_id
        response = client.post(
            "/api/v1/evaluations",
            json={
                "experiment_name": "test_evaluation_run",
                "dataset_id": 99999,  # Non-existent
                "config_id": str(config.id),
                "config_version": 1,
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 404
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("message", str(response_data))
        )
        assert "not found" in error_str.lower() or "not accessible" in error_str.lower()

    def test_start_batch_evaluation_invalid_config_id(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        """Test batch evaluation fails with invalid config_id."""
        # Test with a non-existent config_id (random UUID)
        response = client.post(
            "/api/v1/evaluations",
            json={
                "experiment_name": "test_no_config",
                "dataset_id": 1,  # Dummy ID, config validation happens first
                "config_id": str(uuid4()),  # Non-existent config
                "config_version": 1,
            },
            headers=user_api_key_header,
        )

        # Should fail with either 400 (config not found) or 404 (dataset/config not found)
        assert response.status_code in [400, 404]
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("message", str(response_data))
        )
        # Should mention config or not found
        assert "config" in error_str.lower() or "not found" in error_str.lower()

    def test_start_batch_evaluation_without_authentication(
        self, client, sample_evaluation_config
    ):
        """Test batch evaluation requires authentication."""
        response = client.post(
            "/api/v1/evaluations",
            json={
                "experiment_name": "test_evaluation_run",
                "dataset_id": 1,
                "config": sample_evaluation_config,
            },
        )

        assert response.status_code == 401  # Unauthorized


class TestBatchEvaluationJSONLBuilding:
    """Test JSONL building logic for batch evaluation.

    ``build_openai_evaluation_jsonl`` wraps an already-mapped ``openai_params``
    dict (the output of ``map_kaapi_to_openai_params``) into Responses API batch
    lines: ``body`` is the params dict plus the per-item ``input``. Param shaping
    (temperature inclusion, knowledge_base_ids -> tools) is the mapper's job and
    is covered in tests/services/llm/test_mappers.py, so these tests pass plain
    pre-mapped dicts and assert faithful pass-through.
    """

    def test_build_batch_jsonl_basic(self) -> None:
        """Test basic JSONL building with a fully populated params dict."""
        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "What is 2+2?"},
                "expected_output": {"answer": "4"},
                "metadata": {},
            }
        ]

        openai_params = {
            "model": "gpt-4o",
            "temperature": 0.2,
            "instructions": "You are a helpful assistant",
        }

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        assert len(jsonl_data) == 1
        assert isinstance(jsonl_data[0], dict)

        request = jsonl_data[0]
        assert request["custom_id"] == "item1"
        assert request["method"] == "POST"
        assert request["url"] == "/v1/responses"
        assert request["body"]["model"] == "gpt-4o"
        assert request["body"]["temperature"] == 0.2
        assert request["body"]["instructions"] == "You are a helpful assistant"
        assert request["body"]["input"] == "What is 2+2?"

    def test_build_batch_jsonl_with_tools(self) -> None:
        """Tools present in the params dict are passed through to the body."""
        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Search the docs"},
                "expected_output": {"answer": "Answer from docs"},
                "metadata": {},
            }
        ]

        openai_params = {
            "model": "gpt-4o-mini",
            "instructions": "Search documents",
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_abc123"],
                    "max_num_results": 20,
                }
            ],
        }

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        assert len(jsonl_data) == 1
        request = jsonl_data[0]
        assert request["body"]["tools"][0]["type"] == "file_search"
        assert "vs_abc123" in request["body"]["tools"][0]["vector_store_ids"]

    def test_build_batch_jsonl_minimal_config(self) -> None:
        """Test JSONL building with minimal params (only model)."""
        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test question"},
                "expected_output": {"answer": "Test answer"},
                "metadata": {},
            }
        ]

        openai_params = {"model": "gpt-4o"}

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        assert len(jsonl_data) == 1
        request = jsonl_data[0]
        assert request["body"]["model"] == "gpt-4o"
        assert request["body"]["input"] == "Test question"

    def test_build_batch_jsonl_skips_empty_questions(self) -> None:
        """Test that items with empty questions are skipped."""
        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Valid question"},
                "expected_output": {"answer": "Answer"},
                "metadata": {},
            },
            {
                "id": "item2",
                "input": {"question": ""},  # Empty question
                "expected_output": {"answer": "Answer"},
                "metadata": {},
            },
            {
                "id": "item3",
                "input": {},  # Missing question key
                "expected_output": {"answer": "Answer"},
                "metadata": {},
            },
        ]

        openai_params = {"model": "gpt-4o", "instructions": "Test"}

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        # Should only have 1 valid item
        assert len(jsonl_data) == 1
        assert jsonl_data[0]["custom_id"] == "item1"

    def test_build_batch_jsonl_multiple_items(self) -> None:
        """Test JSONL building with multiple items."""
        dataset_items = [
            {
                "id": f"item{i}",
                "input": {"question": f"Question {i}"},
                "expected_output": {"answer": f"Answer {i}"},
                "metadata": {},
            }
            for i in range(5)
        ]

        openai_params = {
            "model": "gpt-4o",
            "instructions": "Answer questions",
        }

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        assert len(jsonl_data) == 5

        for i, request_dict in enumerate(jsonl_data):
            assert request_dict["custom_id"] == f"item{i}"
            assert request_dict["body"]["input"] == f"Question {i}"
            assert request_dict["body"]["model"] == "gpt-4o"

    def test_build_batch_jsonl_temperature_included_when_explicitly_set(self) -> None:
        """A temperature present in the params dict appears in the JSONL body."""
        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test question"},
                "expected_output": {"answer": "Test answer"},
                "metadata": {},
            }
        ]

        openai_params = {"model": "gpt-4o", "temperature": 0.5}

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        assert len(jsonl_data) == 1
        assert "temperature" in jsonl_data[0]["body"]
        assert jsonl_data[0]["body"]["temperature"] == 0.5

    def test_build_batch_jsonl_temperature_excluded_when_not_set(self) -> None:
        """A temperature absent from the params dict stays absent from the body."""
        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test question"},
                "expected_output": {"answer": "Test answer"},
                "metadata": {},
            }
        ]

        # Mapper omits temperature when it was never set, so it isn't in the dict.
        openai_params = {"model": "gpt-4o"}

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        assert len(jsonl_data) == 1
        assert "temperature" not in jsonl_data[0]["body"]

    def test_build_batch_jsonl_temperature_zero_included_when_explicitly_set(
        self,
    ) -> None:
        """A temperature of 0.0 in the params dict is preserved in the body."""
        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test question"},
                "expected_output": {"answer": "Test answer"},
                "metadata": {},
            }
        ]

        openai_params = {"model": "gpt-4o", "temperature": 0.0}

        jsonl_data = build_openai_evaluation_jsonl(dataset_items, openai_params)

        assert len(jsonl_data) == 1
        assert "temperature" in jsonl_data[0]["body"]
        assert jsonl_data[0]["body"]["temperature"] == 0.0


class TestBuildGoogleEvaluationJSONL:
    """Test JSONL building for Gemini batch evaluation.

    ``build_google_evaluation_jsonl`` wraps an already-mapped ``google_params``
    dict (output of ``map_kaapi_to_google_params``) into Gemini batch lines:
    ``{"key": <item_id>, "request": {...}}``. The request always carries the
    per-item question in ``contents``; ``instructions``/``temperature``/
    ``reasoning`` are folded into ``systemInstruction``/``generationConfig`` only
    when present, so these tests pass pre-mapped dicts and assert that shaping.
    """

    def _items(self, *questions: str) -> list[dict[str, Any]]:
        return [
            {
                "id": f"item{i}",
                "input": {"question": q},
                "expected_output": {"answer": "a"},
                "metadata": {},
            }
            for i, q in enumerate(questions)
        ]

    def test_basic_question_only(self) -> None:
        """A bare params dict produces contents but no system/generation config."""
        jsonl_data = build_google_evaluation_jsonl(
            self._items("What is 2+2?"), {"model": "gemini-2.5-pro"}
        )

        assert len(jsonl_data) == 1
        line = jsonl_data[0]
        assert line["key"] == "item0"
        request = line["request"]
        assert request["contents"] == [
            {"parts": [{"text": "What is 2+2?"}], "role": "user"}
        ]
        assert "systemInstruction" not in request
        assert "generationConfig" not in request

    def test_instructions_become_system_instruction(self) -> None:
        """``instructions`` are mapped into ``systemInstruction``."""
        jsonl_data = build_google_evaluation_jsonl(
            self._items("Q"),
            {"model": "gemini-2.5-pro", "instructions": "Be concise"},
        )

        request = jsonl_data[0]["request"]
        assert request["systemInstruction"] == {"parts": [{"text": "Be concise"}]}

    def test_temperature_in_generation_config(self) -> None:
        """``temperature`` goes into ``generationConfig``."""
        jsonl_data = build_google_evaluation_jsonl(
            self._items("Q"), {"model": "gemini-2.5-pro", "temperature": 0.3}
        )

        assert jsonl_data[0]["request"]["generationConfig"]["temperature"] == 0.3

    def test_temperature_zero_included(self) -> None:
        """A temperature of 0.0 is preserved (not treated as absent)."""
        jsonl_data = build_google_evaluation_jsonl(
            self._items("Q"), {"model": "gemini-2.5-pro", "temperature": 0.0}
        )

        assert jsonl_data[0]["request"]["generationConfig"]["temperature"] == 0.0

    def test_reasoning_becomes_thinking_config(self) -> None:
        """``reasoning`` is mapped into ``thinkingConfig``."""
        jsonl_data = build_google_evaluation_jsonl(
            self._items("Q"), {"model": "gemini-2.5-pro", "reasoning": "high"}
        )

        thinking = jsonl_data[0]["request"]["generationConfig"]["thinkingConfig"]
        assert thinking == {"includeThoughts": False, "thinkingLevel": "high"}

    def test_full_params(self) -> None:
        """All optional fields together populate both config blocks."""
        jsonl_data = build_google_evaluation_jsonl(
            self._items("Q"),
            {
                "model": "gemini-2.5-pro",
                "instructions": "sys",
                "temperature": 0.7,
                "reasoning": "low",
            },
        )

        request = jsonl_data[0]["request"]
        assert request["systemInstruction"] == {"parts": [{"text": "sys"}]}
        assert request["generationConfig"]["temperature"] == 0.7
        assert request["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"

    def test_skips_empty_and_missing_questions(self) -> None:
        """Items with empty or missing questions are dropped."""
        dataset_items = [
            {
                "id": "ok",
                "input": {"question": "Real"},
                "expected_output": {},
                "metadata": {},
            },
            {
                "id": "empty",
                "input": {"question": ""},
                "expected_output": {},
                "metadata": {},
            },
            {"id": "missing", "input": {}, "expected_output": {}, "metadata": {}},
        ]

        jsonl_data = build_google_evaluation_jsonl(
            dataset_items, {"model": "gemini-2.5-pro"}
        )

        assert len(jsonl_data) == 1
        assert jsonl_data[0]["key"] == "ok"

    def test_multiple_items(self) -> None:
        """Each item yields one line keyed by its id."""
        jsonl_data = build_google_evaluation_jsonl(
            self._items("Q0", "Q1", "Q2"), {"model": "gemini-2.5-pro"}
        )

        assert [line["key"] for line in jsonl_data] == ["item0", "item1", "item2"]
        assert jsonl_data[1]["request"]["contents"][0]["parts"][0]["text"] == "Q1"


_BATCH_DATASET_ITEMS = [
    {
        "id": "item1",
        "input": {"question": "What is 2+2?"},
        "expected_output": {"answer": "4"},
        "metadata": {},
    }
]


class TestStartEvaluationBatch:
    """Test start_evaluation_batch, the provider-dispatching orchestrator."""

    @pytest.fixture
    def eval_run(self, db: Session) -> EvaluationRun:
        """A persisted EvaluationRun in the default (pending) state."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=org.id,
            project_id=project.id,
            name="test_dataset_start_batch",
            description="Test dataset",
            original_items_count=1,
            duplication_factor=1,
        )
        config = create_test_config(db, project_id=project.id, use_kaapi_schema=True)
        return create_evaluation_run(
            session=db,
            run_name="start_batch_run",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            organization_id=org.id,
            project_id=project.id,
        )

    def _make_batch_job(
        self, db: Session, eval_run: EvaluationRun, provider: str
    ) -> BatchJob:
        """Persist a BatchJob to stand in for start_batch_job's return value."""
        batch_job = BatchJob(
            provider=provider,
            provider_batch_id="batch_xyz",
            provider_status="submitted",
            job_type=BatchJobType.EVALUATION,
            total_items=1,
            status="submitted",
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        db.add(batch_job)
        db.commit()
        db.refresh(batch_job)
        return batch_job

    @patch("app.crud.evaluations.batch.start_batch_job")
    @patch("app.crud.evaluations.batch.OpenAIBatchProvider")
    @patch("app.crud.evaluations.batch.get_openai_client")
    @patch("app.crud.evaluations.batch.map_kaapi_to_openai_params")
    @patch("app.crud.evaluations.batch.fetch_dataset_items")
    def test_openai_branch(
        self,
        mock_fetch,
        mock_map,
        mock_get_client,
        mock_provider_cls,
        mock_start_batch,
        db: Session,
        eval_run: EvaluationRun,
    ) -> None:
        mock_fetch.return_value = _BATCH_DATASET_ITEMS
        mock_map.return_value = ({"model": "gpt-4o"}, [])
        batch_job = self._make_batch_job(db, eval_run, "openai")
        mock_start_batch.return_value = batch_job

        result = start_evaluation_batch(
            langfuse=MagicMock(),
            session=db,
            eval_run=eval_run,
            params={"model": "gpt-4o"},
            provider="openai",
        )

        assert result.status == "processing"
        assert result.batch_job_id == batch_job.id
        assert result.total_items == 1
        assert mock_start_batch.call_args.kwargs["provider_name"] == "openai"
        mock_get_client.assert_called_once()

    @patch("app.crud.evaluations.batch.start_batch_job")
    @patch("app.crud.evaluations.batch.OpenAIBatchProvider")
    @patch("app.crud.evaluations.batch.get_openai_client")
    @patch("app.crud.evaluations.batch.map_kaapi_to_openai_params")
    @patch("app.crud.evaluations.batch.fetch_dataset_items")
    def test_native_suffix_resolves_to_base_provider(
        self,
        mock_fetch,
        mock_map,
        mock_get_client,
        mock_provider_cls,
        mock_start_batch,
        db: Session,
        eval_run: EvaluationRun,
    ) -> None:
        """An 'openai-native' provider runs through the OpenAI branch."""
        mock_fetch.return_value = _BATCH_DATASET_ITEMS
        mock_map.return_value = ({"model": "gpt-4o"}, ["warn"])
        mock_start_batch.return_value = self._make_batch_job(db, eval_run, "openai")

        result = start_evaluation_batch(
            langfuse=MagicMock(),
            session=db,
            eval_run=eval_run,
            params={"model": "gpt-4o"},
            provider="openai-native",
        )

        assert result.status == "processing"
        assert mock_start_batch.call_args.kwargs["provider_name"] == "openai"

    @patch("app.crud.evaluations.batch.start_batch_job")
    @patch("app.crud.evaluations.batch.GeminiBatchProvider")
    @patch("app.crud.evaluations.batch.GeminiClient")
    @patch("app.crud.evaluations.batch.map_kaapi_to_google_params")
    @patch("app.crud.evaluations.batch.fetch_dataset_items")
    def test_google_branch(
        self,
        mock_fetch,
        mock_map,
        mock_gemini_client,
        mock_provider_cls,
        mock_start_batch,
        db: Session,
        eval_run: EvaluationRun,
    ) -> None:
        mock_fetch.return_value = _BATCH_DATASET_ITEMS
        mock_map.return_value = ({"model": "gemini-2.5-pro"}, [])
        batch_job = self._make_batch_job(db, eval_run, "google-aistudio")
        mock_start_batch.return_value = batch_job

        result = start_evaluation_batch(
            langfuse=MagicMock(),
            session=db,
            eval_run=eval_run,
            params={"model": "gemini-2.5-pro"},
            provider="google-aistudio",
        )

        assert result.status == "processing"
        assert result.batch_job_id == batch_job.id
        assert mock_start_batch.call_args.kwargs["provider_name"] == "google-aistudio"
        mock_gemini_client.from_credentials.assert_called_once()

    @patch("app.crud.evaluations.batch.fetch_dataset_items")
    def test_unsupported_provider_marks_failed(
        self, mock_fetch, db: Session, eval_run: EvaluationRun
    ) -> None:
        mock_fetch.return_value = _BATCH_DATASET_ITEMS

        with pytest.raises(ValueError, match="Unsupported provider"):
            start_evaluation_batch(
                langfuse=MagicMock(),
                session=db,
                eval_run=eval_run,
                params={"model": "claude"},
                provider="anthropic",
            )

        db.refresh(eval_run)
        assert eval_run.status == "failed"
        assert eval_run.error_message

    @patch("app.crud.evaluations.batch.map_kaapi_to_openai_params")
    @patch("app.crud.evaluations.batch.fetch_dataset_items")
    def test_empty_jsonl_marks_failed(
        self, mock_fetch, mock_map, db: Session, eval_run: EvaluationRun
    ) -> None:
        """No questions in the dataset -> empty JSONL -> failure."""
        mock_fetch.return_value = [
            {"id": "x", "input": {}, "expected_output": {}, "metadata": {}}
        ]
        mock_map.return_value = ({"model": "gpt-4o"}, [])

        with pytest.raises(ValueError, match="did not produce any JSONL"):
            start_evaluation_batch(
                langfuse=MagicMock(),
                session=db,
                eval_run=eval_run,
                params={"model": "gpt-4o"},
                provider="openai",
            )

        db.refresh(eval_run)
        assert eval_run.status == "failed"


class TestGetEvaluationRunStatus:
    """Test GET /evaluations/{evaluation_id} endpoint."""

    @pytest.fixture
    def create_test_dataset(
        self, db: Session, user_api_key: TestAuthContext
    ) -> EvaluationDataset:
        """Create a test dataset for evaluation runs."""
        return create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="test_dataset_for_runs",
            description="Test dataset",
            original_items_count=3,
            duplication_factor=1,
        )

    def test_get_evaluation_run_trace_info_not_completed(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        create_test_dataset: EvaluationDataset,
    ) -> None:
        """Test requesting trace info for incomplete evaluation returns error."""
        # Create a config for the evaluation run
        config = create_test_config(db, project_id=user_api_key.project_id)

        eval_run = EvaluationRun(
            run_name="test_pending_run",
            dataset_name=create_test_dataset.name,
            dataset_id=create_test_dataset.id,
            config_id=config.id,
            config_version=1,
            status="pending",
            total_items=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        response = client.get(
            f"/api/v1/evaluations/{eval_run.id}",
            params={"get_trace_info": True},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is False
        assert "only available for completed evaluations" in response_data["error"]
        # Should still include the evaluation run data
        assert response_data["data"]["id"] == eval_run.id

    def test_get_evaluation_run_trace_info_completed(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        create_test_dataset: EvaluationDataset,
    ) -> None:
        """Test requesting trace info for completed evaluation returns cached scores."""
        # Create a config for the evaluation run
        config = create_test_config(db, project_id=user_api_key.project_id)

        eval_run = EvaluationRun(
            run_name="test_completed_run",
            dataset_name=create_test_dataset.name,
            dataset_id=create_test_dataset.id,
            config_id=config.id,
            config_version=1,
            status="completed",
            total_items=3,
            score={
                "traces": [
                    {"trace_id": "trace1", "question": "Q1", "scores": []},
                ],
                "summary_scores": [],
            },
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        response = client.get(
            f"/api/v1/evaluations/{eval_run.id}",
            params={"get_trace_info": True},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert data["id"] == eval_run.id
        assert data["status"] == "completed"
        assert "traces" in data["score"]

    def test_get_evaluation_run_not_found(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        """Test getting non-existent evaluation run returns 404."""
        response = client.get(
            "/api/v1/evaluations/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("error", str(response_data))
        )
        assert "not found" in error_str.lower() or "not accessible" in error_str.lower()

    def test_get_evaluation_run_without_trace_info(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        create_test_dataset: EvaluationDataset,
    ) -> None:
        """Test getting evaluation run without requesting trace info."""
        config = create_test_config(db, project_id=user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="test_simple_run",
            dataset_name=create_test_dataset.name,
            dataset_id=create_test_dataset.id,
            config_id=config.id,
            config_version=1,
            status="completed",
            total_items=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        response = client.get(
            f"/api/v1/evaluations/{eval_run.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert data["id"] == eval_run.id
        assert data["status"] == "completed"

    def test_get_evaluation_run_resync_without_trace_info_fails(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        create_test_dataset: EvaluationDataset,
    ) -> None:
        """Test that resync_score=true requires get_trace_info=true."""
        config = create_test_config(db, project_id=user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="test_run",
            dataset_name=create_test_dataset.name,
            dataset_id=create_test_dataset.id,
            config_id=config.id,
            config_version=1,
            status="completed",
            total_items=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        response = client.get(
            f"/api/v1/evaluations/{eval_run.id}",
            params={"resync_score": True},  # Missing get_trace_info=true
            headers=user_api_key_header,
        )

        assert response.status_code == 400
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("error", str(response_data))
        )
        assert (
            "resync_score" in error_str.lower()
            and "get_trace_info" in error_str.lower()
        )

    def test_get_evaluation_run_grouped_format_without_trace_info_fails(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        create_test_dataset: EvaluationDataset,
    ) -> None:
        eval_run = EvaluationRun(
            run_name="test_run",
            dataset_name=create_test_dataset.name,
            dataset_id=create_test_dataset.id,
            config={"model": "gpt-4o"},
            status="completed",
            total_items=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        response = client.get(
            f"/api/v1/evaluations/{eval_run.id}",
            params={"export_format": "grouped"},  # Missing get_trace_info=true
            headers=user_api_key_header,
        )

        assert response.status_code == 400
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("error", str(response_data))
        )
        assert (
            "export_format" in error_str.lower()
            and "get_trace_info" in error_str.lower()
        )

    def test_get_evaluation_run_grouped_format_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        create_test_dataset: EvaluationDataset,
    ) -> None:
        eval_run = EvaluationRun(
            run_name="test_run",
            dataset_name=create_test_dataset.name,
            dataset_id=create_test_dataset.id,
            config={"model": "gpt-4o"},
            status="completed",
            total_items=4,
            score={
                "traces": [
                    {
                        "trace_id": "trace-1a",
                        "question_id": 1,
                        "question": "What is Python?",
                        "ground_truth_answer": "A programming language",
                        "llm_answer": "Python is a high-level programming language",
                        "scores": [
                            {
                                "name": "cosine_similarity",
                                "value": 0.82,
                                "data_type": "NUMERIC",
                            }
                        ],
                    },
                    {
                        "trace_id": "trace-1b",
                        "question_id": 1,
                        "question": "What is Python?",
                        "ground_truth_answer": "A programming language",
                        "llm_answer": "Python is an interpreted language",
                        "scores": [
                            {
                                "name": "cosine_similarity",
                                "value": 0.75,
                                "data_type": "NUMERIC",
                            }
                        ],
                    },
                    # Row format - 1 trace for question_id=2
                    {
                        "trace_id": "trace-2a",
                        "question_id": 2,
                        "question": "What is Java?",
                        "ground_truth_answer": "An OOP language",
                        "llm_answer": "Java is a statically typed language",
                        "scores": [
                            {
                                "name": "cosine_similarity",
                                "value": 0.80,
                                "data_type": "NUMERIC",
                            }
                        ],
                    },
                ],
                "summary_scores": [
                    {
                        "avg": 0.79,
                        "std": 0.03,
                        "name": "cosine_similarity",
                        "data_type": "NUMERIC",
                        "total_pairs": 3,
                    }
                ],
            },
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        response = client.get(
            f"/api/v1/evaluations/{eval_run.id}",
            params={
                "export_format": "grouped",
                "get_trace_info": True,
            },  # Missing get_trace_info=true
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert data["id"] == eval_run.id
        assert data["status"] == "completed"

        traces = data["score"]["traces"]
        assert (
            isinstance(traces, list)
            and len(traces) > 0
            and "llm_answers" in traces[0]
            and isinstance(traces[0]["llm_answers"], list)
            and "trace_ids" in traces[0]
            and isinstance(traces[0]["trace_ids"], list)
        )


class TestListDatasets:
    """Test GET /evaluations/datasets endpoint."""

    @pytest.fixture
    def create_test_datasets(
        self, db: Session, user_api_key: TestAuthContext
    ) -> list[EvaluationDataset]:
        """Create multiple test datasets."""
        datasets = []
        for i in range(3):
            dataset = create_test_evaluation_dataset(
                db=db,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                name=f"test_dataset_list_{i}",
                description=f"Test dataset {i}",
                original_items_count=i + 1,
                duplication_factor=1,
            )
            datasets.append(dataset)
        return datasets

    def test_list_datasets_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        create_test_datasets: list[EvaluationDataset],
    ) -> None:
        """Test listing datasets returns all datasets."""
        response = client.get(
            "/api/v1/evaluations/datasets",
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_list_datasets_empty(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing datasets when none exist returns empty list."""
        response = client.get(
            "/api/v1/evaluations/datasets",
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert isinstance(data, list)

    def test_list_datasets_pagination(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        create_test_datasets: list[EvaluationDataset],
    ) -> None:
        """Test listing datasets with pagination."""
        response = client.get(
            "/api/v1/evaluations/datasets",
            params={"limit": 2, "offset": 0},
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert isinstance(data, list)
        assert len(data) <= 2

    def test_list_datasets_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test listing datasets requires authentication."""
        response = client.get("/api/v1/evaluations/datasets")
        assert response.status_code == 401


class TestListEvaluationRuns:
    """Test GET /evaluations endpoint."""

    @pytest.fixture
    def create_test_dataset(
        self, db: Session, user_api_key: TestAuthContext
    ) -> EvaluationDataset:
        """Create a test dataset for evaluation runs."""
        return create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="test_dataset_for_list_runs",
            description="Test dataset",
            original_items_count=3,
            duplication_factor=1,
        )

    def test_list_evaluation_runs_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        create_test_dataset: EvaluationDataset,
    ) -> None:
        """Test listing evaluation runs returns results."""
        config = create_test_config(db, project_id=user_api_key.project_id)

        for i in range(2):
            eval_run = EvaluationRun(
                run_name=f"test_list_run_{i}",
                dataset_name=create_test_dataset.name,
                dataset_id=create_test_dataset.id,
                config_id=config.id,
                config_version=1,
                status="completed",
                total_items=3,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )
            db.add(eval_run)
        db.commit()

        response = client.get(
            "/api/v1/evaluations",
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_list_evaluation_runs_empty(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing evaluation runs when none exist."""
        response = client.get(
            "/api/v1/evaluations",
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert isinstance(data, list)

    def test_list_evaluation_runs_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test listing evaluation runs requires authentication."""
        response = client.get("/api/v1/evaluations")
        assert response.status_code == 401


class TestEvaluationRouterOrdering:
    """Regression tests for router ordering (evaluation vs dataset/stt routes).

    The evaluation router has GET /evaluations/{evaluation_id} which can intercept
    requests to GET /evaluations/datasets if registered before the dataset router.
    These tests ensure the router ordering is correct.
    """

    def test_dataset_list_not_intercepted_by_evaluation_route(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """GET /evaluations/datasets must route to dataset list, not evaluation get.

        If evaluation router is registered before dataset router,
        GET /evaluations/datasets would match GET /evaluations/{evaluation_id}
        with evaluation_id='datasets', returning a 422 instead of the dataset list.
        """
        response = client.get(
            "/api/v1/evaluations/datasets",
            headers=user_api_key_header,
        )

        # Must be 200 (dataset list), not 422 (type validation error)
        assert response.status_code == 200, (
            f"Expected 200 but got {response.status_code}. "
            f"This likely means the evaluation router's "
            f"GET /evaluations/{{evaluation_id}} is intercepting "
            f"GET /evaluations/datasets. Check router ordering in "
            f"app/api/routes/evaluations/__init__.py"
        )
        response_data = response.json()
        assert response_data["success"] is True
        assert isinstance(response_data["data"], list)

    def test_evaluation_get_by_id_still_works(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """GET /evaluations/{id} still works correctly after router reordering."""
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        config = create_test_config(db, project_id=user_api_key.project_id)

        eval_run = EvaluationRun(
            run_name="test_router_ordering",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="completed",
            total_items=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        response = client.get(
            f"/api/v1/evaluations/{eval_run.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["data"]["id"] == eval_run.id


class TestGetDataset:
    """Test GET /evaluations/datasets/{dataset_id} endpoint."""

    @pytest.fixture
    def create_test_dataset(
        self, db: Session, user_api_key: TestAuthContext
    ) -> EvaluationDataset:
        """Create a test dataset."""
        return create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="test_dataset_get",
            description="Test dataset for GET",
            original_items_count=5,
            duplication_factor=2,
        )

    def test_get_dataset_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        create_test_dataset: EvaluationDataset,
    ) -> None:
        """Test successfully getting a dataset by ID."""
        response = client.get(
            f"/api/v1/evaluations/datasets/{create_test_dataset.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["dataset_id"] == create_test_dataset.id
        assert data["dataset_name"] == "test_dataset_get"
        assert data["original_items"] == 5
        assert data["total_items"] == 10
        assert data["duplication_factor"] == 2
        assert data["langfuse_dataset_id"].startswith("langfuse")
        assert data["object_store_url"].startswith("s3://test/")

    def test_get_dataset_not_found(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        """Test getting non-existent dataset returns 404."""
        response = client.get(
            "/api/v1/evaluations/datasets/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("error", str(response_data))
        )
        assert "not found" in error_str.lower() or "not accessible" in error_str.lower()

    def test_default_no_signed_url(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test that signed_url is not included by default."""
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        response = client.get(
            f"/api/v1/evaluations/datasets/{dataset.id}",
            headers={"X-API-KEY": user_api_key.key},
        )
        assert response.status_code == 200
        assert response.json()["data"].get("signed_url") is None

    def test_include_signed_url(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test that signed_url is returned when requested."""
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        with patch("app.api.routes.evaluations.dataset.get_cloud_storage") as mock:
            mock.return_value.get_signed_url.return_value = "https://signed.url"
            response = client.get(
                f"/api/v1/evaluations/datasets/{dataset.id}",
                headers={"X-API-KEY": user_api_key.key},
                params={"include_signed_url": True},
            )
        assert response.json()["data"]["signed_url"] == "https://signed.url"

    def test_no_object_store_url_skips_signing(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test that signing is skipped when dataset has no object_store_url."""
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        dataset.object_store_url = None
        db.add(dataset)
        db.commit()

        with patch("app.api.routes.evaluations.dataset.get_cloud_storage") as mock:
            response = client.get(
                f"/api/v1/evaluations/datasets/{dataset.id}",
                headers={"X-API-KEY": user_api_key.key},
                params={"include_signed_url": True},
            )
            mock.assert_not_called()
        assert response.json()["data"].get("signed_url") is None


class TestDeleteDataset:
    """Test DELETE /evaluations/datasets/{dataset_id} endpoint."""

    @pytest.fixture
    def create_test_dataset(
        self, db: Session, user_api_key: TestAuthContext
    ) -> EvaluationDataset:
        """Create a test dataset for deletion."""
        return create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="test_dataset_delete",
            description="Test dataset for deletion",
            original_items_count=3,
            duplication_factor=1,
        )

    def test_delete_dataset_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        create_test_dataset: EvaluationDataset,
        db: Session,
    ) -> None:
        """Test successfully deleting a dataset."""
        dataset_id = create_test_dataset.id

        response = client.delete(
            f"/api/v1/evaluations/datasets/{dataset_id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert data["dataset_id"] == dataset_id
        assert "message" in data

        verify_response = client.get(
            f"/api/v1/evaluations/datasets/{dataset_id}",
            headers=user_api_key_header,
        )
        assert verify_response.status_code == 404

    def test_delete_dataset_not_found(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        """Test deleting non-existent dataset returns 404."""
        response = client.delete(
            "/api/v1/evaluations/datasets/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404
        response_data = response.json()
        error_str = response_data.get(
            "detail", response_data.get("error", str(response_data))
        )
        assert "not found" in error_str.lower()
