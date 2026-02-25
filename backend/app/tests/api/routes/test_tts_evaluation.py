"""Tests for TTS evaluation API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.util import now
from app.crud.language import get_language_by_locale
from app.crud.tts_evaluations import create_tts_run
from app.models import EvaluationDataset
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType
from app.models.tts_evaluation import TTSResult
from app.tests.utils.auth import TestAuthContext


def create_test_tts_dataset(
    db: Session,
    organization_id: int,
    project_id: int,
    name: str = "test_tts_dataset",
    description: str | None = None,
    language_id: int | None = None,
    dataset_metadata: dict | None = None,
    object_store_url: str | None = "s3://test-bucket/tts_datasets/test.csv",
) -> EvaluationDataset:
    """Create a test TTS dataset directly in the database."""
    dataset = EvaluationDataset(
        name=name,
        description=description,
        type=EvaluationType.TTS.value,
        language_id=language_id,
        object_store_url=object_store_url,
        dataset_metadata=dataset_metadata or {"sample_count": 0},
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def create_test_tts_result(
    db: Session,
    evaluation_run_id: int,
    organization_id: int,
    project_id: int,
    sample_text: str = "Hello world",
    provider: str = "gemini-2.5-pro-preview-tts",
    status: str = JobStatus.PENDING.value,
    object_store_url: str | None = None,
    metadata_: dict | None = None,
    is_correct: bool | None = None,
    comment: str | None = None,
) -> TTSResult:
    """Create a test TTS result directly in the database."""
    result = TTSResult(
        sample_text=sample_text,
        object_store_url=object_store_url,
        metadata_=metadata_,
        provider=provider,
        status=status,
        is_correct=is_correct,
        comment=comment,
        evaluation_run_id=evaluation_run_id,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


class TestTTSDatasetCreate:
    """Test POST /evaluations/tts/datasets endpoint."""

    @patch("app.api.routes.tts_evaluations.dataset.upload_tts_dataset")
    def test_create_dataset_success(
        self,
        mock_upload: MagicMock,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test creating a TTS dataset with samples."""
        language = get_language_by_locale(session=db, locale="en")

        # Mock the upload service to return a dataset
        mock_dataset = EvaluationDataset(
            id=1,
            name="test_tts_create",
            description="Test TTS dataset",
            type=EvaluationType.TTS.value,
            language_id=language.id,
            object_store_url="s3://bucket/tts_datasets/test.csv",
            dataset_metadata={"sample_count": 2},
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        mock_upload.return_value = mock_dataset

        response = client.post(
            "/api/v1/evaluations/tts/datasets",
            json={
                "name": "test_tts_create",
                "description": "Test TTS dataset",
                "language_id": language.id,
                "samples": [
                    {"text": "Hello world"},
                    {"text": "How are you today?"},
                ],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["name"] == "test_tts_create"
        assert data["description"] == "Test TTS dataset"
        assert data["type"] == "tts"
        assert data["language_id"] == language.id
        assert data["dataset_metadata"]["sample_count"] == 2

        mock_upload.assert_called_once()

    @patch("app.api.routes.tts_evaluations.dataset.upload_tts_dataset")
    def test_create_dataset_minimal(
        self,
        mock_upload: MagicMock,
        client: TestClient,
        user_api_key_header: dict[str, str],
        user_api_key: TestAuthContext,
    ) -> None:
        """Test creating a TTS dataset with minimal fields."""
        mock_dataset = EvaluationDataset(
            id=1,
            name="minimal_tts",
            description=None,
            type=EvaluationType.TTS.value,
            language_id=None,
            object_store_url=None,
            dataset_metadata={"sample_count": 1},
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        mock_upload.return_value = mock_dataset

        response = client.post(
            "/api/v1/evaluations/tts/datasets",
            json={
                "name": "minimal_tts",
                "samples": [{"text": "Hello"}],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["name"] == "minimal_tts"
        assert data["description"] is None
        assert data["language_id"] is None

    def test_create_dataset_empty_samples(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating a TTS dataset with empty samples fails."""
        response = client.post(
            "/api/v1/evaluations/tts/datasets",
            json={
                "name": "empty_samples_dataset",
                "samples": [],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422

    def test_create_dataset_missing_name(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating a TTS dataset without name fails."""
        response = client.post(
            "/api/v1/evaluations/tts/datasets",
            json={
                "samples": [{"text": "Hello"}],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422

    def test_create_dataset_empty_text(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating a TTS dataset with empty text fails."""
        response = client.post(
            "/api/v1/evaluations/tts/datasets",
            json={
                "name": "empty_text_dataset",
                "samples": [{"text": ""}],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422

    def test_create_dataset_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test creating a TTS dataset without authentication fails."""
        response = client.post(
            "/api/v1/evaluations/tts/datasets",
            json={
                "name": "unauthenticated_dataset",
                "samples": [{"text": "Hello"}],
            },
        )

        assert response.status_code == 401

    def test_create_dataset_invalid_language_id(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating a TTS dataset with invalid language_id fails."""
        response = client.post(
            "/api/v1/evaluations/tts/datasets",
            json={
                "name": "invalid_lang_dataset",
                "language_id": 99999,
                "samples": [{"text": "Hello"}],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 400
        assert "language" in response.json()["error"].lower()


class TestTTSDatasetList:
    """Test GET /evaluations/tts/datasets endpoint."""

    def test_list_datasets_empty(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing TTS datasets when none exist."""
        response = client.get(
            "/api/v1/evaluations/tts/datasets",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert isinstance(response_data["data"], list)

    def test_list_datasets_with_data(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test listing TTS datasets with data."""
        create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="list_test_dataset_1",
        )
        create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="list_test_dataset_2",
        )

        response = client.get(
            "/api/v1/evaluations/tts/datasets",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert len(data) >= 2

        names = [d["name"] for d in data]
        assert "list_test_dataset_1" in names
        assert "list_test_dataset_2" in names

    def test_list_datasets_pagination(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test pagination for listing TTS datasets."""
        for i in range(5):
            create_test_tts_dataset(
                db=db,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                name=f"pagination_test_{i}",
            )

        response = client.get(
            "/api/v1/evaluations/tts/datasets",
            params={"limit": 2, "offset": 0},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["data"]) == 2
        assert response_data["metadata"]["limit"] == 2
        assert response_data["metadata"]["offset"] == 0

    def test_list_datasets_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test listing TTS datasets without authentication fails."""
        response = client.get("/api/v1/evaluations/tts/datasets")
        assert response.status_code == 401


class TestTTSDatasetGet:
    """Test GET /evaluations/tts/datasets/{dataset_id} endpoint."""

    def test_get_dataset_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test getting a TTS dataset by ID."""
        dataset = create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="get_test_dataset",
            description="Test description",
            dataset_metadata={"sample_count": 3},
        )

        response = client.get(
            f"/api/v1/evaluations/tts/datasets/{dataset.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["id"] == dataset.id
        assert data["name"] == "get_test_dataset"
        assert data["description"] == "Test description"
        assert data["type"] == "tts"
        assert data["dataset_metadata"]["sample_count"] == 3
        assert response_data["metadata"]["sample_count"] == 3

    def test_get_dataset_not_found(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a non-existent TTS dataset."""
        response = client.get(
            "/api/v1/evaluations/tts/datasets/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404

    def test_get_dataset_cross_org_access(
        self,
        client: TestClient,
        superuser_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test that a user from another org cannot access a dataset."""
        dataset = create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="cross_org_dataset",
        )

        response = client.get(
            f"/api/v1/evaluations/tts/datasets/{dataset.id}",
            headers=superuser_api_key_header,
        )

        assert response.status_code == 404

    def test_get_dataset_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test getting a TTS dataset without authentication fails."""
        response = client.get("/api/v1/evaluations/tts/datasets/1")
        assert response.status_code == 401


class TestTTSEvaluationRun:
    """Test TTS evaluation run endpoints."""

    @pytest.fixture
    def test_dataset_with_samples(
        self, db: Session, user_api_key: TestAuthContext
    ) -> EvaluationDataset:
        """Create a test dataset with sample_count > 0 for evaluation."""
        return create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="eval_test_dataset",
            dataset_metadata={"sample_count": 3},
        )

    @patch("app.api.routes.tts_evaluations.evaluation.start_low_priority_job")
    def test_start_evaluation_success(
        self,
        mock_start_job: MagicMock,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test successfully starting a TTS evaluation run."""
        dataset = test_dataset_with_samples
        mock_start_job.return_value = "mock-celery-task-id"

        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "success_test_run",
                "dataset_id": dataset.id,
                "models": ["gemini-2.5-pro-preview-tts"],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["run_name"] == "success_test_run"
        assert data["dataset_id"] == dataset.id
        assert data["dataset_name"] == dataset.name
        assert data["type"] == "tts"
        assert data["models"] == ["gemini-2.5-pro-preview-tts"]
        assert data["total_items"] == 3  # 3 samples × 1 model
        assert data["status"] == "pending"
        assert data["organization_id"] == user_api_key.organization_id
        assert data["project_id"] == user_api_key.project_id
        assert data["error_message"] is None

        mock_start_job.assert_called_once()
        call_kwargs = mock_start_job.call_args
        assert call_kwargs.kwargs["function_path"] == (
            "app.services.tts_evaluations.batch_job.execute_batch_submission"
        )
        assert call_kwargs.kwargs["organization_id"] == user_api_key.organization_id
        assert call_kwargs.kwargs["dataset_id"] == dataset.id
        assert call_kwargs.kwargs["models"] == ["gemini-2.5-pro-preview-tts"]

    @patch("app.api.routes.tts_evaluations.evaluation.start_low_priority_job")
    def test_start_evaluation_multiple_models_total_items(
        self,
        mock_start_job: MagicMock,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test that total_items is sample_count * number of models."""
        dataset = create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="multi_model_dataset",
            dataset_metadata={"sample_count": 5},
        )
        mock_start_job.return_value = "mock-celery-task-id"

        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "multi_model_run",
                "dataset_id": dataset.id,
                "models": ["gemini-2.5-pro-preview-tts"],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        data = response.json()["data"]
        # 5 samples × 1 model
        assert data["total_items"] == 5

    @patch("app.api.routes.tts_evaluations.evaluation.start_low_priority_job")
    def test_start_evaluation_celery_failure(
        self,
        mock_start_job: MagicMock,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test that a Celery queue failure marks the run as failed."""
        mock_start_job.side_effect = Exception("RabbitMQ connection refused")

        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "celery_fail_run",
                "dataset_id": test_dataset_with_samples.id,
                "models": ["gemini-2.5-pro-preview-tts"],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 500
        assert "queue" in response.json()["error"].lower()

    def test_start_evaluation_invalid_dataset(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test starting a TTS evaluation with invalid dataset ID."""
        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "invalid_dataset_run",
                "dataset_id": 99999,
                "models": ["gemini-2.5-pro-preview-tts"],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 404
        assert "not found" in response.json()["error"].lower()

    def test_start_evaluation_empty_dataset(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test starting a TTS evaluation with a dataset that has no samples."""
        dataset = create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="empty_eval_dataset",
            dataset_metadata={"sample_count": 0},
        )

        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "empty_dataset_run",
                "dataset_id": dataset.id,
                "models": ["gemini-2.5-pro-preview-tts"],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 400
        assert "no samples" in response.json()["error"].lower()

    def test_start_evaluation_unsupported_model(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test starting a TTS evaluation with an unsupported model."""
        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "unsupported_model_run",
                "dataset_id": 1,
                "models": ["unsupported-model-xyz"],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422

    def test_start_evaluation_empty_models_list(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test starting a TTS evaluation with an empty models list."""
        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "no_models_run",
                "dataset_id": 1,
                "models": [],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422

    def test_start_evaluation_cross_org_access(
        self,
        client: TestClient,
        superuser_api_key_header: dict[str, str],
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test that a user from another org cannot start a run on a dataset they don't own."""
        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "cross_org_run",
                "dataset_id": test_dataset_with_samples.id,
                "models": ["gemini-2.5-pro-preview-tts"],
            },
            headers=superuser_api_key_header,
        )

        assert response.status_code == 404
        assert "not found" in response.json()["error"].lower()

    def test_start_evaluation_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test starting a TTS evaluation without authentication fails."""
        response = client.post(
            "/api/v1/evaluations/tts/runs",
            json={
                "run_name": "test_run",
                "dataset_id": 1,
                "models": ["gemini-2.5-pro-preview-tts"],
            },
        )

        assert response.status_code == 401

    def test_list_runs_empty(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing TTS runs when none exist."""
        response = client.get(
            "/api/v1/evaluations/tts/runs",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert isinstance(response_data["data"], list)

    def test_list_runs_with_data(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test listing TTS runs with data."""
        dataset = test_dataset_with_samples

        create_tts_run(
            session=db,
            run_name="list_run_1",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=3,
        )
        create_tts_run(
            session=db,
            run_name="list_run_2",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=3,
        )

        response = client.get(
            "/api/v1/evaluations/tts/runs",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        names = [r["run_name"] for r in response_data["data"]]
        assert "list_run_1" in names
        assert "list_run_2" in names

    def test_list_runs_filter_by_dataset(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test listing TTS runs filtered by dataset_id."""
        dataset = test_dataset_with_samples

        create_tts_run(
            session=db,
            run_name="filtered_run",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=3,
        )

        response = client.get(
            "/api/v1/evaluations/tts/runs",
            params={"dataset_id": dataset.id},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert all(r["dataset_id"] == dataset.id for r in data)

    def test_list_runs_filter_by_status(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test listing TTS runs filtered by status."""
        dataset = test_dataset_with_samples

        create_tts_run(
            session=db,
            run_name="status_filter_run",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=3,
        )

        response = client.get(
            "/api/v1/evaluations/tts/runs",
            params={"status": "pending"},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert all(r["status"] == "pending" for r in data)

    def test_list_runs_pagination(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test pagination for listing TTS runs."""
        dataset = test_dataset_with_samples
        for i in range(5):
            create_tts_run(
                session=db,
                run_name=f"pagination_run_{i}",
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                org_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                models=["gemini-2.5-pro-preview-tts"],
                total_items=3,
            )

        response = client.get(
            "/api/v1/evaluations/tts/runs",
            params={"limit": 2, "offset": 0},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["data"]) == 2
        assert response_data["metadata"]["limit"] == 2
        assert response_data["metadata"]["offset"] == 0

    def test_list_runs_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test listing TTS runs without authentication fails."""
        response = client.get("/api/v1/evaluations/tts/runs")
        assert response.status_code == 401

    def test_get_run_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test getting a TTS evaluation run by ID."""
        dataset = test_dataset_with_samples

        run = create_tts_run(
            session=db,
            run_name="get_test_run",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=3,
        )

        response = client.get(
            f"/api/v1/evaluations/tts/runs/{run.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["id"] == run.id
        assert data["run_name"] == "get_test_run"
        assert data["type"] == "tts"
        assert data["status"] == "pending"
        assert isinstance(data["results"], list)

    def test_get_run_with_results(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test getting a TTS evaluation run includes results."""
        dataset = test_dataset_with_samples

        run = create_tts_run(
            session=db,
            run_name="run_with_results",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=2,
        )

        create_test_tts_result(
            db=db,
            evaluation_run_id=run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            sample_text="Hello world",
        )
        create_test_tts_result(
            db=db,
            evaluation_run_id=run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            sample_text="Good morning",
        )

        response = client.get(
            f"/api/v1/evaluations/tts/runs/{run.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["results"]) == 2
        assert data["results_total"] == 2
        texts = {r["sample_text"] for r in data["results"]}
        assert "Hello world" in texts
        assert "Good morning" in texts

    def test_get_run_without_results(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test getting a TTS evaluation run excluding results."""
        dataset = test_dataset_with_samples

        run = create_tts_run(
            session=db,
            run_name="no_results_run",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=1,
        )

        create_test_tts_result(
            db=db,
            evaluation_run_id=run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        response = client.get(
            f"/api/v1/evaluations/tts/runs/{run.id}",
            params={"include_results": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["results"] == []
        assert data["results_total"] == 0

    def test_get_run_not_found(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a non-existent TTS run."""
        response = client.get(
            "/api/v1/evaluations/tts/runs/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404

    def test_get_run_cross_org_access(
        self,
        client: TestClient,
        superuser_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test that a user from another org cannot access a run."""
        dataset = test_dataset_with_samples

        run = create_tts_run(
            session=db,
            run_name="cross_org_run",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=3,
        )

        response = client.get(
            f"/api/v1/evaluations/tts/runs/{run.id}",
            headers=superuser_api_key_header,
        )

        assert response.status_code == 404

    def test_get_run_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test getting a TTS run without authentication fails."""
        response = client.get("/api/v1/evaluations/tts/runs/1")
        assert response.status_code == 401


class TestTTSResultGet:
    """Test GET /evaluations/tts/results/{result_id} endpoint."""

    def test_get_result_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test getting a TTS result by ID."""
        dataset = create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="result_get_dataset",
            dataset_metadata={"sample_count": 1},
        )

        run = create_tts_run(
            session=db,
            run_name="result_get_run",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=1,
        )

        result = create_test_tts_result(
            db=db,
            evaluation_run_id=run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            sample_text="Test speech text",
            status=JobStatus.SUCCESS.value,
            object_store_url="s3://bucket/audio/test.wav",
            metadata_={"duration_seconds": 2.5, "size_bytes": 40000},
        )

        response = client.get(
            f"/api/v1/evaluations/tts/results/{result.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["id"] == result.id
        assert data["sample_text"] == "Test speech text"
        assert data["provider"] == "gemini-2.5-pro-preview-tts"
        assert data["status"] == JobStatus.SUCCESS.value
        assert data["object_store_url"] == "s3://bucket/audio/test.wav"
        assert data["duration_seconds"] == 2.5
        assert data["size_bytes"] == 40000

    def test_get_result_not_found(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a non-existent TTS result."""
        response = client.get(
            "/api/v1/evaluations/tts/results/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404

    def test_get_result_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test getting a TTS result without authentication fails."""
        response = client.get("/api/v1/evaluations/tts/results/1")
        assert response.status_code == 401


class TestTTSResultFeedback:
    """Test PATCH /evaluations/tts/results/{result_id} endpoint."""

    @pytest.fixture
    def test_result(self, db: Session, user_api_key: TestAuthContext) -> TTSResult:
        """Create a test result for feedback tests."""
        dataset = create_test_tts_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="feedback_test_dataset",
            dataset_metadata={"sample_count": 1},
        )

        run = create_tts_run(
            session=db,
            run_name="feedback_test_run",
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=["gemini-2.5-pro-preview-tts"],
            total_items=1,
        )

        return create_test_tts_result(
            db=db,
            evaluation_run_id=run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            sample_text="Feedback test text",
            status=JobStatus.SUCCESS.value,
        )

    def test_update_feedback_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        test_result: TTSResult,
    ) -> None:
        """Test updating human feedback on a TTS result."""
        response = client.patch(
            f"/api/v1/evaluations/tts/results/{test_result.id}",
            json={
                "is_correct": True,
                "comment": "Great audio quality",
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["id"] == test_result.id
        assert data["is_correct"] is True
        assert data["comment"] == "Great audio quality"

    def test_update_feedback_is_correct_only(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        test_result: TTSResult,
    ) -> None:
        """Test updating only is_correct without comment."""
        response = client.patch(
            f"/api/v1/evaluations/tts/results/{test_result.id}",
            json={"is_correct": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["is_correct"] is False

    def test_update_feedback_comment_only(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        test_result: TTSResult,
    ) -> None:
        """Test updating only comment without is_correct."""
        response = client.patch(
            f"/api/v1/evaluations/tts/results/{test_result.id}",
            json={"comment": "Sounds robotic"},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["comment"] == "Sounds robotic"

    def test_update_feedback_not_found(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test updating feedback for non-existent result."""
        response = client.patch(
            "/api/v1/evaluations/tts/results/99999",
            json={
                "is_correct": True,
                "comment": "Test comment",
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 404

    def test_update_feedback_cross_org_access(
        self,
        client: TestClient,
        superuser_api_key_header: dict[str, str],
        test_result: TTSResult,
    ) -> None:
        """Test that a user from another org cannot update feedback."""
        response = client.patch(
            f"/api/v1/evaluations/tts/results/{test_result.id}",
            json={"is_correct": True},
            headers=superuser_api_key_header,
        )

        assert response.status_code == 404

    def test_update_feedback_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test updating feedback without authentication fails."""
        response = client.patch(
            "/api/v1/evaluations/tts/results/1",
            json={"is_correct": True},
        )

        assert response.status_code == 401
