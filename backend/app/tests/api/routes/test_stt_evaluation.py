"""Tests for STT evaluation API routes."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import EvaluationDataset, File, FileType
from app.models.stt_evaluation import STTSample, EvaluationType
from app.crud.language import get_language_by_locale
from app.tests.utils.auth import TestAuthContext
from app.core.util import now


# Helper functions
def create_test_file(
    db: Session,
    organization_id: int,
    project_id: int,
    object_store_url: str = "s3://test-bucket/audio/test.mp3",
    filename: str = "test.mp3",
    size_bytes: int = 1024,
    content_type: str = "audio/mpeg",
    file_type: str = FileType.AUDIO.value,
) -> File:
    """Create a test file record."""
    file = File(
        object_store_url=object_store_url,
        filename=filename,
        size_bytes=size_bytes,
        content_type=content_type,
        file_type=file_type,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def create_test_stt_dataset(
    db: Session,
    organization_id: int,
    project_id: int,
    name: str = "test_stt_dataset",
    description: str | None = None,
    language_id: int | None = None,
    dataset_metadata: dict | None = None,
) -> EvaluationDataset:
    """Create a test STT dataset."""
    dataset = EvaluationDataset(
        name=name,
        description=description,
        type=EvaluationType.STT.value,
        language_id=language_id,
        dataset_metadata=dataset_metadata
        or {"sample_count": 0, "has_ground_truth_count": 0},
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def create_test_stt_sample(
    db: Session,
    dataset_id: int,
    organization_id: int,
    project_id: int,
    file_id: int | None = None,
    ground_truth: str | None = None,
) -> STTSample:
    """Create a test STT sample."""
    # If no file_id provided, create a test file first
    if file_id is None:
        file = create_test_file(
            db=db,
            organization_id=organization_id,
            project_id=project_id,
        )
        file_id = file.id

    sample = STTSample(
        file_id=file_id,
        ground_truth=ground_truth,
        dataset_id=dataset_id,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(sample)
    db.commit()
    db.refresh(sample)
    return sample


class TestSTTDatasetCreate:
    """Test POST /evaluations/stt/datasets endpoint."""

    def test_create_stt_dataset_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test creating an STT dataset with samples."""
        # Get seeded English language
        language = get_language_by_locale(session=db, locale="en")
        file1 = create_test_file(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            object_store_url="s3://bucket/audio1.mp3",
            filename="audio1.mp3",
        )
        file2 = create_test_file(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            object_store_url="s3://bucket/audio2.mp3",
            filename="audio2.mp3",
        )

        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "test_stt_dataset_create",
                "description": "Test STT dataset",
                "language_id": language.id,
                "samples": [
                    {"file_id": file1.id},
                    {
                        "file_id": file2.id,
                        "ground_truth": "Hello world",
                    },
                ],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["name"] == "test_stt_dataset_create"
        assert data["description"] == "Test STT dataset"
        assert data["type"] == "stt"
        assert data["language_id"] == language.id
        assert data["dataset_metadata"]["sample_count"] == 2
        assert data["dataset_metadata"]["has_ground_truth_count"] == 1

    def test_create_stt_dataset_minimal(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test creating an STT dataset with minimal fields."""
        # Create a test file first
        file = create_test_file(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            object_store_url="s3://bucket/audio.mp3",
            filename="audio.mp3",
        )

        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "minimal_stt_dataset",
                "samples": [
                    {"file_id": file.id},
                ],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 200, response.text
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["name"] == "minimal_stt_dataset"
        assert data["description"] is None
        assert data["language_id"] is None
        assert data["dataset_metadata"]["sample_count"] == 1

    def test_create_stt_dataset_empty_samples(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating an STT dataset with empty samples fails."""
        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "empty_samples_dataset",
                "samples": [],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422

    def test_create_stt_dataset_missing_name(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating an STT dataset without name fails."""
        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "samples": [
                    {"file_id": 1},
                ],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 422

    def test_create_stt_dataset_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test creating an STT dataset without authentication fails."""
        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "unauthenticated_dataset",
                "samples": [
                    {"file_id": 1},
                ],
            },
        )

        assert response.status_code == 401

    def test_create_stt_dataset_duplicate_name(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test creating an STT dataset with duplicate name fails."""
        # Create a test file first
        file = create_test_file(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        # Create first dataset
        create_test_stt_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="duplicate_name_test",
        )

        # Try to create another with same name
        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "duplicate_name_test",
                "samples": [
                    {"file_id": file.id},
                ],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 400
        response_data = response.json()
        error_str = response_data.get("detail", response_data.get("error", ""))
        assert "already exists" in error_str.lower()

    def test_create_stt_dataset_invalid_file_id(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating an STT dataset with invalid file_id fails."""
        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "invalid_file_dataset",
                "samples": [
                    {"file_id": 99999},
                ],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 400
        response_data = response.json()
        error_str = response_data.get("detail", response_data.get("error", ""))
        assert "not found" in error_str.lower()


class TestSTTDatasetList:
    """Test GET /evaluations/stt/datasets endpoint."""

    def test_list_stt_datasets_empty(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing STT datasets when none exist."""
        response = client.get(
            "/api/v1/evaluations/stt/datasets",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert isinstance(response_data["data"], list)

    def test_list_stt_datasets_with_data(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test listing STT datasets with data."""
        # Create test datasets
        dataset1 = create_test_stt_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="list_test_dataset_1",
        )
        create_test_stt_sample(
            db=db,
            dataset_id=dataset1.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        dataset2 = create_test_stt_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="list_test_dataset_2",
        )
        create_test_stt_sample(
            db=db,
            dataset_id=dataset2.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        response = client.get(
            "/api/v1/evaluations/stt/datasets",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]
        assert len(data) >= 2

        # Check that our datasets are in the list
        names = [d["name"] for d in data]
        assert "list_test_dataset_1" in names
        assert "list_test_dataset_2" in names

    def test_list_stt_datasets_pagination(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test pagination for listing STT datasets."""
        # Create multiple datasets
        for i in range(5):
            create_test_stt_dataset(
                db=db,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                name=f"pagination_test_dataset_{i}",
            )

        # Test with limit
        response = client.get(
            "/api/v1/evaluations/stt/datasets",
            params={"limit": 2, "offset": 0},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["data"]) == 2
        assert response_data["metadata"]["limit"] == 2
        assert response_data["metadata"]["offset"] == 0


class TestSTTDatasetGet:
    """Test GET /evaluations/stt/datasets/{dataset_id} endpoint."""

    def test_get_stt_dataset_success(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test getting an STT dataset by ID."""
        dataset = create_test_stt_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="get_test_dataset",
            description="Test description",
        )
        sample = create_test_stt_sample(
            db=db,
            dataset_id=dataset.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            ground_truth="Test transcription",
        )

        response = client.get(
            f"/api/v1/evaluations/stt/datasets/{dataset.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        data = response_data["data"]

        assert data["id"] == dataset.id
        assert data["name"] == "get_test_dataset"
        assert data["description"] == "Test description"
        assert data["type"] == "stt"
        assert len(data["samples"]) == 1
        assert data["samples"][0]["id"] == sample.id
        assert data["samples"][0]["ground_truth"] == "Test transcription"

    def test_get_stt_dataset_not_found(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a non-existent STT dataset."""
        response = client.get(
            "/api/v1/evaluations/stt/datasets/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404

    def test_get_stt_dataset_without_samples(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ) -> None:
        """Test getting an STT dataset without including samples."""
        # Create dataset with sample_count in metadata set correctly
        dataset = create_test_stt_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="get_no_samples_dataset",
            dataset_metadata={"sample_count": 1, "has_ground_truth_count": 0},
        )
        create_test_stt_sample(
            db=db,
            dataset_id=dataset.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        response = client.get(
            f"/api/v1/evaluations/stt/datasets/{dataset.id}",
            params={"include_samples": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        data = response_data["data"]

        assert data["id"] == dataset.id
        assert data["samples"] == []
        assert data["dataset_metadata"]["sample_count"] == 1


class TestSTTEvaluationRun:
    """Test STT evaluation run endpoints."""

    @pytest.fixture
    def test_dataset_with_samples(
        self, db: Session, user_api_key: TestAuthContext
    ) -> EvaluationDataset:
        """Create a test dataset with samples for evaluation."""
        dataset = create_test_stt_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            name="eval_test_dataset",
            dataset_metadata={"sample_count": 3, "has_ground_truth_count": 0},
        )
        # Create some samples (file will be created automatically)
        for i in range(3):
            file = create_test_file(
                db=db,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                object_store_url=f"s3://bucket/audio_{i}.mp3",
                filename=f"audio_{i}.mp3",
            )
            create_test_stt_sample(
                db=db,
                dataset_id=dataset.id,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                file_id=file.id,
            )
        return dataset

    @patch("app.api.routes.stt_evaluations.evaluation.start_low_priority_job")
    def test_start_stt_evaluation_success(
        self,
        mock_start_job,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test successfully starting an STT evaluation run."""
        dataset = test_dataset_with_samples
        mock_start_job.return_value = "mock-celery-task-id"

        response = client.post(
            "/api/v1/evaluations/stt/runs",
            json={
                "run_name": "success_test_run",
                "dataset_id": dataset.id,
                "models": ["gemini-2.5-pro"],
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
        assert data["type"] == "stt"
        assert data["models"] == ["gemini-2.5-pro"]
        assert data["total_items"] == 3  # 3 samples × 1 model
        assert data["status"] == "pending"
        assert data["organization_id"] == user_api_key.organization_id
        assert data["project_id"] == user_api_key.project_id
        assert data["error_message"] is None

        mock_start_job.assert_called_once()

    def test_start_stt_evaluation_invalid_dataset(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test starting an STT evaluation with invalid dataset ID."""
        response = client.post(
            "/api/v1/evaluations/stt/runs",
            json={
                "run_name": "test_run",
                "dataset_id": 99999,
                "models": ["gemini-2.5-pro"],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 404
        response_data = response.json()
        error_str = response_data.get("detail", response_data.get("error", ""))
        assert "not found" in error_str.lower()

    def test_start_stt_evaluation_cross_org_access(
        self,
        client: TestClient,
        superuser_api_key_header: dict[str, str],
        test_dataset_with_samples: EvaluationDataset,
    ) -> None:
        """Test that a user from another organization cannot start a run on a dataset they don't own."""
        dataset = test_dataset_with_samples  # belongs to user_api_key (Dalgo org)

        response = client.post(
            "/api/v1/evaluations/stt/runs",
            json={
                "run_name": "cross_org_run",
                "dataset_id": dataset.id,
                "models": ["gemini-2.5-pro"],
            },
            headers=superuser_api_key_header,  # Glific org
        )

        assert response.status_code == 404
        response_data = response.json()
        error_str = response_data.get("detail", response_data.get("error", ""))
        assert "not found" in error_str.lower()

    def test_start_stt_evaluation_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test starting an STT evaluation without authentication."""
        response = client.post(
            "/api/v1/evaluations/stt/runs",
            json={
                "run_name": "test_run",
                "dataset_id": 1,
                "models": ["gemini-2.5-pro"],
            },
        )

        assert response.status_code == 401

    def test_list_stt_runs_empty(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing STT runs when none exist."""
        response = client.get(
            "/api/v1/evaluations/stt/runs",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert isinstance(response_data["data"], list)

    def test_get_stt_run_not_found(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a non-existent STT run."""
        response = client.get(
            "/api/v1/evaluations/stt/runs/99999",
            headers=user_api_key_header,
        )

        assert response.status_code == 404


class TestSTTResultFeedback:
    """Test STT result feedback endpoint."""

    def test_update_feedback_not_found(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test updating feedback for non-existent result."""
        response = client.patch(
            "/api/v1/evaluations/stt/results/99999",
            json={
                "is_correct": True,
                "comment": "Test comment",
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 404

    def test_update_feedback_without_authentication(
        self,
        client: TestClient,
    ) -> None:
        """Test updating feedback without authentication."""
        response = client.patch(
            "/api/v1/evaluations/stt/results/1",
            json={
                "is_correct": True,
            },
        )

        assert response.status_code == 401
