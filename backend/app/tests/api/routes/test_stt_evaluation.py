"""Tests for STT evaluation API routes."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import EvaluationDataset
from app.models.stt_evaluation import STTSample, EvaluationType
from app.tests.utils.auth import TestAuthContext
from app.core.util import now


# Helper functions
def create_test_stt_dataset(
    db: Session,
    organization_id: int,
    project_id: int,
    name: str = "test_stt_dataset",
    description: str | None = None,
    language: str | None = "en",
) -> EvaluationDataset:
    """Create a test STT dataset."""
    dataset = EvaluationDataset(
        name=name,
        description=description,
        type=EvaluationType.STT.value,
        language=language,
        dataset_metadata={"sample_count": 0, "has_ground_truth_count": 0},
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
    object_store_url: str = "s3://test-bucket/audio/test.mp3",
    ground_truth: str | None = None,
) -> STTSample:
    """Create a test STT sample."""
    sample = STTSample(
        object_store_url=object_store_url,
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
    ) -> None:
        """Test creating an STT dataset with samples."""
        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "test_stt_dataset_create",
                "description": "Test STT dataset",
                "language": "en",
                "samples": [
                    {"object_store_url": "s3://bucket/audio1.mp3"},
                    {
                        "object_store_url": "s3://bucket/audio2.mp3",
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
        assert data["language"] == "en"
        assert data["sample_count"] == 2
        assert data["dataset_metadata"]["has_ground_truth_count"] == 1

    def test_create_stt_dataset_minimal(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test creating an STT dataset with minimal fields."""
        response = client.post(
            "/api/v1/evaluations/stt/datasets",
            json={
                "name": "minimal_stt_dataset",
                "samples": [
                    {"object_store_url": "s3://bucket/audio.mp3"},
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
        assert data["language"] is None
        assert data["sample_count"] == 1

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
                    {"object_store_url": "s3://bucket/audio.mp3"},
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
                    {"object_store_url": "s3://bucket/audio.mp3"},
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
                    {"object_store_url": "s3://bucket/audio.mp3"},
                ],
            },
            headers=user_api_key_header,
        )

        assert response.status_code == 400
        response_data = response.json()
        error_str = response_data.get("detail", response_data.get("error", ""))
        assert "already exists" in error_str.lower()


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
        )
        create_test_stt_sample(
            db=db,
            dataset_id=dataset.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        # Update dataset metadata to reflect the sample count
        dataset.dataset_metadata = {"sample_count": 1, "has_ground_truth_count": 0}
        db.add(dataset)
        db.commit()

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
        assert data["sample_count"] == 1  # Count should still be correct


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
        )
        # Create some samples
        for i in range(3):
            create_test_stt_sample(
                db=db,
                dataset_id=dataset.id,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                object_store_url=f"s3://bucket/audio_{i}.mp3",
            )
        return dataset

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
                "providers": ["gemini-2.5-pro"],
            },
            headers=user_api_key_header,
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
                "providers": ["gemini-2.5-pro"],
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
