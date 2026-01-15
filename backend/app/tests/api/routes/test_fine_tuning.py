import io
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
from moto import mock_aws
from sqlmodel import Session
from fastapi.testclient import TestClient
import boto3

from app.tests.utils.test_data import create_test_fine_tuning_jobs
from app.tests.utils.utils import get_document
from app.models import (
    Fine_Tuning,
    FineTuningStatus,
    ModelEvaluation,
    ModelEvaluationStatus,
)
from app.core.config import settings


def create_file_mock(file_type):
    counter = {"train": 0, "test": 0}

    def _side_effect(file=None, purpose=None):
        if purpose == "fine-tune":
            if "train" in file.name:
                counter["train"] += 1
                return MagicMock(id=f"file_{counter['train']}")
            elif "test" in file.name:
                counter["test"] += 1
                return MagicMock(id=f"file_{counter['test']}")

    return _side_effect


@pytest.mark.usefixtures("client", "db", "user_api_key_header")
class TestCreateFineTuningJobAPI:
    @mock_aws
    def test_finetune_from_csv_multiple_split_ratio(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        # Setup S3 bucket for moto
        s3 = boto3.client("s3", region_name=settings.AWS_DEFAULT_REGION)
        bucket_name = settings.AWS_S3_BUCKET_PREFIX
        if settings.AWS_DEFAULT_REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    "LocationConstraint": settings.AWS_DEFAULT_REGION
                },
            )

        # Create a test CSV file content
        csv_content = "prompt,label\ntest1,label1\ntest2,label2\ntest3,label3"

        # Setup test files for preprocessing
        for path in ["/tmp/train.jsonl", "/tmp/test.jsonl"]:
            with open(path, "w") as f:
                f.write('{"prompt": "test", "completion": "label"}')

        with patch(
            "app.api.routes.fine_tuning.get_cloud_storage"
        ) as mock_get_cloud_storage:
            with patch(
                "app.api.routes.fine_tuning.get_openai_client"
            ) as mock_get_openai_client:
                with patch(
                    "app.api.routes.fine_tuning.process_fine_tuning_job"
                ) as mock_process_job:
                    mock_storage = MagicMock()
                    mock_storage.put.return_value = (
                        f"s3://{settings.AWS_S3_BUCKET_PREFIX}/test.csv"
                    )
                    mock_get_cloud_storage.return_value = mock_storage

                    mock_openai = MagicMock()
                    mock_get_openai_client.return_value = mock_openai

                    csv_file = io.BytesIO(csv_content.encode())
                    response = client.post(
                        "/api/v1/fine_tuning/fine_tune",
                        files={"file": ("test.csv", csv_file, "text/csv")},
                        data={
                            "base_model": "gpt-4",
                            "split_ratio": "0.5,0.7,0.9",
                            "system_prompt": "you are a model able to classify",
                        },
                        headers=user_api_key_header,
                    )

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["success"] is True
        assert json_data["data"]["message"] == "Fine-tuning job(s) started."
        assert json_data["metadata"] is None
        assert "jobs" in json_data["data"]
        assert len(json_data["data"]["jobs"]) == 3

        # Verify that the background task was called for each split ratio
        assert mock_process_job.call_count == 3

        jobs = db.query(Fine_Tuning).all()
        assert len(jobs) == 3

        for job in jobs:
            db.refresh(job)
            assert (
                job.status == "pending"
            )  # Since background processing is mocked, status remains pending
            assert job.split_ratio in [0.5, 0.7, 0.9]


@pytest.mark.usefixtures("client", "db", "user_api_key_header")
@patch("app.api.routes.fine_tuning.get_openai_client")
class TestRetriveFineTuningJobAPI:
    def test_retrieve_fine_tuning_job(
        self,
        mock_get_openai_client: Any,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        jobs, _ = create_test_fine_tuning_jobs(db, [0.3])
        job = jobs[0]
        job.provider_job_id = "ftjob-mock_job_123"
        db.flush()

        mock_openai_job = MagicMock(
            status="succeeded",
            fine_tuned_model="ft:gpt-4:custom-model",
            error=None,
        )

        mock_openai = MagicMock()
        mock_openai.fine_tuning.jobs.retrieve.return_value = mock_openai_job
        mock_get_openai_client.return_value = mock_openai

        response = client.get(
            f"/api/v1/fine_tuning/{job.id}/refresh", headers=user_api_key_header
        )

        assert response.status_code == 200
        json_data = response.json()

        assert json_data["data"]["status"] == "completed"
        assert json_data["data"]["fine_tuned_model"] == "ft:gpt-4:custom-model"
        assert json_data["data"]["id"] == job.id

    def test_retrieve_fine_tuning_job_failed(
        self,
        mock_get_openai_client: Any,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        jobs, _ = create_test_fine_tuning_jobs(db, [0.3])
        job = jobs[0]
        job.provider_job_id = "ftjob-mock_job_123"
        db.flush()

        mock_openai_job = MagicMock(
            status="failed",
            fine_tuned_model=None,
            error=MagicMock(message="Invalid file format for openai fine tuning"),
        )

        mock_openai = MagicMock()
        mock_openai.fine_tuning.jobs.retrieve.return_value = mock_openai_job
        mock_get_openai_client.return_value = mock_openai

        response = client.get(
            f"/api/v1/fine_tuning/{job.id}/refresh", headers=user_api_key_header
        )

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["data"]["status"] == "failed"
        assert (
            json_data["data"]["error_message"]
            == "Invalid file format for openai fine tuning"
        )
        assert json_data["data"]["id"] == job.id


@pytest.mark.usefixtures("client", "db", "user_api_key_header")
class TestFetchJob:
    def test_fetch_jobs_document(
        self, client: TestClient, db: Session, user_api_key_header: dict[str, str]
    ) -> None:
        jobs, _ = create_test_fine_tuning_jobs(db, [0.3, 0.4])
        document = get_document(db, "dalgo_sample.json")

        response = client.get(
            f"/api/v1/fine_tuning/{document.id}", headers=user_api_key_header
        )
        assert response.status_code == 200
        json_data = response.json()

        assert json_data["success"] is True
        assert json_data["metadata"] is None
        assert len(json_data["data"]) == 2

        job_ratios = sorted([job["split_ratio"] for job in json_data["data"]])
        assert job_ratios == sorted([0.3, 0.4])

        for job in json_data["data"]:
            assert job["document_id"] == str(document.id)
            assert job["status"] == "pending"


@pytest.mark.usefixtures("client", "db", "user_api_key_header")
@patch("app.api.routes.fine_tuning.get_openai_client")
@patch("app.api.routes.fine_tuning.get_cloud_storage")
@patch("app.api.routes.fine_tuning.run_model_evaluation")
class TestAutoEvaluationTrigger:
    """Test cases for automatic evaluation triggering when fine-tuning completes."""

    def test_successful_auto_evaluation_trigger(
        self,
        mock_run_model_evaluation: Any,
        mock_get_cloud_storage: Any,
        mock_get_openai_client: Any,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test that evaluation is automatically triggered when job status changes from running to completed."""
        jobs, _ = create_test_fine_tuning_jobs(db, [0.7])
        job = jobs[0]
        job.status = FineTuningStatus.running
        job.provider_job_id = "ftjob-mock_job_123"
        # Add required fields for model evaluation
        job.test_data_s3_object = f"{settings.AWS_S3_BUCKET_PREFIX}/test-data.csv"
        job.system_prompt = "You are a helpful assistant"
        db.add(job)
        db.commit()
        db.refresh(job)

        mock_storage = MagicMock()
        mock_storage.get_signed_url.return_value = (
            "https://test.s3.amazonaws.com/signed-url"
        )
        mock_get_cloud_storage.return_value = mock_storage

        mock_openai_job = MagicMock(
            status="succeeded",
            fine_tuned_model="ft:gpt-4:custom-model:12345",
            error=None,
        )
        mock_openai = MagicMock()
        mock_openai.fine_tuning.jobs.retrieve.return_value = mock_openai_job
        mock_get_openai_client.return_value = mock_openai

        response = client.get(
            f"/api/v1/fine_tuning/{job.id}/refresh", headers=user_api_key_header
        )

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["data"]["status"] == "completed"
        assert json_data["data"]["fine_tuned_model"] == "ft:gpt-4:custom-model:12345"

        mock_run_model_evaluation.assert_called_once()
        call_args = mock_run_model_evaluation.call_args[0]
        eval_id = call_args[0]

        model_eval = (
            db.query(ModelEvaluation).filter(ModelEvaluation.id == eval_id).first()
        )
        assert model_eval is not None
        assert model_eval.fine_tuning_id == job.id
        assert model_eval.status == ModelEvaluationStatus.pending

    def test_skip_evaluation_when_already_exists(
        self,
        mock_run_model_evaluation: Any,
        mock_get_cloud_storage: Any,
        mock_get_openai_client: Any,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test that evaluation is skipped when an active evaluation already exists."""
        jobs, _ = create_test_fine_tuning_jobs(db, [0.7])
        job = jobs[0]
        job.status = FineTuningStatus.running
        job.provider_job_id = "ftjob-mock_job_123"
        job.test_data_s3_object = f"{settings.AWS_S3_BUCKET_PREFIX}/test-data.csv"
        job.system_prompt = "You are a helpful assistant"
        db.add(job)
        db.commit()

        existing_eval = ModelEvaluation(
            fine_tuning_id=job.id,
            status=ModelEvaluationStatus.pending,
            project_id=job.project_id,
            organization_id=job.organization_id,
            document_id=job.document_id,
            fine_tuned_model="ft:gpt-4:test-model:123",
            test_data_s3_object=f"{settings.AWS_S3_BUCKET_PREFIX}/test-data.csv",
            base_model="gpt-4",
            split_ratio=0.7,
            system_prompt="You are a helpful assistant",
        )
        db.add(existing_eval)
        db.commit()

        mock_storage = MagicMock()
        mock_storage.get_signed_url.return_value = (
            "https://test.s3.amazonaws.com/signed-url"
        )
        mock_get_cloud_storage.return_value = mock_storage

        mock_openai_job = MagicMock(
            status="succeeded",
            fine_tuned_model="ft:gpt-4:custom-model:12345",
            error=None,
        )
        mock_openai = MagicMock()
        mock_openai.fine_tuning.jobs.retrieve.return_value = mock_openai_job
        mock_get_openai_client.return_value = mock_openai

        response = client.get(
            f"/api/v1/fine_tuning/{job.id}/refresh", headers=user_api_key_header
        )

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["data"]["status"] == "completed"

        mock_run_model_evaluation.assert_not_called()

        evaluations = (
            db.query(ModelEvaluation)
            .filter(ModelEvaluation.fine_tuning_id == job.id)
            .all()
        )
        assert len(evaluations) == 1
        assert evaluations[0].id == existing_eval.id

    def test_evaluation_not_triggered_for_non_completion_status_changes(
        self,
        mock_run_model_evaluation: Any,
        mock_get_cloud_storage: Any,
        mock_get_openai_client: Any,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test that evaluation is not triggered for status changes other than to completed."""
        # Test Case 1: pending to running
        jobs, _ = create_test_fine_tuning_jobs(db, [0.7])
        job = jobs[0]
        job.status = FineTuningStatus.pending
        job.provider_job_id = "ftjob-mock_job_123"
        db.add(job)
        db.commit()

        # Mock cloud storage
        mock_storage = MagicMock()
        mock_storage.get_signed_url.return_value = (
            "https://test.s3.amazonaws.com/signed-url"
        )
        mock_get_cloud_storage.return_value = mock_storage

        mock_openai_job = MagicMock(
            status="running",
            fine_tuned_model=None,
            error=None,
        )
        mock_openai = MagicMock()
        mock_openai.fine_tuning.jobs.retrieve.return_value = mock_openai_job
        mock_get_openai_client.return_value = mock_openai

        response = client.get(
            f"/api/v1/fine_tuning/{job.id}/refresh", headers=user_api_key_header
        )

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["data"]["status"] == "running"
        mock_run_model_evaluation.assert_not_called()

        job.status = FineTuningStatus.running
        db.add(job)
        db.commit()

        mock_openai_job.status = "failed"
        mock_openai_job.error = MagicMock(message="Training failed")

        response = client.get(
            f"/api/v1/fine_tuning/{job.id}/refresh", headers=user_api_key_header
        )

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["data"]["status"] == "failed"
        mock_run_model_evaluation.assert_not_called()

    def test_evaluation_not_triggered_for_already_completed_jobs(
        self,
        mock_run_model_evaluation: Any,
        mock_get_cloud_storage: Any,
        mock_get_openai_client: Any,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test that evaluation is not triggered when refreshing an already completed job."""
        jobs, _ = create_test_fine_tuning_jobs(db, [0.7])
        job = jobs[0]
        job.status = FineTuningStatus.completed
        job.provider_job_id = "ftjob-mock_job_123"
        job.fine_tuned_model = "ft:gpt-4:custom-model:12345"
        db.add(job)
        db.commit()

        # Mock cloud storage
        mock_storage = MagicMock()
        mock_storage.get_signed_url.return_value = (
            "https://test.s3.amazonaws.com/signed-url"
        )
        mock_get_cloud_storage.return_value = mock_storage

        # Mock OpenAI response (job remains succeeded)
        mock_openai_job = MagicMock(
            status="succeeded",
            fine_tuned_model="ft:gpt-4:custom-model:12345",
            error=None,
        )
        mock_openai = MagicMock()
        mock_openai.fine_tuning.jobs.retrieve.return_value = mock_openai_job
        mock_get_openai_client.return_value = mock_openai

        response = client.get(
            f"/api/v1/fine_tuning/{job.id}/refresh", headers=user_api_key_header
        )

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["data"]["status"] == "completed"

        mock_run_model_evaluation.assert_not_called()

        # Verify no evaluations exist in database for this job
        evaluations = (
            db.query(ModelEvaluation)
            .filter(ModelEvaluation.fine_tuning_id == job.id)
            .all()
        )
        assert len(evaluations) == 0
