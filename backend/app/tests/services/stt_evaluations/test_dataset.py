"""Test cases for STT dataset management service."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.exception_handlers import HTTPException
from app.models.stt_evaluation import STTSampleCreate
from app.services.stt_evaluations.dataset import (
    _samples_to_csv,
    _upload_samples_to_object_store,
    upload_stt_dataset,
)


class TestSamplesToCSV:
    """Test cases for _samples_to_csv function."""

    def test_single_sample_without_ground_truth(self):
        """Test CSV conversion with single sample without ground truth."""
        samples = [
            STTSampleCreate(file_id=1),
        ]
        result = _samples_to_csv(samples)

        # Decode and verify - handle both \n and \r\n line endings
        csv_str = result.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        lines = csv_str.strip().split("\n")

        assert len(lines) == 2  # Header + 1 sample
        assert lines[0] == "file_id,ground_truth"
        assert lines[1] == "1,"

    def test_single_sample_with_ground_truth(self):
        """Test CSV conversion with single sample with ground truth."""
        samples = [
            STTSampleCreate(
                file_id=1,
                ground_truth="Hello world",
            ),
        ]
        result = _samples_to_csv(samples)

        csv_str = result.decode("utf-8")
        lines = csv_str.strip().split("\n")

        assert len(lines) == 2
        assert "Hello world" in lines[1]

    def test_multiple_samples(self):
        """Test CSV conversion with multiple samples."""
        samples = [
            STTSampleCreate(
                file_id=1,
                ground_truth="First transcription",
            ),
            STTSampleCreate(
                file_id=2,
                ground_truth="Second transcription",
            ),
            STTSampleCreate(
                file_id=3,
            ),
        ]
        result = _samples_to_csv(samples)

        csv_str = result.decode("utf-8")
        lines = csv_str.strip().split("\n")

        assert len(lines) == 4  # Header + 3 samples

    def test_empty_samples_list(self):
        """Test CSV conversion with empty samples list."""
        samples = []
        result = _samples_to_csv(samples)

        csv_str = result.decode("utf-8")
        lines = csv_str.strip().split("\n")

        # Should only have header
        assert len(lines) == 1
        assert lines[0] == "file_id,ground_truth"

    def test_sample_with_unicode(self):
        """Test CSV conversion with unicode characters."""
        samples = [
            STTSampleCreate(
                file_id=1,
                ground_truth="Hello 世界 🌍",
            ),
        ]
        result = _samples_to_csv(samples)

        csv_str = result.decode("utf-8")
        assert "Hello 世界 🌍" in csv_str

    def test_sample_with_comma_in_ground_truth(self):
        """Test CSV conversion handles commas in ground truth."""
        samples = [
            STTSampleCreate(
                file_id=1,
                ground_truth="Hello, world",
            ),
        ]
        result = _samples_to_csv(samples)

        csv_str = result.decode("utf-8")
        # CSV should properly quote fields with commas
        assert '"Hello, world"' in csv_str or "Hello, world" in csv_str

    def test_sample_with_quotes_in_ground_truth(self):
        """Test CSV conversion handles quotes in ground truth."""
        samples = [
            STTSampleCreate(
                file_id=1,
                ground_truth='He said "hello"',
            ),
        ]
        result = _samples_to_csv(samples)

        # Should produce valid CSV bytes
        assert isinstance(result, bytes)


class TestUploadSamplesToObjectStore:
    """Test cases for _upload_samples_to_object_store function."""

    @patch("app.services.stt_evaluations.dataset.get_cloud_storage")
    @patch("app.services.stt_evaluations.dataset.upload_to_object_store")
    def test_successful_upload(self, mock_upload_csv, mock_get_storage):
        """Test successful upload to object store."""
        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage
        mock_upload_csv.return_value = "s3://bucket/stt_datasets/dataset.csv"

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(file_id=1),
        ]

        result = _upload_samples_to_object_store(
            session=mock_session,
            project_id=1,
            dataset_name="test_dataset",
            samples=samples,
        )

        assert result == "s3://bucket/stt_datasets/dataset.csv"
        mock_upload_csv.assert_called_once()

    @patch("app.services.stt_evaluations.dataset.get_cloud_storage")
    @patch("app.services.stt_evaluations.dataset.upload_to_object_store")
    def test_upload_returns_none_on_failure(self, mock_upload_csv, mock_get_storage):
        """Test upload returns None on failure."""
        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage
        mock_upload_csv.return_value = None

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(file_id=1),
        ]

        result = _upload_samples_to_object_store(
            session=mock_session,
            project_id=1,
            dataset_name="test_dataset",
            samples=samples,
        )

        assert result is None

    @patch("app.services.stt_evaluations.dataset.get_cloud_storage")
    def test_upload_handles_exception(self, mock_get_storage):
        """Test upload handles exceptions gracefully."""
        mock_get_storage.side_effect = Exception("Storage connection failed")

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(file_id=1),
        ]

        result = _upload_samples_to_object_store(
            session=mock_session,
            project_id=1,
            dataset_name="test_dataset",
            samples=samples,
        )

        assert result is None


class TestUploadSTTDataset:
    """Test cases for upload_stt_dataset function."""

    @patch("app.services.stt_evaluations.dataset._upload_samples_to_object_store")
    @patch("app.services.stt_evaluations.dataset.create_stt_samples")
    @patch("app.services.stt_evaluations.dataset.create_stt_dataset")
    def test_successful_upload(
        self, mock_create_dataset, mock_create_samples, mock_upload_samples
    ):
        """Test successful dataset upload orchestration."""
        # Setup mocks
        mock_upload_samples.return_value = "s3://bucket/stt_datasets/test.csv"

        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_create_dataset.return_value = mock_dataset

        mock_sample = MagicMock()
        mock_create_samples.return_value = [mock_sample]

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(
                file_id=1,
                ground_truth="Test transcription",
            ),
        ]

        dataset, created_samples = upload_stt_dataset(
            session=mock_session,
            name="test_dataset",
            samples=samples,
            organization_id=1,
            project_id=1,
        )

        assert dataset == mock_dataset
        assert len(created_samples) == 1

        # Verify create_stt_dataset was called with correct metadata
        mock_create_dataset.assert_called_once()
        call_kwargs = mock_create_dataset.call_args.kwargs
        assert call_kwargs["name"] == "test_dataset"
        assert call_kwargs["dataset_metadata"]["sample_count"] == 1
        assert call_kwargs["dataset_metadata"]["has_ground_truth_count"] == 1

        # Verify single commit after both operations
        mock_session.commit.assert_called_once()

    @patch("app.services.stt_evaluations.dataset._upload_samples_to_object_store")
    @patch("app.services.stt_evaluations.dataset.create_stt_samples")
    @patch("app.services.stt_evaluations.dataset.create_stt_dataset")
    def test_upload_with_description_and_language(
        self, mock_create_dataset, mock_create_samples, mock_upload_samples
    ):
        """Test upload with optional description and language."""
        mock_upload_samples.return_value = None
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_create_dataset.return_value = mock_dataset
        mock_create_samples.return_value = []

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(file_id=1),
        ]

        upload_stt_dataset(
            session=mock_session,
            name="test_dataset",
            samples=samples,
            organization_id=1,
            project_id=1,
            description="Test description",
            language_id=1,
        )

        call_kwargs = mock_create_dataset.call_args.kwargs
        assert call_kwargs["description"] == "Test description"
        assert call_kwargs["language_id"] == 1

    @patch("app.services.stt_evaluations.dataset._upload_samples_to_object_store")
    @patch("app.services.stt_evaluations.dataset.create_stt_samples")
    @patch("app.services.stt_evaluations.dataset.create_stt_dataset")
    def test_upload_counts_ground_truth_correctly(
        self, mock_create_dataset, mock_create_samples, mock_upload_samples
    ):
        """Test that ground truth count is calculated correctly."""
        mock_upload_samples.return_value = None
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_create_dataset.return_value = mock_dataset
        mock_create_samples.return_value = []

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(
                file_id=1,
                ground_truth="Has ground truth",
            ),
            STTSampleCreate(
                file_id=2,
            ),
            STTSampleCreate(
                file_id=3,
                ground_truth="Also has ground truth",
            ),
            STTSampleCreate(
                file_id=4,
                ground_truth="",  # Empty string should not count
            ),
        ]

        upload_stt_dataset(
            session=mock_session,
            name="test_dataset",
            samples=samples,
            organization_id=1,
            project_id=1,
        )

        call_kwargs = mock_create_dataset.call_args.kwargs
        assert call_kwargs["dataset_metadata"]["sample_count"] == 4
        assert call_kwargs["dataset_metadata"]["has_ground_truth_count"] == 2

    @patch("app.services.stt_evaluations.dataset._upload_samples_to_object_store")
    @patch("app.services.stt_evaluations.dataset.create_stt_samples")
    @patch("app.services.stt_evaluations.dataset.create_stt_dataset")
    def test_upload_continues_without_object_store_url(
        self, mock_create_dataset, mock_create_samples, mock_upload_samples
    ):
        """Test that upload continues even when object store upload fails."""
        mock_upload_samples.return_value = None  # Simulates failed upload

        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_create_dataset.return_value = mock_dataset
        mock_create_samples.return_value = []

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(file_id=1),
        ]

        dataset, created_samples = upload_stt_dataset(
            session=mock_session,
            name="test_dataset",
            samples=samples,
            organization_id=1,
            project_id=1,
        )

        # Should still create the dataset
        assert dataset is not None
        call_kwargs = mock_create_dataset.call_args.kwargs
        assert call_kwargs["object_store_url"] is None

    @patch("app.services.stt_evaluations.dataset._upload_samples_to_object_store")
    @patch("app.services.stt_evaluations.dataset.create_stt_samples")
    @patch("app.services.stt_evaluations.dataset.create_stt_dataset")
    def test_upload_rolls_back_on_sample_creation_failure(
        self, mock_create_dataset, mock_create_samples, mock_upload_samples
    ):
        """Test that dataset is rolled back if sample creation fails."""
        mock_upload_samples.return_value = "s3://bucket/stt_datasets/test.csv"

        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_create_dataset.return_value = mock_dataset

        mock_create_samples.side_effect = HTTPException(
            status_code=400, detail="File IDs not found: [999]"
        )

        mock_session = MagicMock()
        samples = [
            STTSampleCreate(file_id=999),
        ]

        with pytest.raises(HTTPException) as exc_info:
            upload_stt_dataset(
                session=mock_session,
                name="test_dataset",
                samples=samples,
                organization_id=1,
                project_id=1,
            )

        assert exc_info.value.status_code == 400
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
