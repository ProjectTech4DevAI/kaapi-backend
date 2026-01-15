from unittest.mock import MagicMock

import pytest
from sqlmodel import Session, select

from app.core.cloud.storage import CloudStorageError
from app.crud.evaluations.dataset import (
    create_evaluation_dataset,
    download_csv_from_object_store,
    get_dataset_by_id,
    get_dataset_by_name,
    list_datasets,
    update_dataset_langfuse_id,
    upload_csv_to_object_store,
)
from app.models import Organization, Project
from app.core.util import now
from app.models import EvaluationRun
from app.crud.evaluations.dataset import delete_dataset


class TestCreateEvaluationDataset:
    """Test creating evaluation datasets."""

    def test_create_evaluation_dataset_minimal(self, db: Session) -> None:
        """Test creating a dataset with minimal required fields."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset",
            dataset_metadata={"original_items_count": 10, "total_items_count": 50},
            organization_id=org.id,
            project_id=project.id,
        )

        assert dataset.id is not None
        assert dataset.name == "test_dataset"
        assert dataset.dataset_metadata["original_items_count"] == 10
        assert dataset.dataset_metadata["total_items_count"] == 50
        assert dataset.organization_id == org.id
        assert dataset.project_id == project.id
        assert dataset.description is None
        assert dataset.object_store_url is None
        assert dataset.langfuse_dataset_id is None

    def test_create_evaluation_dataset_complete(self, db: Session) -> None:
        """Test creating a dataset with all fields."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="complete_dataset",
            description="A complete test dataset",
            dataset_metadata={
                "original_items_count": 5,
                "total_items_count": 25,
                "duplication_factor": 5,
            },
            object_store_url="s3://bucket/datasets/complete_dataset.csv",
            langfuse_dataset_id="langfuse_123",
            organization_id=org.id,
            project_id=project.id,
        )

        assert dataset.id is not None
        assert dataset.name == "complete_dataset"
        assert dataset.description == "A complete test dataset"
        assert dataset.dataset_metadata["duplication_factor"] == 5
        assert dataset.object_store_url == "s3://bucket/datasets/complete_dataset.csv"
        assert dataset.langfuse_dataset_id == "langfuse_123"
        assert dataset.inserted_at is not None
        assert dataset.updated_at is not None


class TestGetDatasetById:
    """Test fetching datasets by ID."""

    def test_get_dataset_by_id_success(self, db: Session) -> None:
        """Test fetching an existing dataset by ID."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        fetched = get_dataset_by_id(
            session=db,
            dataset_id=dataset.id,
            organization_id=org.id,
            project_id=project.id,
        )

        assert fetched is not None
        assert fetched.id == dataset.id
        assert fetched.name == "test_dataset"

    def test_get_dataset_by_id_not_found(self, db: Session) -> None:
        """Test fetching a non-existent dataset."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        fetched = get_dataset_by_id(
            session=db,
            dataset_id=99999,
            organization_id=org.id,
            project_id=project.id,
        )

        assert fetched is None

    def test_get_dataset_by_id_wrong_org(self, db: Session) -> None:
        """Test that datasets from other orgs can't be fetched."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        fetched = get_dataset_by_id(
            session=db,
            dataset_id=dataset.id,
            organization_id=99999,
            project_id=project.id,
        )

        assert fetched is None


class TestGetDatasetByName:
    """Test fetching datasets by name."""

    def test_get_dataset_by_name_success(self, db: Session) -> None:
        """Test fetching an existing dataset by name."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        create_evaluation_dataset(
            session=db,
            name="unique_dataset",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        fetched = get_dataset_by_name(
            session=db,
            name="unique_dataset",
            organization_id=org.id,
            project_id=project.id,
        )

        assert fetched is not None
        assert fetched.name == "unique_dataset"

    def test_get_dataset_by_name_not_found(self, db: Session) -> None:
        """Test fetching a non-existent dataset by name."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        fetched = get_dataset_by_name(
            session=db,
            name="nonexistent_dataset",
            organization_id=org.id,
            project_id=project.id,
        )

        assert fetched is None


class TestListDatasets:
    """Test listing datasets."""

    def test_list_datasets_empty(self, db: Session) -> None:
        """Test listing datasets when none exist."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        datasets = list_datasets(
            session=db, organization_id=org.id, project_id=project.id
        )

        assert len(datasets) == 0

    def test_list_datasets_multiple(self, db: Session) -> None:
        """Test listing multiple datasets."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        for i in range(5):
            create_evaluation_dataset(
                session=db,
                name=f"dataset_{i}",
                dataset_metadata={"original_items_count": i},
                organization_id=org.id,
                project_id=project.id,
            )

        datasets = list_datasets(
            session=db, organization_id=org.id, project_id=project.id
        )

        assert len(datasets) == 5
        assert datasets[0].name == "dataset_4"
        assert datasets[4].name == "dataset_0"

    def test_list_datasets_pagination(self, db: Session) -> None:
        """Test pagination of datasets."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        for i in range(10):
            create_evaluation_dataset(
                session=db,
                name=f"dataset_{i}",
                dataset_metadata={"original_items_count": i},
                organization_id=org.id,
                project_id=project.id,
            )

        page1 = list_datasets(
            session=db, organization_id=org.id, project_id=project.id, limit=5, offset=0
        )

        page2 = list_datasets(
            session=db, organization_id=org.id, project_id=project.id, limit=5, offset=5
        )

        assert len(page1) == 5
        assert len(page2) == 5
        page1_names = [d.name for d in page1]
        page2_names = [d.name for d in page2]
        assert len(set(page1_names) & set(page2_names)) == 0


class TestUploadCsvToObjectStore:
    """Test CSV upload to object store."""

    def test_upload_csv_to_object_store_success(self) -> None:
        """Test successful object store upload."""
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/datasets/test_dataset.csv"

        csv_content = b"question,answer\nWhat is 2+2?,4\n"

        object_store_url = upload_csv_to_object_store(
            storage=mock_storage, csv_content=csv_content, dataset_name="test_dataset"
        )

        assert object_store_url == "s3://bucket/datasets/test_dataset.csv"
        mock_storage.put.assert_called_once()

    def test_upload_csv_to_object_store_cloud_storage_error(self) -> None:
        """Test object store upload with CloudStorageError."""
        mock_storage = MagicMock()
        mock_storage.put.side_effect = CloudStorageError(
            "Object store bucket not found"
        )

        csv_content = b"question,answer\nWhat is 2+2?,4\n"

        object_store_url = upload_csv_to_object_store(
            storage=mock_storage, csv_content=csv_content, dataset_name="test_dataset"
        )

        assert object_store_url is None

    def test_upload_csv_to_object_store_unexpected_error(self) -> None:
        """Test object store upload with unexpected error."""
        mock_storage = MagicMock()
        mock_storage.put.side_effect = Exception("Unexpected error")

        csv_content = b"question,answer\nWhat is 2+2?,4\n"

        object_store_url = upload_csv_to_object_store(
            storage=mock_storage, csv_content=csv_content, dataset_name="test_dataset"
        )

        assert object_store_url is None


class TestDownloadCsvFromObjectStore:
    """Test CSV download from object store."""

    def test_download_csv_from_object_store_success(self) -> None:
        """Test successful object store download."""
        mock_storage = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"question,answer\nWhat is 2+2?,4\n"
        mock_storage.stream.return_value = mock_body

        csv_content = download_csv_from_object_store(
            storage=mock_storage, object_store_url="s3://bucket/datasets/test.csv"
        )

        assert csv_content == b"question,answer\nWhat is 2+2?,4\n"
        mock_storage.stream.assert_called_once_with("s3://bucket/datasets/test.csv")

    def test_download_csv_from_object_store_empty_url(self) -> None:
        """Test download with empty URL."""
        mock_storage = MagicMock()

        with pytest.raises(
            ValueError, match="object_store_url cannot be None or empty"
        ):
            download_csv_from_object_store(storage=mock_storage, object_store_url=None)

    def test_download_csv_from_object_store_error(self) -> None:
        """Test download with storage error."""
        mock_storage = MagicMock()
        mock_storage.stream.side_effect = Exception("Object store download failed")

        with pytest.raises(Exception, match="Object store download failed"):
            download_csv_from_object_store(
                storage=mock_storage, object_store_url="s3://bucket/datasets/test.csv"
            )


class TestUpdateDatasetLangfuseId:
    """Test updating Langfuse ID."""

    def test_update_dataset_langfuse_id(self, db: Session) -> None:
        """Test updating Langfuse dataset ID."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        assert dataset.langfuse_dataset_id is None

        update_dataset_langfuse_id(
            session=db, dataset_id=dataset.id, langfuse_dataset_id="langfuse_123"
        )

        db.refresh(dataset)
        assert dataset.langfuse_dataset_id == "langfuse_123"

    def test_update_dataset_langfuse_id_nonexistent(self, db: Session) -> None:
        """Test updating Langfuse ID for non-existent dataset."""
        update_dataset_langfuse_id(
            session=db, dataset_id=99999, langfuse_dataset_id="langfuse_123"
        )


class TestDeleteDataset:
    """Test deleting evaluation datasets."""

    def test_delete_dataset_success(self, db: Session) -> None:
        """Test successfully deleting a dataset."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="dataset_to_delete",
            dataset_metadata={"original_items_count": 5},
            organization_id=org.id,
            project_id=project.id,
        )
        dataset_id = dataset.id

        # New signature: delete_dataset(session, dataset) returns str | None
        error = delete_dataset(session=db, dataset=dataset)

        assert error is None  # None means success

        # Verify dataset is deleted
        fetched = get_dataset_by_id(
            session=db,
            dataset_id=dataset_id,
            organization_id=org.id,
            project_id=project.id,
        )
        assert fetched is None

    def test_delete_dataset_not_found(self, db: Session) -> None:
        """Test deleting a non-existent dataset - dataset must be fetched first."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        # Try to fetch a non-existent dataset
        dataset = get_dataset_by_id(
            session=db,
            dataset_id=99999,
            organization_id=org.id,
            project_id=project.id,
        )

        # The pattern now is: fetch dataset first, if not found, handle in caller
        assert dataset is None

    def test_delete_dataset_wrong_org(self, db: Session) -> None:
        """Test that dataset cannot be fetched with wrong organization."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="dataset_to_delete",
            dataset_metadata={"original_items_count": 5},
            organization_id=org.id,
            project_id=project.id,
        )

        # Try to fetch with wrong org - should return None
        fetched_wrong_org = get_dataset_by_id(
            session=db,
            dataset_id=dataset.id,
            organization_id=99999,
            project_id=project.id,
        )
        assert fetched_wrong_org is None

        # Original dataset should still exist
        fetched = get_dataset_by_id(
            session=db,
            dataset_id=dataset.id,
            organization_id=org.id,
            project_id=project.id,
        )
        assert fetched is not None

    def test_delete_dataset_with_evaluation_runs(self, db: Session) -> None:
        """Test that dataset cannot be deleted if it has evaluation runs."""

        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="dataset_with_runs",
            dataset_metadata={"original_items_count": 5},
            organization_id=org.id,
            project_id=project.id,
        )

        eval_run = EvaluationRun(
            run_name="test_run",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config={"model": "gpt-4o"},
            status="pending",
            organization_id=org.id,
            project_id=project.id,
            inserted_at=now(),
            updated_at=now(),
        )
        db.add(eval_run)
        db.commit()

        # Attempt to delete - should return an error message
        error = delete_dataset(session=db, dataset=dataset)

        assert error is not None
        assert "cannot delete" in error.lower() or "being used" in error.lower()
        assert "evaluation run" in error.lower()

        # Dataset should still exist
        fetched = get_dataset_by_id(
            session=db,
            dataset_id=dataset.id,
            organization_id=org.id,
            project_id=project.id,
        )
        assert fetched is not None
