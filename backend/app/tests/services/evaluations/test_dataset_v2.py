"""Tests for the Langfuse-free dataset upload service (`upload_dataset_v2`).

Covers the v2 upload slice of the three-metric SRD
(docs/srd-three-metric-evaluation-verdict.md, FR-19/FR-20):

- FR-19: creates the `evaluation_dataset` row with `langfuse_dataset_id` null and
  never touches the Langfuse client.
- FR-20: stores only the original rows (no physical duplication) and records the
  run-time-duplication metadata.

Object storage is the external boundary and is mocked; the dataset row lands in
the real (transactional) DB.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.crud.evaluations.dataset import (
    DATASET_META_DUPLICATE_AT_RUNTIME,
    DATASET_META_DUPLICATION_FACTOR,
    DATASET_META_ORIGINAL_ITEMS,
    DATASET_META_TOTAL_ITEMS,
)
from app.models import EvaluationDataset
from app.services.evaluations.dataset import upload_dataset_v2
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.utils import random_lower_string

_DATASET = "app.services.evaluations.dataset"

_CSV = b"question,answer\nQ0?,A0\nQ1?,A1\nQ2?,A2\nQ3?,A3\n"  # 4 original rows


class TestUploadDatasetV2:
    def test_creates_langfuse_free_row_without_calling_langfuse(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """FR-19: row has null langfuse id; no Langfuse client/upload is called."""
        name = f"v2-upload-{random_lower_string()}"
        with (
            patch(f"{_DATASET}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_DATASET}.upload_csv_to_object_store",
                return_value="s3://bucket/datasets/v2.csv",
            ),
            patch(f"{_DATASET}.get_langfuse_client") as mock_langfuse_client,
            patch(f"{_DATASET}.upload_dataset_to_langfuse") as mock_langfuse_upload,
        ):
            dataset = upload_dataset_v2(
                session=db,
                csv_content=_CSV,
                dataset_name=name,
                description=None,
                duplication_factor=5,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )

        mock_langfuse_client.assert_not_called()
        mock_langfuse_upload.assert_not_called()

        persisted = db.get(EvaluationDataset, dataset.id)
        assert persisted is not None
        assert persisted.langfuse_dataset_id is None
        assert persisted.object_store_url == "s3://bucket/datasets/v2.csv"

    def test_stores_original_rows_and_runtime_dup_metadata(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """FR-20: original CSV stored verbatim; metadata records run-time dup."""
        name = f"v2-meta-{random_lower_string()}"
        with (
            patch(f"{_DATASET}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_DATASET}.upload_csv_to_object_store",
                return_value="s3://bucket/datasets/v2.csv",
            ) as mock_upload,
            patch(f"{_DATASET}.get_langfuse_client"),
            patch(f"{_DATASET}.upload_dataset_to_langfuse"),
        ):
            dataset = upload_dataset_v2(
                session=db,
                csv_content=_CSV,
                dataset_name=name,
                description=None,
                duplication_factor=5,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )

        # The bytes handed to S3 are the original rows, not a duplicated payload.
        assert mock_upload.call_args.kwargs["csv_content"] == _CSV

        meta = dataset.dataset_metadata
        assert meta[DATASET_META_DUPLICATE_AT_RUNTIME] is True
        assert meta[DATASET_META_DUPLICATION_FACTOR] == 5
        assert meta[DATASET_META_ORIGINAL_ITEMS] == 4
        assert meta[DATASET_META_TOTAL_ITEMS] == 20  # 4 rows × factor 5

    def test_object_store_returns_no_url_raises_500(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """S3 is the sole source in v2, so a missing url is fatal (no Langfuse fallback)."""
        with (
            patch(f"{_DATASET}.get_cloud_storage", return_value=MagicMock()),
            patch(f"{_DATASET}.upload_csv_to_object_store", return_value=None),
            patch(f"{_DATASET}.get_langfuse_client"),
        ):
            with pytest.raises(HTTPException) as exc:
                upload_dataset_v2(
                    session=db,
                    csv_content=_CSV,
                    dataset_name=f"v2-nourl-{random_lower_string()}",
                    description=None,
                    duplication_factor=2,
                    organization_id=user_api_key.organization_id,
                    project_id=user_api_key.project_id,
                )

        assert exc.value.status_code == 500
