"""Tests for the v2 Langfuse-free dataset upload route.

`POST {API_V2_STR}/evaluations/datasets` — the three-metric SRD's Langfuse-free
upload (docs/srd-three-metric-evaluation-verdict.md, FR-19/FR-20):

- FR-19: 200, row created with null langfuse id, CSV stored in S3, Langfuse never
  called.
- FR-20: response + persisted metadata carry the run-time-duplication marker and
  original/total item counts.

Object storage and Langfuse are the external boundaries and are mocked; the
dataset row lands in the real (transactional) DB.
"""

import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.dataset import (
    DATASET_META_DUPLICATE_AT_RUNTIME,
    DATASET_META_DUPLICATION_FACTOR,
    DATASET_META_ORIGINAL_ITEMS,
    DATASET_META_TOTAL_ITEMS,
)
from app.models import EvaluationDataset
from app.tests.utils.utils import random_lower_string

V2_DATASETS = f"{settings.API_V2_STR}/evaluations/datasets"
_DATASET = "app.services.evaluations.dataset"

_CSV_TEXT = "question,answer\nQ0?,A0\nQ1?,A1\nQ2?,A2\n"  # 3 original rows


def _csv_upload() -> tuple[str, io.BytesIO, str]:
    return ("dataset.csv", io.BytesIO(_CSV_TEXT.encode("utf-8")), "text/csv")


class TestUploadDatasetV2Route:
    def test_upload_creates_langfuse_free_dataset(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
    ) -> None:
        """FR-19/FR-20: 200, null langfuse id, S3 stored, run-time-dup metadata."""
        name = f"v2-route-{random_lower_string()}"
        with (
            patch(f"{_DATASET}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_DATASET}.upload_csv_to_object_store",
                return_value="s3://bucket/datasets/v2.csv",
            ) as mock_upload,
            patch(f"{_DATASET}.get_langfuse_client") as mock_langfuse_client,
            patch(f"{_DATASET}.upload_dataset_to_langfuse") as mock_langfuse_upload,
        ):
            resp = client.post(
                V2_DATASETS,
                files={"file": _csv_upload()},
                data={"dataset_name": name, "duplication_factor": 5},
                headers=user_api_key_header,
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["langfuse_dataset_id"] is None
        assert body["original_items"] == 3
        assert body["total_items"] == 15  # 3 rows × factor 5
        assert body["duplication_factor"] == 5
        assert body["object_store_url"] == "s3://bucket/datasets/v2.csv"

        mock_upload.assert_called_once()
        mock_langfuse_client.assert_not_called()
        mock_langfuse_upload.assert_not_called()

        persisted = db.get(EvaluationDataset, body["dataset_id"])
        assert persisted is not None
        assert persisted.langfuse_dataset_id is None
        meta = persisted.dataset_metadata
        assert meta[DATASET_META_DUPLICATE_AT_RUNTIME] is True
        assert meta[DATASET_META_DUPLICATION_FACTOR] == 5
        assert meta[DATASET_META_ORIGINAL_ITEMS] == 3
        assert meta[DATASET_META_TOTAL_ITEMS] == 15

    def test_upload_defaults_duplication_factor_to_one(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
    ) -> None:
        """An omitted duplication_factor defaults to 1 — total equals original."""
        name = f"v2-default-{random_lower_string()}"
        with (
            patch(f"{_DATASET}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_DATASET}.upload_csv_to_object_store",
                return_value="s3://bucket/datasets/v2.csv",
            ),
            patch(f"{_DATASET}.get_langfuse_client"),
            patch(f"{_DATASET}.upload_dataset_to_langfuse"),
        ):
            resp = client.post(
                V2_DATASETS,
                files={"file": _csv_upload()},
                data={"dataset_name": name},
                headers=user_api_key_header,
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["duplication_factor"] == 1
        assert body["original_items"] == 3
        assert body["total_items"] == 3

    def test_object_store_no_url_returns_500(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """FR-19: with no Langfuse fallback, a failed S3 store is a 500."""
        with (
            patch(f"{_DATASET}.get_cloud_storage", return_value=MagicMock()),
            patch(f"{_DATASET}.upload_csv_to_object_store", return_value=None),
            patch(f"{_DATASET}.get_langfuse_client"),
        ):
            resp = client.post(
                V2_DATASETS,
                files={"file": _csv_upload()},
                data={
                    "dataset_name": f"v2-nourl-{random_lower_string()}",
                    "duplication_factor": 2,
                },
                headers=user_api_key_header,
            )

        assert resp.status_code == 500

    def test_duplication_factor_above_max_rejected(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """The route caps duplication_factor at 5 (ge=1, le=5)."""
        resp = client.post(
            V2_DATASETS,
            files={"file": _csv_upload()},
            data={
                "dataset_name": f"v2-toobig-{random_lower_string()}",
                "duplication_factor": 6,
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 422

    def test_duplication_factor_below_min_rejected(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ) -> None:
        """duplication_factor below 1 is rejected before any upload."""
        resp = client.post(
            V2_DATASETS,
            files={"file": _csv_upload()},
            data={
                "dataset_name": f"v2-toosmall-{random_lower_string()}",
                "duplication_factor": 0,
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 422
