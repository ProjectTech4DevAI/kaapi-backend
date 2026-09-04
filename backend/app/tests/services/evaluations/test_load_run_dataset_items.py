"""Tests for the run-time dataset item loader (`load_run_dataset_items`).

Covers the v2 run-time-duplication slice of the three-metric SRD
(docs/srd-three-metric-evaluation-verdict.md, FR-21/FR-22):

- FR-21: a v2 dataset (null Langfuse id, run-time-duplication marker, factor N)
  expands each original row ×N with unique ids at run time.
- FR-22: a v1 dataset (Langfuse-backed) is read from Langfuse as-is, never
  re-multiplied, and its S3 CSV is not touched.

The S3 download and the Langfuse fetch are the external boundaries and are
mocked; the dataset row and its metadata live in the real (transactional) DB.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.crud.evaluations.dataset import (
    DATASET_META_DUPLICATE_AT_RUNTIME,
    DATASET_META_DUPLICATION_FACTOR,
    DATASET_META_ORIGINAL_ITEMS,
    DATASET_META_TOTAL_ITEMS,
    create_evaluation_dataset,
)
from app.models import EvaluationDataset
from app.services.evaluations.fast import (
    _load_items_from_object_store,
    execute_fast_evaluation_chunk,
    load_run_dataset_items,
    validate_and_start_fast_evaluation,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.utils import random_lower_string

_FAST = "app.services.evaluations.fast"


def _csv_bytes(n_rows: int) -> bytes:
    lines = ["question,answer"]
    for i in range(n_rows):
        lines.append(f"Q{i}?,A{i}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_v2_dataset(
    *,
    db: Session,
    auth: TestAuthContext,
    original_items_count: int,
    duplication_factor: int,
    duplicate_at_runtime: bool = True,
) -> EvaluationDataset:
    """A Langfuse-free dataset: null langfuse id, S3 url, run-time-dup metadata."""
    return create_evaluation_dataset(
        session=db,
        name=f"v2_ds_{random_lower_string()}",
        dataset_metadata={
            DATASET_META_ORIGINAL_ITEMS: original_items_count,
            DATASET_META_TOTAL_ITEMS: original_items_count * duplication_factor,
            DATASET_META_DUPLICATION_FACTOR: duplication_factor,
            DATASET_META_DUPLICATE_AT_RUNTIME: duplicate_at_runtime,
        },
        object_store_url="s3://bucket/datasets/v2.csv",
        langfuse_dataset_id=None,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )


class TestV2RunTimeDuplication:
    def test_expands_each_row_by_duplication_factor(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """FR-21: 8 original rows × factor 5 → 40 items with unique ids."""
        dataset = _make_v2_dataset(
            db=db, auth=user_api_key, original_items_count=8, duplication_factor=5
        )

        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_FAST}.download_csv_from_object_store",
                return_value=_csv_bytes(8),
            ) as mock_download,
        ):
            items = load_run_dataset_items(session=db, dataset=dataset, langfuse=None)

        assert len(items) == 40
        assert len({item["id"] for item in items}) == 40
        mock_download.assert_called_once()

    def test_duplicates_of_a_row_share_question_id(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """FR-21: the N copies of one original row carry the same question_id."""
        dataset = _make_v2_dataset(
            db=db, auth=user_api_key, original_items_count=3, duplication_factor=4
        )

        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_FAST}.download_csv_from_object_store",
                return_value=_csv_bytes(3),
            ),
        ):
            items = load_run_dataset_items(session=db, dataset=dataset, langfuse=None)

        # Group ids by their (shared) question_id; each original row → one group of 4.
        groups: dict[int, set[str]] = {}
        for item in items:
            groups.setdefault(item["metadata"]["question_id"], set()).add(item["id"])

        assert sorted(groups) == [1, 2, 3]
        assert all(len(ids) == 4 for ids in groups.values())

    def test_marker_absent_does_not_multiply(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """A null-langfuse dataset without the run-time-dup marker loads as-is."""
        dataset = _make_v2_dataset(
            db=db,
            auth=user_api_key,
            original_items_count=5,
            duplication_factor=5,
            duplicate_at_runtime=False,
        )

        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_FAST}.download_csv_from_object_store",
                return_value=_csv_bytes(5),
            ),
        ):
            items = load_run_dataset_items(session=db, dataset=dataset, langfuse=None)

        assert len(items) == 5


class TestDuplicationFactorOverride:
    def test_override_replaces_stored_factor_for_runtime_dataset(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Runtime-dup dataset stored ×5, override 2 → 8 rows × 2 = 16, not 40."""
        dataset = _make_v2_dataset(
            db=db, auth=user_api_key, original_items_count=8, duplication_factor=5
        )

        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_FAST}.download_csv_from_object_store",
                return_value=_csv_bytes(8),
            ),
        ):
            items = _load_items_from_object_store(
                session=db, dataset=dataset, duplication_factor=2
            )

        assert len(items) == 16
        assert len({item["id"] for item in items}) == 16
        # Each original row is emitted exactly twice, keyed item_{row}_{dup}.
        assert sorted(item["id"] for item in items) == sorted(
            f"item_{row}_{dup}" for row in range(8) for dup in range(2)
        )

    def test_no_override_uses_stored_factor(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Absent override → the dataset's stored factor (×5) is used unchanged."""
        dataset = _make_v2_dataset(
            db=db, auth=user_api_key, original_items_count=8, duplication_factor=5
        )

        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_FAST}.download_csv_from_object_store",
                return_value=_csv_bytes(8),
            ),
        ):
            items = _load_items_from_object_store(
                session=db, dataset=dataset, duplication_factor=None
            )

        assert len(items) == 40

    def test_non_runtime_dataset_forces_one_despite_override(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Defensive: a non-runtime dataset ignores the override and stays ×1."""
        dataset = _make_v2_dataset(
            db=db,
            auth=user_api_key,
            original_items_count=6,
            duplication_factor=5,
            duplicate_at_runtime=False,
        )

        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_FAST}.download_csv_from_object_store",
                return_value=_csv_bytes(6),
            ),
        ):
            items = _load_items_from_object_store(
                session=db, dataset=dataset, duplication_factor=3
            )

        assert len(items) == 6

    def test_load_run_dataset_items_threads_override_to_object_store(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """The public loader forwards the override into the S3 expansion (×3)."""
        dataset = _make_v2_dataset(
            db=db, auth=user_api_key, original_items_count=4, duplication_factor=5
        )

        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=MagicMock()),
            patch(
                f"{_FAST}.download_csv_from_object_store",
                return_value=_csv_bytes(4),
            ),
        ):
            items = load_run_dataset_items(
                session=db, dataset=dataset, langfuse=None, duplication_factor=3
            )

        assert len(items) == 12

    def test_langfuse_path_ignores_override(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """A v1 (Langfuse-backed) dataset is read as-is; the override never applies."""
        dataset = create_evaluation_dataset(
            session=db,
            name=f"v1_ds_{random_lower_string()}",
            dataset_metadata={
                DATASET_META_ORIGINAL_ITEMS: 3,
                DATASET_META_TOTAL_ITEMS: 15,
                DATASET_META_DUPLICATION_FACTOR: 5,
            },
            object_store_url="s3://bucket/datasets/v1.csv",
            langfuse_dataset_id="langfuse_override",
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        langfuse_items = [{"id": f"lf_{i}"} for i in range(15)]

        with (
            patch(f"{_FAST}.fetch_dataset_items", return_value=langfuse_items),
            patch(f"{_FAST}.download_csv_from_object_store") as mock_download,
        ):
            items = load_run_dataset_items(
                session=db,
                dataset=dataset,
                langfuse=MagicMock(),
                duplication_factor=2,
            )

        assert len(items) == 15
        mock_download.assert_not_called()


class TestChunkReloadUsesRunFactor:
    def test_chunk_reload_passes_run_duplication_factor(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """The chunk re-load must size off the run's persisted override (×7), so its
        slice count matches the fan-out sizing rather than the stale dataset factor."""
        run = MagicMock()
        run.status = "processing"
        run.duplication_factor = 7
        run.organization_id = user_api_key.organization_id
        run.project_id = user_api_key.project_id

        loaded_items = [{"id": f"item_{i}_0"} for i in range(3)]

        with (
            patch(f"{_FAST}.Session"),
            patch(f"{_FAST}._get_fast_run", return_value=run),
            patch(f"{_FAST}.get_dataset_by_id", return_value=MagicMock()),
            patch(
                f"{_FAST}._resolve_config_and_clients",
                return_value=(MagicMock(), MagicMock(), None),
            ),
            patch(
                f"{_FAST}.load_run_dataset_items", return_value=loaded_items
            ) as mock_load,
            patch(f"{_FAST}.run_response_chunk"),
        ):
            execute_fast_evaluation_chunk(eval_run_id=123, chunk_index=0)

        assert mock_load.call_args.kwargs["duplication_factor"] == 7


class TestV1DatasetReadAsIs:
    def test_langfuse_backed_dataset_is_not_re_multiplied(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """FR-22: a v1 dataset is read from Langfuse verbatim; S3 untouched."""
        dataset = create_evaluation_dataset(
            session=db,
            name=f"v1_ds_{random_lower_string()}",
            dataset_metadata={
                DATASET_META_ORIGINAL_ITEMS: 3,
                DATASET_META_TOTAL_ITEMS: 15,
                DATASET_META_DUPLICATION_FACTOR: 5,
            },
            object_store_url="s3://bucket/datasets/v1.csv",
            langfuse_dataset_id="langfuse_abc",
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        # v1 datasets are already physically duplicated in Langfuse: 3 rows × 5 = 15.
        langfuse_items = [{"id": f"lf_{i}"} for i in range(15)]

        with (
            patch(
                f"{_FAST}.fetch_dataset_items", return_value=langfuse_items
            ) as mock_fetch,
            patch(f"{_FAST}.download_csv_from_object_store") as mock_download,
        ):
            items = load_run_dataset_items(
                session=db, dataset=dataset, langfuse=MagicMock()
            )

        assert len(items) == 15
        mock_fetch.assert_called_once()
        mock_download.assert_not_called()


class TestV1RunStillRequiresLangfuseId:
    def test_non_judge_run_on_null_langfuse_dataset_400s(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """The relaxed langfuse-id check must not leak to v1 (is_judge_run=False)."""
        dataset = _make_v2_dataset(
            db=db, auth=user_api_key, original_items_count=3, duplication_factor=1
        )

        with pytest.raises(HTTPException) as exc:
            validate_and_start_fast_evaluation(
                session=db,
                dataset_id=dataset.id,
                run_name=f"v1-null-lf-{random_lower_string()}",
                config_id=uuid4(),
                config_version=1,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                is_judge_run=False,
            )

        assert exc.value.status_code == 400
        assert "langfuse" in exc.value.detail.lower()
