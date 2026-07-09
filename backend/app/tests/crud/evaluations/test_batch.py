from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.crud.evaluations.batch import load_evaluation_dataset_items


def _eval_run() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        dataset_id=7,
        dataset_name="ds",
        organization_id=1,
        project_id=1,
    )


def _dataset(**kw) -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        project_id=1,
        langfuse_dataset_id=kw.get("langfuse_dataset_id"),
        object_store_url=kw.get("object_store_url", "s3://bucket/ds.csv"),
        dataset_metadata=kw.get("dataset_metadata", {}),
    )


class TestLoadEvaluationDatasetItems:
    def test_with_client_reads_langfuse(self) -> None:
        """A client present reads items from Langfuse."""
        expected = [{"id": "lf_1", "input": {"question": "q"}}]

        with (
            patch(
                "app.crud.evaluations.batch.get_dataset_by_id",
                return_value=_dataset(langfuse_dataset_id="lf_ds_1"),
            ),
            patch(
                "app.crud.evaluations.batch.fetch_dataset_items",
                return_value=expected,
            ) as mock_fetch,
        ):
            items = load_evaluation_dataset_items(
                session=MagicMock(), eval_run=_eval_run(), langfuse=MagicMock()
            )

        mock_fetch.assert_called_once()
        assert items == expected

    def test_without_client_reads_object_store(self) -> None:
        """No client sources items from the object-store CSV with deterministic
        ids and applied duplication."""
        with (
            patch(
                "app.crud.evaluations.batch.get_dataset_by_id",
                return_value=_dataset(dataset_metadata={"duplication_factor": 2}),
            ),
            patch(
                "app.crud.evaluations.batch.download_csv_from_object_store",
                return_value=b"question,answer\nq1,a1\nq2,a2\n",
            ),
            patch(
                "app.crud.evaluations.batch.get_cloud_storage",
                return_value=MagicMock(),
            ),
        ):
            items = load_evaluation_dataset_items(
                session=MagicMock(), eval_run=_eval_run(), langfuse=None
            )

        assert [i["id"] for i in items] == [
            "item_0_0",
            "item_0_1",
            "item_1_0",
            "item_1_1",
        ]
        assert items[0]["input"] == {"question": "q1"}
        assert items[0]["expected_output"] == {"answer": "a1"}

        # question_id: 1-based int (Langfuse-upload parity), shared across a
        # row's duplicates so the Q.ID column groups numerically.
        assert [i["metadata"]["question_id"] for i in items] == [1, 1, 2, 2]

    def test_object_store_without_url_raises(self) -> None:
        """No client and no object-store URL cannot source items."""
        with patch(
            "app.crud.evaluations.batch.get_dataset_by_id",
            return_value=_dataset(object_store_url=None),
        ):
            with pytest.raises(ValueError, match="object-store"):
                load_evaluation_dataset_items(
                    session=MagicMock(), eval_run=_eval_run(), langfuse=None
                )

    def test_dataset_not_found_raises(self) -> None:
        with patch("app.crud.evaluations.batch.get_dataset_by_id", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                load_evaluation_dataset_items(
                    session=MagicMock(), eval_run=_eval_run(), langfuse=None
                )
