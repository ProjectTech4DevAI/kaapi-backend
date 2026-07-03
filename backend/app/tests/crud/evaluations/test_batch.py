from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.crud.evaluations.batch import load_evaluation_dataset_items


def _eval_run() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        dataset_id=7,
        dataset_name="ds",
        organization_id=1,
        project_id=1,
    )


class TestLoadEvaluationDatasetItems:
    def test_langfuse_backed_dataset_delegates_to_langfuse(self) -> None:
        """A Langfuse-backed dataset (langfuse_dataset_id set) reads from Langfuse."""
        langfuse = MagicMock()
        expected = [{"id": "lf_1", "input": {"question": "q"}}]
        dataset = SimpleNamespace(
            id=7,
            langfuse_dataset_id="lf_ds_1",
            object_store_url="s3://bucket/ds.csv",
            dataset_metadata={},
        )

        with (
            patch(
                "app.crud.evaluations.dataset.get_dataset_by_id",
                return_value=dataset,
            ),
            patch(
                "app.crud.evaluations.batch.fetch_dataset_items",
                return_value=expected,
            ) as mock_fetch,
        ):
            items = load_evaluation_dataset_items(
                session=MagicMock(), eval_run=_eval_run(), langfuse=langfuse
            )

        mock_fetch.assert_called_once_with(langfuse=langfuse, dataset_name="ds")
        assert items == expected

    def test_langfuse_backed_but_tracing_off_falls_back_to_object_store(self) -> None:
        """A Langfuse-backed dataset run with tracing off sources from object
        store (cosine-only) rather than failing."""
        csv_bytes = b"question,answer\nq1,a1\n"
        dataset = SimpleNamespace(
            id=7,
            langfuse_dataset_id="lf_ds_1",
            object_store_url="s3://bucket/ds.csv",
            dataset_metadata={},
        )

        with (
            patch(
                "app.crud.evaluations.dataset.get_dataset_by_id",
                return_value=dataset,
            ),
            patch(
                "app.crud.evaluations.dataset.download_csv_from_object_store",
                return_value=csv_bytes,
            ),
            patch("app.core.cloud.get_cloud_storage", return_value=MagicMock()),
        ):
            items = load_evaluation_dataset_items(
                session=MagicMock(), eval_run=_eval_run(), langfuse=None
            )

        assert [i["id"] for i in items] == ["item_0_0"]

    def test_object_store_backed_sources_from_object_store(self) -> None:
        """An object-store-backed dataset (no langfuse_dataset_id) sources items
        from the CSV with deterministic ids and applied duplication — regardless
        of whether a live Langfuse client is present."""
        csv_bytes = b"question,answer\nq1,a1\nq2,a2\n"
        dataset = SimpleNamespace(
            id=7,
            langfuse_dataset_id=None,
            object_store_url="s3://bucket/ds.csv",
            dataset_metadata={"duplication_factor": 2},
        )

        with (
            patch(
                "app.crud.evaluations.dataset.get_dataset_by_id",
                return_value=dataset,
            ),
            patch(
                "app.crud.evaluations.dataset.download_csv_from_object_store",
                return_value=csv_bytes,
            ),
            patch("app.core.cloud.get_cloud_storage", return_value=MagicMock()),
        ):
            # Even with a live client, an object-store-backed dataset ignores it
            # for DATA so ids stay stable.
            items = load_evaluation_dataset_items(
                session=MagicMock(), eval_run=_eval_run(), langfuse=MagicMock()
            )

        assert [i["id"] for i in items] == [
            "item_0_0",
            "item_0_1",
            "item_1_0",
            "item_1_1",
        ]
        assert items[0]["input"] == {"question": "q1"}
        assert items[0]["expected_output"] == {"answer": "a1"}
        assert items[2]["input"] == {"question": "q2"}

    def test_object_store_backed_without_url_raises(self) -> None:
        """No Langfuse and no object-store URL cannot source items."""
        dataset = SimpleNamespace(
            id=7, langfuse_dataset_id=None, object_store_url=None, dataset_metadata={}
        )

        with patch(
            "app.crud.evaluations.dataset.get_dataset_by_id", return_value=dataset
        ):
            try:
                load_evaluation_dataset_items(
                    session=MagicMock(), eval_run=_eval_run(), langfuse=None
                )
                raise AssertionError("expected ValueError")
            except ValueError as e:
                assert "object-store" in str(e)
