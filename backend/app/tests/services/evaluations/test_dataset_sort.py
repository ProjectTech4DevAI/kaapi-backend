"""Test the id-sort branch of `upload_dataset`.

Verifies that when a CSV provides an `id` column, the upload service:
  1. Sorts the parsed items by integer external_id before any upload.
  2. Rewrites the raw CSV bytes to S3 in sorted order (so the
     signed-url view matches every other downstream view).
  3. Calls the Langfuse upload with items in sorted order (so dataset
     items + traces both come back in id order).

The 3 lines we care about (`if had_id_column:` body) are otherwise
uncovered because every existing test of `upload_dataset` uses a
2-column CSV.
"""

from unittest.mock import MagicMock, patch


def test_upload_dataset_sorts_when_id_column_present():
    """End-to-end: out-of-order CSV → S3 stores sorted bytes, Langfuse
    receives items in id order, internal items list ends up sorted."""
    from app.services.evaluations.dataset import upload_dataset

    csv_bytes_uploaded_to_s3: dict[str, bytes] = {}
    items_seen_by_langfuse: dict[str, list] = {}

    def fake_upload_csv_to_object_store(*, storage, csv_content, dataset_name):
        csv_bytes_uploaded_to_s3["bytes"] = csv_content
        return f"s3://bucket/{dataset_name}.csv"

    def fake_upload_dataset_to_langfuse(
        *, langfuse, items, dataset_name, duplication_factor
    ):
        items_seen_by_langfuse["items"] = items
        return ("lf_dataset_id_abc", len(items))

    fake_dataset = MagicMock()
    fake_dataset.id = 42

    with patch(
        "app.services.evaluations.dataset.upload_csv_to_object_store",
        side_effect=fake_upload_csv_to_object_store,
    ), patch(
        "app.services.evaluations.dataset.upload_dataset_to_langfuse",
        side_effect=fake_upload_dataset_to_langfuse,
    ), patch(
        "app.services.evaluations.dataset.create_evaluation_dataset",
        return_value=fake_dataset,
    ), patch(
        "app.services.evaluations.dataset.get_cloud_storage",
        return_value=MagicMock(),
    ), patch(
        "app.services.evaluations.dataset.get_langfuse_client",
        return_value=MagicMock(),
    ):
        # CSV with ids 3, 1, 2 — intentionally out of order
        raw_csv = (
            b"id,question,answer\n"
            b"3,Q_three,A_three\n"
            b"1,Q_one,A_one\n"
            b"2,Q_two,A_two\n"
        )
        upload_dataset(
            session=MagicMock(),
            csv_content=raw_csv,
            dataset_name="test_ds",
            description=None,
            duplication_factor=1,
            organization_id=1,
            project_id=1,
        )

    # 1. The bytes written to S3 should now have id=1 first.
    stored_csv = csv_bytes_uploaded_to_s3["bytes"].decode()
    lines = stored_csv.strip().split("\n")
    assert lines[0] == "id,category,question,answer"  # header
    assert lines[1].startswith("1,")  # id=1 first
    assert lines[2].startswith("2,")  # id=2 second
    assert lines[3].startswith("3,")  # id=3 third

    # 2. Langfuse should receive items in id-sorted order.
    items = items_seen_by_langfuse["items"]
    assert [i["external_id"] for i in items] == ["1", "2", "3"]
    assert [i["question"] for i in items] == ["Q_one", "Q_two", "Q_three"]


def test_upload_dataset_no_sort_when_id_column_absent():
    """Legacy 2-column CSV: original bytes preserved, items in CSV order."""
    from app.services.evaluations.dataset import upload_dataset

    csv_bytes_uploaded_to_s3: dict[str, bytes] = {}
    items_seen_by_langfuse: dict[str, list] = {}

    def fake_upload_csv_to_object_store(*, storage, csv_content, dataset_name):
        csv_bytes_uploaded_to_s3["bytes"] = csv_content
        return f"s3://bucket/{dataset_name}.csv"

    def fake_upload_dataset_to_langfuse(
        *, langfuse, items, dataset_name, duplication_factor
    ):
        items_seen_by_langfuse["items"] = items
        return ("lf_id", len(items))

    fake_dataset = MagicMock()
    fake_dataset.id = 100

    raw_csv = b"question,answer\nQa,Aa\nQb,Ab\nQc,Ac\n"

    with patch(
        "app.services.evaluations.dataset.upload_csv_to_object_store",
        side_effect=fake_upload_csv_to_object_store,
    ), patch(
        "app.services.evaluations.dataset.upload_dataset_to_langfuse",
        side_effect=fake_upload_dataset_to_langfuse,
    ), patch(
        "app.services.evaluations.dataset.create_evaluation_dataset",
        return_value=fake_dataset,
    ), patch(
        "app.services.evaluations.dataset.get_cloud_storage",
        return_value=MagicMock(),
    ), patch(
        "app.services.evaluations.dataset.get_langfuse_client",
        return_value=MagicMock(),
    ):
        upload_dataset(
            session=MagicMock(),
            csv_content=raw_csv,
            dataset_name="legacy_ds",
            description=None,
            duplication_factor=1,
            organization_id=1,
            project_id=1,
        )

    # Original CSV bytes preserved verbatim — no rewrite path taken.
    assert csv_bytes_uploaded_to_s3["bytes"] == raw_csv

    # Items come through in CSV row order with no external_id set.
    items = items_seen_by_langfuse["items"]
    assert [i["question"] for i in items] == ["Qa", "Qb", "Qc"]
    assert all(i["external_id"] is None for i in items)
