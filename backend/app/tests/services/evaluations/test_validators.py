"""Unit tests for the CSV parser and id-sort helpers.

Covers the optional `id` and `category` column logic in
`app.services.evaluations.validators`:
  - parse_csv_items: header validation + per-row id validation
  - sort_items_by_external_id: sorts when ids present, no-op when absent
  - items_to_csv_bytes: round-trips parsed items back to CSV bytes
"""

import pytest
from fastapi import HTTPException

from app.services.evaluations.validators import (
    DEFAULT_CATEGORY,
    items_to_csv_bytes,
    parse_csv_items,
    sort_items_by_external_id,
)


class TestParseCsvItemsHeaders:
    """Header-shape validation, including case-insensitivity for id/Category."""

    def test_legacy_two_column_csv_works(self):
        items = parse_csv_items(b"question,answer\nQ1,A1\nQ2,A2\n")
        assert len(items) == 2
        assert items[0]["question"] == "Q1"
        assert items[0]["answer"] == "A1"
        assert items[0]["external_id"] is None
        assert items[0]["category"] == DEFAULT_CATEGORY

    def test_id_column_case_insensitive(self):
        for header in (b"ID", b"Id", b"iD", b"id"):
            data = header + b",question,answer\n7,Q,A\n"
            items = parse_csv_items(data)
            assert items[0]["external_id"] == "7"

    def test_category_column_case_insensitive(self):
        items = parse_csv_items(b"question,answer,CaTeGoRy\nQ,A,health\n")
        assert items[0]["category"] == "Health"

    def test_unknown_column_rejected(self):
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(b"question,answer,xyz\nQ,A,foo\n")
        assert exc.value.status_code == 422
        assert "Unexpected" in exc.value.detail

    def test_missing_required_column_rejected(self):
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(b"question\nQ\n")
        assert exc.value.status_code == 422
        assert "Missing" in exc.value.detail

    def test_empty_csv_no_headers(self):
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(b"")
        assert exc.value.status_code == 422

    def test_no_valid_items_rejected(self):
        # Header only, no data rows.
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(b"question,answer\n")
        assert exc.value.status_code == 422
        assert "No valid items" in exc.value.detail


class TestParseCsvItemsIdValidation:
    """Strict integer validation when the `id` column is present."""

    def test_sequential_integer_ids_accepted(self):
        items = parse_csv_items(b"id,question,answer\n1,Q,A\n2,Q,A\n3,Q,A\n")
        assert [i["external_id"] for i in items] == ["1", "2", "3"]

    def test_negative_zero_and_positive_ids_accepted(self):
        items = parse_csv_items(b"id,question,answer\n-5,Q,A\n0,Q,A\n10,Q,A\n")
        assert [i["external_id"] for i in items] == ["-5", "0", "10"]

    def test_leading_zeros_preserved(self):
        items = parse_csv_items(b"id,question,answer\n01,Q,A\n002,Q,A\n")
        # We preserve display, but they parse as int for sort.
        assert items[0]["external_id"] == "01"
        assert items[1]["external_id"] == "002"

    def test_whitespace_stripped(self):
        items = parse_csv_items(b"id,question,answer\n  3  ,Q,A\n")
        assert items[0]["external_id"] == "3"

    @pytest.mark.parametrize(
        "bad_id",
        [
            "1.",  # trailing dot
            "1)",  # paren
            "1...",  # ellipsis
            "1.5",  # decimal
            "abc",  # pure letters
            "1a",  # alphanumeric
        ],
    )
    def test_non_integer_id_rejected(self, bad_id: str):
        body = f"id,question,answer\n{bad_id},Q,A\n".encode()
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(body)
        assert exc.value.status_code == 422
        assert "not a valid integer" in exc.value.detail
        # row number in the message starts at 2 (row 1 is the header).
        assert "Row 2" in exc.value.detail

    def test_blank_id_when_column_present_rejected(self):
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(b"id,question,answer\n,Q,A\n")
        assert exc.value.status_code == 422
        assert "value is missing" in exc.value.detail

    def test_whitespace_only_id_rejected(self):
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(b"id,question,answer\n   ,Q,A\n")
        assert exc.value.status_code == 422
        assert "value is missing" in exc.value.detail

    def test_validation_failure_points_at_correct_row(self):
        # Two valid rows then a broken one — error message should call out row 4.
        body = b"id,question,answer\n1,Q,A\n2,Q,A\nbroken,Q,A\n"
        with pytest.raises(HTTPException) as exc:
            parse_csv_items(body)
        assert "Row 4" in exc.value.detail

    def test_duplicate_id_logged_but_accepted(self, caplog):
        # Duplicates are a warning, not a 422 — sort stability handles them.
        items = parse_csv_items(b"id,question,answer\n1,Q,A\n1,Q,A\n2,Q,A\n")
        assert len(items) == 3
        assert any("Duplicate id" in r.message for r in caplog.records)


class TestParseCsvItemsCategoryHandling:
    """Category defaulting + title-casing rules."""

    def test_category_defaults_to_other_when_blank(self):
        items = parse_csv_items(
            b"question,answer,category\nQ,A,Health\nQ,A,\nQ,A,Education\n"
        )
        assert [i["category"] for i in items] == ["Health", "Other", "Education"]

    def test_category_lowercase_normalized_to_title_case(self):
        items = parse_csv_items(b"question,answer,category\nQ,A,other\nQ,A,OTHER\n")
        # Both forms collapse into the canonical "Other" bucket.
        assert items[0]["category"] == "Other"
        assert items[1]["category"] == "Other"


class TestSortItemsByExternalId:
    """Sort is numeric (not lexicographic) and no-op without ids."""

    def test_sorts_numerically_not_lexically(self):
        items = [
            {"question": "Q10", "answer": "A", "category": "X", "external_id": "10"},
            {"question": "Q1", "answer": "A", "category": "X", "external_id": "1"},
            {"question": "Q2", "answer": "A", "category": "X", "external_id": "2"},
        ]
        sort_items_by_external_id(items)
        assert [i["external_id"] for i in items] == ["1", "2", "10"]

    def test_no_op_when_all_external_ids_are_none(self):
        items = [
            {"question": "Q1", "answer": "A", "category": "X", "external_id": None},
            {"question": "Q2", "answer": "A", "category": "X", "external_id": None},
            {"question": "Q3", "answer": "A", "category": "X", "external_id": None},
        ]
        original_order = [i["question"] for i in items]
        sort_items_by_external_id(items)
        # Stable + no rearrangement.
        assert [i["question"] for i in items] == original_order

    def test_returns_same_list_reference(self):
        items = [
            {"question": "Q", "answer": "A", "category": "X", "external_id": "1"},
        ]
        out = sort_items_by_external_id(items)
        assert out is items


class TestItemsToCsvBytes:
    """Round-trip parsed items back to CSV, with conditional column inclusion."""

    def test_full_csv_with_id_and_category(self):
        items = [
            {
                "question": "Q1",
                "answer": "A1",
                "category": "Health",
                "external_id": "1",
            },
        ]
        out = items_to_csv_bytes(items).decode()
        first_line = out.split("\n")[0]
        assert first_line == "id,category,question,answer"
        assert "1,Health,Q1,A1" in out

    def test_csv_without_id_omits_id_column(self):
        items = [
            {
                "question": "Q",
                "answer": "A",
                "category": "Health",
                "external_id": None,
            },
        ]
        out = items_to_csv_bytes(items).decode()
        first_line = out.split("\n")[0]
        assert first_line == "category,question,answer"
        assert "id" not in first_line

    def test_legacy_two_column_csv_round_trips(self):
        # Items without category key at all → just question,answer.
        items = [{"question": "Q", "answer": "A", "external_id": None}]
        out = items_to_csv_bytes(items).decode()
        first_line = out.split("\n")[0]
        assert first_line == "question,answer"

    def test_empty_items_returns_empty_bytes(self):
        assert items_to_csv_bytes([]) == b""

    def test_sort_then_serialize_round_trip(self):
        """Parse → sort → re-serialize → parse again should yield items in id order."""
        original = b"id,category,question,answer\n2,Health,Q2,A2\n1,Education,Q1,A1\n"
        items = parse_csv_items(original)
        sort_items_by_external_id(items)
        new_bytes = items_to_csv_bytes(items)
        reparsed = parse_csv_items(new_bytes)
        assert [i["external_id"] for i in reparsed] == ["1", "2"]
        assert reparsed[0]["question"] == "Q1"
        assert reparsed[1]["question"] == "Q2"
