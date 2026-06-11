"""Tests for CSV parsing in app.services.evaluations.validators."""

import pytest
from fastapi import HTTPException

from app.crud.evaluations.score import DEFAULT_CATEGORY
from app.services.evaluations.validators import parse_csv_items


class TestParseCsvItemsCategory:
    """Tests that exercise the optional `category` column behaviour."""

    def test_no_category_column_items_omit_category_key(self) -> None:
        """When the CSV has no `category` column, items must not carry the key.

        Datasets uploaded without opting in to category should stay clean of a
        category dimension end-to-end (Langfuse metadata, traces, API response).
        """
        csv = b"question,answer\nq1,a1\nq2,a2\n"
        items = parse_csv_items(csv)
        assert items == [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]
        assert all("category" not in item for item in items)

    def test_category_column_present_fills_blanks_with_default(self) -> None:
        """When the column exists, blank cells default to `Other`, others are title-cased."""
        csv = b"question,answer,Category\nq1,a1,health\nq2,a2,\nq3,a3,EDUCATION\n"
        items = parse_csv_items(csv)
        assert items[0]["category"] == "Health"
        assert items[1]["category"] == DEFAULT_CATEGORY
        assert items[2]["category"] == "Education"

    def test_category_column_header_is_case_insensitive(self) -> None:
        """Header matching is case-insensitive — `CATEGORY` is the same column as `category`."""
        csv = b"Question,Answer,CATEGORY\nq1,a1,Sports\n"
        items = parse_csv_items(csv)
        assert items[0]["category"] == "Sports"

    def test_missing_required_columns_raises_422(self) -> None:
        csv = b"question\nq1\n"
        with pytest.raises(HTTPException) as excinfo:
            parse_csv_items(csv)
        assert excinfo.value.status_code == 422

    def test_unexpected_column_raises_422(self) -> None:
        csv = b"question,answer,unexpected_col\nq1,a1,x\n"
        with pytest.raises(HTTPException) as excinfo:
            parse_csv_items(csv)
        assert excinfo.value.status_code == 422

    def test_empty_csv_raises_422(self) -> None:
        csv = b"question,answer\n"
        with pytest.raises(HTTPException) as excinfo:
            parse_csv_items(csv)
        assert excinfo.value.status_code == 422
