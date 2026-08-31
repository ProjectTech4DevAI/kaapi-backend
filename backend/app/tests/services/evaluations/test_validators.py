"""Tests for CSV parsing in app.services.evaluations.validators."""

import codecs

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


class TestParseCsvItemsEncoding:
    """Tests that exercise the encodings real spreadsheet apps emit."""

    HEADER = "question,answer\n"
    ROW = 'q1,"The fee is 10 – 20 and he said “hi”"\n'
    EXPECTED = "The fee is 10 – 20 and he said “hi”"

    def test_utf8_smart_punctuation(self) -> None:
        csv = (self.HEADER + self.ROW).encode("utf-8")
        assert parse_csv_items(csv)[0]["answer"] == self.EXPECTED

    def test_utf8_with_bom_resolves_headers(self) -> None:
        """Excel's "CSV UTF-8" export prefixes a BOM, which used to leave the
        `question` header as `﻿question` and fail the required-column check."""
        csv = codecs.BOM_UTF8 + (self.HEADER + self.ROW).encode("utf-8")
        items = parse_csv_items(csv)
        assert items[0]["question"] == "q1"
        assert items[0]["answer"] == self.EXPECTED

    def test_cp1252_recovers_smart_punctuation(self) -> None:
        """Windows Excel's plain "CSV" export writes cp1252, where the en dash is
        the single byte 0x96 that UTF-8 rejects as an invalid start byte."""
        csv = (self.HEADER + self.ROW).encode("cp1252")
        assert b"\x96" in csv
        assert parse_csv_items(csv)[0]["answer"] == self.EXPECTED

    def test_utf16_with_bom(self) -> None:
        csv = (self.HEADER + self.ROW).encode("utf-16")
        assert parse_csv_items(csv)[0]["answer"] == self.EXPECTED

    def test_utf32_bom_not_misread_as_utf16(self) -> None:
        """The UTF-32-LE BOM starts with the UTF-16-LE BOM, so probe order matters."""
        csv = (self.HEADER + self.ROW).encode("utf-32")
        assert parse_csv_items(csv)[0]["answer"] == self.EXPECTED

    def test_byte_undefined_in_cp1252_falls_through_to_latin1(self) -> None:
        csv = (self.HEADER + "q1,weird \x81 byte\n").encode("latin-1")
        with pytest.raises(UnicodeDecodeError):
            csv.decode("cp1252")
        assert parse_csv_items(csv)[0]["answer"] == "weird \x81 byte"
