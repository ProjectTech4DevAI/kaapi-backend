"""Tests for assessment/dataset.py upload and row counting behavior."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from openpyxl.utils.exceptions import InvalidFileException

from app.assessment.dataset import (
    _count_csv_rows,
    _count_excel_rows,
    _count_rows,
    upload_dataset,
)


class TestCountRows:
    def test_legacy_xls_rejected(self) -> None:
        with pytest.raises(ValueError, match="Legacy Excel format"):
            _count_rows(b"legacy-xls-content", ".xls")

    def test_count_excel_rows_invalid_file_re_raises(self) -> None:
        with patch(
            "openpyxl.load_workbook",
            side_effect=InvalidFileException("bad xlsx"),
        ):
            with pytest.raises(InvalidFileException):
                _count_excel_rows(b"bad")

    def test_count_excel_rows_unexpected_error_raises_value_error(self) -> None:
        with patch("openpyxl.load_workbook", side_effect=RuntimeError("boom")):
            with pytest.raises(ValueError, match="Failed to parse XLSX file"):
                _count_excel_rows(b"bad")

    def test_count_csv_rows(self) -> None:
        assert _count_csv_rows(b"a,b\n1,2\n\n3,4\n") == 2

    def test_count_rows_csv_and_xlsx(self) -> None:
        with patch("app.assessment.dataset._count_excel_rows", return_value=5):
            assert _count_rows(b"x", ".xlsx") == 5
        assert _count_rows(b"a,b\n1,2\n", ".csv") == 1


class TestUploadDataset:
    def test_invalid_xlsx_returns_422(self) -> None:
        session = MagicMock()
        with patch(
            "app.assessment.dataset.sanitize_dataset_name", return_value="ds-1"
        ), patch(
            "app.assessment.dataset._count_rows",
            side_effect=InvalidFileException("bad xlsx"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                upload_dataset(
                    session=session,
                    file_content=b"invalid-xlsx",
                    file_ext=".xlsx",
                    dataset_name="ds-1",
                    description=None,
                    organization_id=1,
                    project_id=1,
                )
        assert exc_info.value.status_code == 422
        assert "Invalid XLSX file content" in exc_info.value.detail

    def test_count_rows_value_error_returns_422(self) -> None:
        session = MagicMock()
        with patch(
            "app.assessment.dataset.sanitize_dataset_name", return_value="ds-1"
        ), patch(
            "app.assessment.dataset._count_rows",
            side_effect=ValueError("Legacy Excel format (.xls) is not supported."),
        ):
            with pytest.raises(HTTPException) as exc_info:
                upload_dataset(
                    session=session,
                    file_content=b"bad",
                    file_ext=".xls",
                    dataset_name="ds-1",
                    description=None,
                    organization_id=1,
                    project_id=1,
                )
        assert exc_info.value.status_code == 422
        assert "Legacy Excel format" in exc_info.value.detail

    def test_count_rows_unexpected_error_returns_generic_422(self) -> None:
        session = MagicMock()
        with patch(
            "app.assessment.dataset.sanitize_dataset_name", return_value="ds-1"
        ), patch(
            "app.assessment.dataset._count_rows",
            side_effect=RuntimeError("unexpected"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                upload_dataset(
                    session=session,
                    file_content=b"bad",
                    file_ext=".xlsx",
                    dataset_name="ds-1",
                    description=None,
                    organization_id=1,
                    project_id=1,
                )
        assert exc_info.value.status_code == 422
        assert "Unable to parse dataset file" in exc_info.value.detail

    def test_upload_dataset_success(self) -> None:
        session = MagicMock()
        created = MagicMock()
        created.id = 9
        with patch(
            "app.assessment.dataset.sanitize_dataset_name", return_value="ds-1"
        ), patch("app.assessment.dataset._count_rows", return_value=2), patch(
            "app.assessment.dataset._upload_file_to_object_store",
            return_value="s3://datasets/file.csv",
        ), patch(
            "app.assessment.dataset.create_evaluation_dataset",
            return_value=created,
        ) as create_ds:
            result = upload_dataset(
                session=session,
                file_content=b"a,b\n1,2\n",
                file_ext=".csv",
                dataset_name="ds-1",
                description="desc",
                organization_id=1,
                project_id=1,
            )
        assert result.id == 9
        create_ds.assert_called_once()

    def test_upload_dataset_object_store_failure_returns_500(self) -> None:
        session = MagicMock()
        with patch(
            "app.assessment.dataset.sanitize_dataset_name", return_value="ds-1"
        ), patch("app.assessment.dataset._count_rows", return_value=1), patch(
            "app.assessment.dataset._upload_file_to_object_store",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                upload_dataset(
                    session=session,
                    file_content=b"a,b\n1,2\n",
                    file_ext=".csv",
                    dataset_name="ds-1",
                    description=None,
                    organization_id=1,
                    project_id=1,
                )
        assert exc_info.value.status_code == 500
