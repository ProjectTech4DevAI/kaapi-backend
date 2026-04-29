"""Tests for assessment/dataset.py upload and row counting behavior."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from openpyxl.utils.exceptions import InvalidFileException

from app.assessment.dataset import _count_rows, upload_dataset


class TestCountRows:
    def test_legacy_xls_rejected(self) -> None:
        with pytest.raises(ValueError, match="Legacy Excel format"):
            _count_rows(b"legacy-xls-content", ".xls")


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
