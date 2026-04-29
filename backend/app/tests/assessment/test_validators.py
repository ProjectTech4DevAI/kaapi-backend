"""Tests for assessment/validators.py."""

import io

import pytest
from fastapi import UploadFile

from app.assessment.validators import MAX_FILE_SIZE, validate_dataset_file


def _make_upload(
    filename: str,
    content: bytes,
    content_type: str = "text/csv",
) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers={"content-type": content_type},
    )


class TestValidateDatasetFile:
    @pytest.mark.asyncio
    async def test_valid_csv_accepted(self) -> None:
        file = _make_upload("data.csv", b"col1,col2\nval1,val2")
        content, ext = await validate_dataset_file(file)
        assert ext == ".csv"
        assert content == b"col1,col2\nval1,val2"

    @pytest.mark.asyncio
    async def test_valid_xlsx_accepted(self) -> None:
        file = _make_upload(
            "data.xlsx",
            b"PK\x03\x04fake_xlsx_content",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        content, ext = await validate_dataset_file(file)
        assert ext == ".xlsx"

    @pytest.mark.asyncio
    async def test_xls_rejected_with_clear_error(self) -> None:
        file = _make_upload(
            "data.xls",
            b"fake_xls",
            content_type="application/vnd.ms-excel",
        )
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await validate_dataset_file(file)
        assert exc_info.value.status_code == 422
        assert "Legacy Excel format (.xls) is not supported" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_missing_filename_raises_422(self) -> None:
        from fastapi import HTTPException

        file = _make_upload("", b"data")
        file.filename = None  # type: ignore[assignment]
        with pytest.raises(HTTPException) as exc_info:
            await validate_dataset_file(file)
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_extension_raises_422(self) -> None:
        from fastapi import HTTPException

        file = _make_upload("data.txt", b"some data", content_type="text/plain")
        with pytest.raises(HTTPException) as exc_info:
            await validate_dataset_file(file)
        assert exc_info.value.status_code == 422
        assert "Invalid file type" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_empty_file_raises_422(self) -> None:
        from fastapi import HTTPException

        file = _make_upload("data.csv", b"")
        with pytest.raises(HTTPException) as exc_info:
            await validate_dataset_file(file)
        assert exc_info.value.status_code == 422
        assert "Empty" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_file_too_large_raises_413(self) -> None:
        from fastapi import HTTPException

        oversized = b"x" * (MAX_FILE_SIZE + 1)
        file = _make_upload("data.csv", oversized)
        with pytest.raises(HTTPException) as exc_info:
            await validate_dataset_file(file)
        assert exc_info.value.status_code == 413
        assert "too large" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_unexpected_content_type_still_accepted_by_extension(self) -> None:
        # Unknown MIME type but valid extension — should proceed with a warning log
        file = _make_upload(
            "data.csv", b"a,b\n1,2", content_type="application/octet-stream"
        )
        content, ext = await validate_dataset_file(file)
        assert ext == ".csv"
