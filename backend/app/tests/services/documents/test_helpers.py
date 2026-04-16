from io import BytesIO

from fastapi import UploadFile

from app.services.documents.helpers import calculate_file_size


def make_upload_file(content: bytes, size: int | None = None) -> UploadFile:
    """Create an UploadFile with the given content and optional pre-set size."""
    return UploadFile(file=BytesIO(content), size=size)


class TestCalculateFileSizeWithSizeAttribute:
    def test_uses_size_attribute_when_set(self) -> None:
        """Uses file.size directly when it is provided."""
        file = make_upload_file(b"irrelevant", size=2048)
        assert calculate_file_size(file) == 2  # 2048 / 1024 = 2.0

    def test_rounds_fractional_kb(self) -> None:
        """Rounds the result when size is not an exact multiple of 1024."""
        file = make_upload_file(b"irrelevant", size=1536)  # 1.5 KB → rounds to 2
        assert calculate_file_size(file) == 2

    def test_rounds_down_fractional_kb(self) -> None:
        """Rounds down when fractional part is below .5."""
        file = make_upload_file(b"irrelevant", size=1300)  # ~1.27 KB → rounds to 1
        assert calculate_file_size(file) == 1

    def test_large_file_size(self) -> None:
        """Correctly converts large sizes."""
        file = make_upload_file(b"irrelevant", size=10 * 1024 * 1024)  # 10 MB
        assert calculate_file_size(file) == 10 * 1024  # 10240 KB


class TestCalculateFileSizeViaSeek:
    def test_falls_back_to_seek_when_size_is_none(self) -> None:
        """Falls back to seek/tell when file.size is None."""
        file = make_upload_file(b"x" * 2048, size=None)
        assert calculate_file_size(file) == 2  # 2048 / 1024 = 2

    def test_falls_back_to_seek_when_size_is_zero(self) -> None:
        """Falls back to seek/tell when file.size is 0 (falsy)."""
        file = make_upload_file(b"x" * 3072, size=0)
        assert calculate_file_size(file) == 3  # 3072 / 1024 = 3

    def test_resets_file_pointer_after_seek(self) -> None:
        """File pointer is back at position 0 after size calculation."""
        file = make_upload_file(b"hello world", size=None)
        calculate_file_size(file)
        assert file.file.tell() == 0

    def test_seek_with_fractional_kb(self) -> None:
        """Rounds correctly when content size is not a multiple of 1024."""
        file = make_upload_file(b"x" * 1600, size=None)  # ~1.56 KB → rounds to 2
        assert calculate_file_size(file) == 2

    def test_empty_file_via_seek(self) -> None:
        """Returns 0 for an empty file when size is None."""
        file = make_upload_file(b"", size=None)
        assert calculate_file_size(file) == 0
