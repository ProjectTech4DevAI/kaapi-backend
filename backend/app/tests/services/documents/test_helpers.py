import csv
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from app.services.documents.helpers import (
    SNIFF_HEAD_BYTES,
    SNIFF_TAIL_BYTES,
    DocumentValidationError,
    _check_csv,
    _check_docx,
    _check_json,
    _check_pdf,
    _check_xlsx,
    _decode_utf8,
    _read_edges,
    calculate_file_size,
    validate_document_content,
    validate_upload,
)


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


class TestDocumentValidationErrorClientMessage:
    def test_client_message_hides_internal_reason(self) -> None:
        """The client-facing wording must never echo the internal `reason`, which
        can describe byte offsets / binary sniffing details we don't expose."""
        err = DocumentValidationError("report.pdf", "NUL byte found at offset 12")
        assert "report.pdf" in err.client_message
        assert "NUL byte found at offset 12" not in err.client_message
        assert "corrupted" in err.client_message


class TestDecodeUtf8:
    def test_nul_byte_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _decode_utf8("f.csv", b"abc\x00def")
        assert "NUL byte" in exc.value.reason

    def test_invalid_utf8_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _decode_utf8("f.csv", b"\xff\xfe")
        assert "not valid UTF-8" in exc.value.reason

    def test_valid_utf8_returns_decoded_str(self) -> None:
        assert _decode_utf8("f.csv", b"hello, world") == "hello, world"


class TestCheckPdf:
    def test_missing_eof_marker_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _check_pdf("f.pdf", b"%PDF-1.4", b"no marker", make_upload_file(b""))
        assert "end-of-file marker" in exc.value.reason

    def test_encrypted_pdf_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _check_pdf(
                "f.pdf", b"%PDF-1.4", b"/Encrypt 4 0 R %%EOF", make_upload_file(b"")
            )
        assert "password protected" in exc.value.reason

    def test_valid_pdf_passes(self) -> None:
        _check_pdf("f.pdf", b"%PDF-1.4", b"trailer\n%%EOF", make_upload_file(b""))


class TestCheckOoxml:
    def test_docx_missing_content_types_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _check_docx("f.docx", b"PK\x03\x04junk", b"", make_upload_file(b""))
        assert "no OOXML content-types part" in exc.value.reason

    def test_docx_that_is_really_xlsx_raises(self) -> None:
        sample = b"[Content_Types].xml ... xl/worksheets/sheet1.xml"
        with pytest.raises(DocumentValidationError) as exc:
            _check_docx("f.docx", sample, b"", make_upload_file(b""))
        assert "not the format its extension claims" in exc.value.reason

    def test_valid_docx_passes(self) -> None:
        sample = b"[Content_Types].xml ... word/document.xml"
        _check_docx("f.docx", sample, b"", make_upload_file(b""))

    def test_xlsx_that_is_really_docx_raises(self) -> None:
        sample = b"[Content_Types].xml ... word/document.xml"
        with pytest.raises(DocumentValidationError) as exc:
            _check_xlsx("f.xlsx", sample, b"", make_upload_file(b""))
        assert "not the format its extension claims" in exc.value.reason

    def test_valid_xlsx_passes(self) -> None:
        sample = b"[Content_Types].xml ... xl/worksheets/sheet1.xml"
        _check_xlsx("f.xlsx", sample, b"", make_upload_file(b""))


class TestCheckCsv:
    def test_binary_signature_renamed_to_csv_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _check_csv("f.csv", b"%PDF-1.4 pretending", b"", make_upload_file(b""))
        assert "renamed to a CSV" in exc.value.reason

    def test_nul_byte_raises_via_decode(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _check_csv("f.csv", b"a,b\x00,c\n", b"", make_upload_file(b""))
        assert "NUL byte" in exc.value.reason

    def test_no_rows_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _check_csv("f.csv", b"", b"", make_upload_file(b""))
        assert "no readable CSV rows" in exc.value.reason

    def test_malformed_csv_raises(self) -> None:
        """A single field larger than csv's field-size limit makes the reader
        raise csv.Error, which surfaces as a validation failure."""
        head = b"x" * (csv.field_size_limit() + 1)
        with pytest.raises(DocumentValidationError) as exc:
            _check_csv("f.csv", head, b"", make_upload_file(b""))
        assert "malformed CSV" in exc.value.reason

    def test_inconsistent_column_count_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            _check_csv("f.csv", b"a,b,c\n1,2\n", b"", make_upload_file(b""))
        assert "inconsistent column count at row 2" in exc.value.reason

    def test_clean_csv_passes(self) -> None:
        _check_csv("f.csv", b"a,b,c\n1,2,3\n4,5,6\n", b"", make_upload_file(b""))

    def test_full_head_drops_trailing_partial_line(self) -> None:
        """At exactly SNIFF_HEAD_BYTES the final row is likely cut mid-line, so the
        sniffer drops it instead of flagging a bogus column-count mismatch."""
        head = (b"a,b,c\n" * 700)[:SNIFF_HEAD_BYTES]
        assert len(head) == SNIFF_HEAD_BYTES
        _check_csv("f.csv", head, b"", make_upload_file(b""))


class TestCheckJson:
    def test_body_not_starting_with_bracket_raises(self) -> None:
        content = b"plain text, not json"
        with pytest.raises(DocumentValidationError) as exc:
            _check_json("f.json", content, b"", make_upload_file(content))
        assert "must start with" in exc.value.reason

    def test_small_valid_json_passes(self) -> None:
        content = b'{"a": 1, "b": [2, 3]}'
        _check_json("f.json", content, content, make_upload_file(content))

    def test_small_invalid_json_raises(self) -> None:
        content = b'{"a": 1, oops}'
        with pytest.raises(DocumentValidationError) as exc:
            _check_json("f.json", content, content, make_upload_file(content))
        assert "invalid JSON" in exc.value.reason

    def test_large_json_closed_correctly_passes(self) -> None:
        """Over the full-parse ceiling only the tail is checked for a matching
        closer, so a properly closed large array passes without parsing."""
        content = b"[" + b" " * 300000 + b"]"
        head = content[:SNIFF_HEAD_BYTES]
        tail = content[-SNIFF_TAIL_BYTES:]
        _check_json("f.json", head, tail, make_upload_file(content))

    def test_large_json_not_closed_raises(self) -> None:
        content = b"[" + b" " * 300000
        head = content[:SNIFF_HEAD_BYTES]
        tail = content[-SNIFF_TAIL_BYTES:]
        with pytest.raises(DocumentValidationError) as exc:
            _check_json("f.json", head, tail, make_upload_file(content))
        assert "not closed correctly" in exc.value.reason


class TestReadEdges:
    def test_head_only_when_tail_not_needed(self) -> None:
        file = make_upload_file(b"hello world")
        head, tail = _read_edges(file, needs_tail=False)
        assert head == b"hello world"
        assert tail == b""
        assert file.file.tell() == 0

    def test_head_and_tail_when_needed(self) -> None:
        content = b"START" + b"x" * 10000 + b"END"
        file = make_upload_file(content)
        head, tail = _read_edges(file, needs_tail=True)
        assert head == content[:SNIFF_HEAD_BYTES]
        assert tail == content[-SNIFF_TAIL_BYTES:]
        assert file.file.tell() == 0


class TestValidateDocumentContent:
    def test_empty_file_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            validate_document_content(file=make_upload_file(b""), source_format="csv")
        assert "the file is empty" in exc.value.reason

    def test_unknown_format_skips_validation(self) -> None:
        """A format with no FormatSpec is accepted as-is, even binary junk."""
        result = validate_document_content(
            file=make_upload_file(b"\x00\x01 arbitrary"), source_format="txt"
        )
        assert result is None

    def test_signature_mismatch_raises(self) -> None:
        with pytest.raises(DocumentValidationError) as exc:
            validate_document_content(
                file=make_upload_file(b"not a pdf %%EOF"), source_format="pdf"
            )
        assert "does not match the pdf file signature" in exc.value.reason

    def test_valid_pdf_dispatches_to_checker_and_passes(self) -> None:
        content = b"%PDF-1.4\n" + b"body " * 100 + b"trailer\n%%EOF\n"
        validate_document_content(
            file=make_upload_file(content), source_format="pdf"
        )


class TestValidateUpload:
    def test_valid_csv_returns_format_and_transformer(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.services.documents.helpers.pre_transform_validation",
            lambda **kwargs: ("csv", None),
        )
        result = validate_upload(
            src=make_upload_file(b"a,b,c\n1,2,3\n"),
            target_format=None,
            transformer=None,
        )
        assert result == ("csv", None)

    def test_invalid_content_raises_http_400_with_client_message(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "app.services.documents.helpers.pre_transform_validation",
            lambda **kwargs: ("csv", None),
        )
        with pytest.raises(HTTPException) as exc:
            validate_upload(
                src=make_upload_file(b"a,b\x00,c\n"),
                target_format=None,
                transformer=None,
            )
        assert exc.value.status_code == 400
        assert "NUL byte" not in exc.value.detail
        assert "corrupted" in exc.value.detail
