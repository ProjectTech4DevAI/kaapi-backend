import csv
import json
import logging
from codecs import getincrementaldecoder
from dataclasses import dataclass
from typing import Callable, Tuple

from fastapi import UploadFile


logger = logging.getLogger(__name__)

SNIFF_HEAD_BYTES = 4096
SNIFF_TAIL_BYTES = 2048

PDF_EOF_MARKER = b"%%EOF"
PDF_ENCRYPT_MARKER = b"/Encrypt"
OOXML_CONTENT_TYPES = b"[Content_Types].xml"
OOXML_WORD_PART = b"word/"
OOXML_EXCEL_PART = b"xl/"
JSON_OPENERS = (b"{", b"[")
JSON_CLOSERS: dict[int, int] = {ord("{"): ord("}"), ord("["): ord("]")}
UTF8_BOM = b"\xef\xbb\xbf"

JSON_FULL_PARSE_MAX_BYTES = 256 * 1024

CSV_STRICT_COLUMN_COUNT = True
CSV_SAMPLE_MAX_ROWS = 200

OLE2_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# The OLE2 signature is a generic Compound File Binary container shared by .xls and
# .doc, so it can't tell them apart. The CFB directory stores per-stream names as
# UTF-16LE: an .xls carries a Workbook (BIFF8) or Book (BIFF5) stream, a .doc carries
# WordDocument.
OLE2_EXCEL_STREAMS = ("Workbook".encode("utf-16-le"), "Book".encode("utf-16-le"))
OLE2_WORD_STREAMS = ("WordDocument".encode("utf-16-le"),)

MISLABELLED_BINARY_SIGNATURES: dict[bytes, str] = {
    b"PK\x03\x04": "an Office/zip file (xlsx, docx)",
    OLE2_SIGNATURE: "a legacy Office file (xls, doc)",
    b"%PDF-": "a PDF",
}

UNNAMED_FILE_PLACEHOLDER = "<unnamed>"


CLIENT_VALIDATION_MESSAGE = (
    "Document '{filename}' failed parsing. The document appears to be corrupted "
    "or invalid. Please re-upload a valid document."
)


class DocumentValidationError(Exception):
    """Raised when a document fails the pre-upload sanity check."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Document '{filename}' failed validation: {reason}")

    @property
    def client_message(self) -> str:
        """Client-safe wording. `reason` stays internal - it is log-only."""
        return CLIENT_VALIDATION_MESSAGE.format(filename=self.filename)


def _decode_utf8(filename: str, sample: bytes) -> str:
    if b"\x00" in sample:
        raise DocumentValidationError(
            filename,
            "parsing error - NUL byte found in a text-based file; "
            "the file is binary, corrupt, or has the wrong extension",
        )
    try:
        return getincrementaldecoder("utf-8")().decode(sample)
    except UnicodeDecodeError as e:
        raise DocumentValidationError(
            filename,
            f"parsing error - content is not valid UTF-8 at byte offset {e.start}; "
            "re-export the file as UTF-8",
        ) from e


def _check_pdf(filename: str, head: bytes, tail: bytes, file: UploadFile) -> None:
    if PDF_EOF_MARKER not in tail:
        raise DocumentValidationError(
            filename,
            "parsing error - PDF is missing its end-of-file marker; "
            "the upload is truncated or incomplete",
        )
    if PDF_ENCRYPT_MARKER in tail:
        raise DocumentValidationError(
            filename,
            "parsing error - PDF is password protected or encrypted and cannot be read",
        )


def _check_ooxml(filename: str, sample: bytes, expected_part: bytes) -> None:
    if OOXML_CONTENT_TYPES not in sample:
        raise DocumentValidationError(
            filename,
            "parsing error - file is a zip archive but not a valid Office document "
            "(no OOXML content-types part)",
        )

    other_part = (
        OOXML_EXCEL_PART if expected_part == OOXML_WORD_PART else OOXML_WORD_PART
    )
    if other_part in sample and expected_part not in sample:
        raise DocumentValidationError(
            filename,
            f"parsing error - file is a '{other_part.decode()}' Office package, "
            "not the format its extension claims",
        )


def _check_docx(filename: str, head: bytes, tail: bytes, file: UploadFile) -> None:
    """Head+tail: the zip central directory lists all entry names and sits at the
    end; the head holds only the first entry."""
    _check_ooxml(filename, head + tail, OOXML_WORD_PART)


def _check_xlsx(filename: str, head: bytes, tail: bytes, file: UploadFile) -> None:
    _check_ooxml(filename, head + tail, OOXML_EXCEL_PART)


def _check_ole2_stream(
    filename: str,
    sample: bytes,
    wanted: tuple[bytes, ...],
    rivals: tuple[bytes, ...],
    rival_label: str,
) -> None:
    """Reject an OLE2 file whose extension claims one format but whose CFB directory
    holds the rival's stream. Inconclusive samples pass — the directory can sit past
    the sampled edges, and rejecting a valid file is worse than the loose check."""
    if any(name in sample for name in wanted):
        return
    if any(name in sample for name in rivals):
        raise DocumentValidationError(
            filename,
            f"parsing error - file is a {rival_label} document, "
            "not the format its extension claims",
        )


def _check_xls(filename: str, head: bytes, tail: bytes, file: UploadFile) -> None:
    _check_ole2_stream(
        filename, head + tail, OLE2_EXCEL_STREAMS, OLE2_WORD_STREAMS, "Word (.doc)"
    )


def _check_doc(filename: str, head: bytes, tail: bytes, file: UploadFile) -> None:
    _check_ole2_stream(
        filename, head + tail, OLE2_WORD_STREAMS, OLE2_EXCEL_STREAMS, "Excel (.xls)"
    )


def _check_csv(filename: str, head: bytes, tail: bytes, file: UploadFile) -> None:
    for signature, description in MISLABELLED_BINARY_SIGNATURES.items():
        if head.startswith(signature):
            raise DocumentValidationError(
                filename,
                f"parsing error - file is {description} renamed to a CSV, not text; "
                "export it as CSV instead of renaming it",
            )

    text = _decode_utf8(filename, head)

    lines = text.splitlines()
    if len(head) == SNIFF_HEAD_BYTES and len(lines) > 1:
        lines = lines[:-1]

    try:
        rows = [row for row in csv.reader(lines[:CSV_SAMPLE_MAX_ROWS]) if row]
    except csv.Error as e:
        raise DocumentValidationError(
            filename, f"parsing error - malformed CSV: {e}"
        ) from e

    if not rows:
        raise DocumentValidationError(
            filename, "parsing error - no readable CSV rows found"
        )

    if CSV_STRICT_COLUMN_COUNT:
        expected_columns = len(rows[0])
        for line_number, row in enumerate(rows[1:], start=2):
            if len(row) != expected_columns:
                raise DocumentValidationError(
                    filename,
                    f"parsing error - inconsistent column count at row {line_number} "
                    f"({len(row)} fields, expected {expected_columns})",
                )


def _check_json_tail(filename: str, opener: int, tail: bytes) -> None:
    """Truncation check for JSON too large to parse: the opening bracket must be
    matched by the last non-whitespace byte."""
    closing = tail.rstrip()
    if not closing or closing[-1] != JSON_CLOSERS[opener]:
        raise DocumentValidationError(
            filename,
            "parsing error - JSON is not closed correctly; "
            "the upload is truncated or incomplete",
        )


def _check_json(filename: str, head: bytes, tail: bytes, file: UploadFile) -> None:
    body = head.lstrip(UTF8_BOM).lstrip()
    if not body.startswith(JSON_OPENERS):
        raise DocumentValidationError(
            filename, "parsing error - JSON document must start with '{' or '['"
        )

    stream = file.file
    stream.seek(0, 2)
    size_bytes = stream.tell()
    stream.seek(0)

    if size_bytes > JSON_FULL_PARSE_MAX_BYTES:
        logger.info(
            f"[_check_json] Too large for a full parse, checking closure only | "
            f"filename: {filename} | size_bytes: {size_bytes}"
        )
        _check_json_tail(filename, body[0], tail)
        return

    try:
        json.loads(stream.read(), object_pairs_hook=lambda pairs: None)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise DocumentValidationError(
            filename, f"parsing error - invalid JSON: {e}"
        ) from e
    finally:
        stream.seek(0)


@dataclass(frozen=True)
class FormatSpec:
    signatures: tuple[bytes, ...] = ()
    needs_tail: bool = False
    checker: Callable[[str, bytes, bytes, UploadFile], None] | None = None


FORMAT_SPECS: dict[str, FormatSpec] = {
    "pdf": FormatSpec(signatures=(b"%PDF-",), needs_tail=True, checker=_check_pdf),
    "docx": FormatSpec(
        signatures=(b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
        needs_tail=True,
        checker=_check_docx,
    ),
    "xlsx": FormatSpec(
        signatures=(b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
        needs_tail=True,
        checker=_check_xlsx,
    ),
    "xls": FormatSpec(
        signatures=(OLE2_SIGNATURE,), needs_tail=True, checker=_check_xls
    ),
    "doc": FormatSpec(
        signatures=(OLE2_SIGNATURE,), needs_tail=True, checker=_check_doc
    ),
    "csv": FormatSpec(checker=_check_csv),
    "json": FormatSpec(needs_tail=True, checker=_check_json),
}


def _read_edges(file: UploadFile, needs_tail: bool) -> Tuple[bytes, bytes]:
    """Sample the head (and optionally tail) of the upload, leaving it rewound."""
    stream = file.file

    stream.seek(0)
    head = stream.read(SNIFF_HEAD_BYTES)

    tail = b""
    if needs_tail:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - SNIFF_TAIL_BYTES))
        tail = stream.read(SNIFF_TAIL_BYTES)

    stream.seek(0)
    return head, tail


def validate_document_content(*, file: UploadFile, source_format: str) -> None:
    """
    Sanity-check an upload's bytes before it is persisted. Only the first
    SNIFF_HEAD_BYTES (plus SNIFF_TAIL_BYTES for PDF) are read, so the cost is
    constant in file size - small JSON, which is fully parsed, is the exception.

    Raises:
        DocumentValidationError: naming the offending file and the reason.
    """
    filename = (file.filename or "").strip() or UNNAMED_FILE_PLACEHOLDER

    spec = FORMAT_SPECS.get(source_format)
    head, tail = _read_edges(file, needs_tail=bool(spec and spec.needs_tail))

    if not head:
        raise DocumentValidationError(filename, "the file is empty")

    if spec is None:
        return

    if spec.signatures and not head.startswith(spec.signatures):
        raise DocumentValidationError(
            filename,
            f"parsing error - content does not match the {source_format} file "
            "signature; the file is corrupt or has the wrong extension",
        )

    if spec.checker:
        spec.checker(filename, head, tail, file)
