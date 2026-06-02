"""Tests for assessment/batch.py provider routing in submit_assessment_batch."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openpyxl import Workbook
from openpyxl.utils.exceptions import InvalidFileException

from app.crud.assessment.batch import (
    _build_text_prompt,
    _load_dataset_rows,
    _parse_excel_rows,
    build_google_jsonl,
    build_openai_jsonl,
    submit_assessment_batch,
)
from app.models.assessment import AssessmentAttachment
from app.services.assessment.utils.attachments import (
    _decode_base64_prefix,
    _guess_image_mime_from_base64,
    _guess_image_mime_from_url,
    detect_item_type,
    resolve_attachment_values,
    resolve_image_mime_and_payload,
    split_attachment_urls,
    split_data_url,
    to_direct_attachment_url,
)


def _make_run() -> MagicMock:
    run = MagicMock()
    run.id = 99
    return run


def _make_assessment() -> MagicMock:
    assessment = MagicMock()
    assessment.id = 21
    assessment.experiment_name = "exp-v1"
    return assessment


def _make_dataset() -> MagicMock:
    dataset = MagicMock()
    dataset.id = 8
    return dataset


class TestSubmitAssessmentBatchProviderRouting:
    def test_openai_native_routes_to_openai_batch(self) -> None:
        session = MagicMock()
        run = _make_run()
        dataset = _make_dataset()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(
                provider="openai-native",
                params={"instructions": "config system"},
            )
        )
        batch_job = MagicMock()
        batch_job.id = 1
        batch_job.total_items = 1

        with (
            patch(
                "app.crud.assessment.batch._load_dataset_rows",
                return_value=[{"question": "q1"}],
            ),
            patch(
                "app.crud.assessment.batch.map_kaapi_to_openai_params",
                return_value=({}, []),
            ) as map_params,
            patch(
                "app.crud.assessment.batch.build_openai_jsonl",
                return_value=[{"custom_id": "row_0"}],
            ),
            patch(
                "app.utils.get_openai_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.crud.assessment.batch.OpenAIBatchProvider",
                return_value=MagicMock(),
            ),
            patch(
                "app.crud.assessment.batch.start_batch_job",
                return_value=batch_job,
            ) as start_batch,
        ):
            result = submit_assessment_batch(
                session=session,
                run=run,
                assessment=_make_assessment(),
                dataset=dataset,
                config_blob=config_blob,
                assessment_input={
                    "text_columns": ["question"],
                    "attachments": [],
                    "system_instruction": "request system",
                },
                organization_id=1,
                project_id=1,
            )

        assert result.id == 1
        assert map_params.call_args.kwargs["session"] is session
        assert map_params.call_args.kwargs["kaapi_params"]["instructions"] == (
            "request system"
        )
        assert start_batch.call_args.kwargs["provider_name"] == "openai"

    def test_config_instruction_is_not_used_without_request_instruction(self) -> None:
        session = MagicMock()
        run = _make_run()
        dataset = _make_dataset()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(
                provider="openai",
                params={"instructions": "config system", "model": "gpt-4.1-mini"},
            )
        )
        batch_job = MagicMock()
        batch_job.id = 3
        batch_job.total_items = 1

        with (
            patch(
                "app.crud.assessment.batch._load_dataset_rows",
                return_value=[{"question": "q1"}],
            ),
            patch(
                "app.crud.assessment.batch.map_kaapi_to_openai_params",
                return_value=({"model": "gpt-4.1-mini"}, []),
            ) as map_params,
            patch(
                "app.crud.assessment.batch.build_openai_jsonl",
                return_value=[{"custom_id": "row_0"}],
            ),
            patch(
                "app.utils.get_openai_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.crud.assessment.batch.OpenAIBatchProvider",
                return_value=MagicMock(),
            ),
            patch(
                "app.crud.assessment.batch.start_batch_job",
                return_value=batch_job,
            ),
        ):
            submit_assessment_batch(
                session=session,
                run=run,
                assessment=_make_assessment(),
                dataset=dataset,
                config_blob=config_blob,
                assessment_input={"text_columns": ["question"], "attachments": []},
                organization_id=1,
                project_id=1,
            )

        assert map_params.call_args.kwargs["session"] is session
        assert "instructions" not in map_params.call_args.kwargs["kaapi_params"]

    def test_google_native_routes_to_google_batch(self) -> None:
        session = MagicMock()
        run = _make_run()
        dataset = _make_dataset()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(
                provider="google-native",
                params={"instructions": "config system"},
            )
        )
        batch_job = MagicMock()
        batch_job.id = 2
        batch_job.total_items = 1
        gemini_client = MagicMock()
        gemini_client.client = MagicMock()

        with (
            patch(
                "app.crud.assessment.batch._load_dataset_rows",
                return_value=[{"question": "q1"}],
            ),
            patch(
                "app.crud.assessment.batch.map_kaapi_to_google_params",
                return_value=({"model": "gemini-2.5-pro"}, []),
            ) as map_params,
            patch(
                "app.crud.assessment.batch.build_google_jsonl",
                return_value=[{"key": "row_0"}],
            ),
            patch("app.core.batch.client.GeminiClient") as gemini_cls,
            patch(
                "app.core.batch.GeminiBatchProvider",
                return_value=MagicMock(),
            ),
            patch(
                "app.crud.assessment.batch.start_batch_job",
                return_value=batch_job,
            ) as start_batch,
        ):
            gemini_cls.from_credentials.return_value = gemini_client
            result = submit_assessment_batch(
                session=session,
                run=run,
                assessment=_make_assessment(),
                dataset=dataset,
                config_blob=config_blob,
                assessment_input={
                    "text_columns": ["question"],
                    "attachments": [],
                    "system_instruction": "request system",
                },
                organization_id=1,
                project_id=1,
            )

        assert result.id == 2
        assert map_params.call_args.args[0]["instructions"] == "request system"
        assert start_batch.call_args.kwargs["provider_name"] == "google"


class TestBatchDatasetParsing:
    def test_load_dataset_rows_routes_xlsx_to_excel_parser(self) -> None:
        session = MagicMock()
        dataset = MagicMock()
        dataset.id = 8
        dataset.project_id = 1
        dataset.object_store_url = "s3://bucket/key"
        dataset.dataset_metadata = {"file_extension": ".xlsx"}

        storage = MagicMock()
        stream_body = MagicMock()
        stream_body.read.return_value = b"xlsx-content"
        storage.stream.return_value = stream_body

        expected = [{"question": "q1"}]
        with (
            patch("app.crud.assessment.batch.get_cloud_storage", return_value=storage),
            patch(
                "app.crud.assessment.batch._parse_excel_rows",
                return_value=expected,
            ) as parse_excel,
        ):
            result = _load_dataset_rows(session=session, dataset=dataset)

        assert result == expected
        parse_excel.assert_called_once_with(b"xlsx-content")

    def test_load_dataset_rows_rejects_legacy_xls(self) -> None:
        session = MagicMock()
        dataset = MagicMock()
        dataset.id = 8
        dataset.project_id = 1
        dataset.object_store_url = "s3://bucket/key"
        dataset.dataset_metadata = {"file_extension": ".xls"}

        storage = MagicMock()
        stream_body = MagicMock()
        stream_body.read.return_value = b"legacy-xls-content"
        storage.stream.return_value = stream_body

        with patch("app.crud.assessment.batch.get_cloud_storage", return_value=storage):
            with pytest.raises(ValueError, match="Legacy Excel format"):
                _load_dataset_rows(session=session, dataset=dataset)

    def test_parse_excel_rows_invalid_payload_raises(self) -> None:
        with pytest.raises((ValueError, InvalidFileException)):
            _parse_excel_rows(b"not-a-valid-xlsx")

    def test_parse_excel_rows_success(self) -> None:
        wb = Workbook()
        ws = wb.active
        assert ws is not None
        ws.append(["question", "answer"])
        ws.append(["What is 2+2?", "4"])
        ws.append(["", None])  # empty row should be skipped
        buf = io.BytesIO()
        wb.save(buf)
        wb.close()

        rows = _parse_excel_rows(buf.getvalue())
        assert rows == [{"question": "What is 2+2?", "answer": "4"}]

    def test_parse_excel_rows_returns_empty_when_sheet_missing(self) -> None:
        fake_wb = MagicMock()
        fake_wb.active = None
        with patch(
            "app.crud.assessment.batch.openpyxl.load_workbook", return_value=fake_wb
        ):
            assert _parse_excel_rows(b"irrelevant") == []
        fake_wb.close.assert_called_once()

    def test_parse_excel_rows_returns_empty_when_header_missing(self) -> None:
        fake_ws = MagicMock()
        fake_ws.iter_rows.return_value = iter([])
        fake_wb = MagicMock()
        fake_wb.active = fake_ws
        with patch(
            "app.crud.assessment.batch.openpyxl.load_workbook", return_value=fake_wb
        ):
            assert _parse_excel_rows(b"irrelevant") == []
        fake_wb.close.assert_called_once()

    def test_parse_excel_rows_invalid_file_exception_re_raises(self) -> None:
        with patch(
            "app.crud.assessment.batch.openpyxl.load_workbook",
            side_effect=InvalidFileException("bad xlsx"),
        ):
            with pytest.raises(InvalidFileException):
                _parse_excel_rows(b"bad")

    def test_parse_excel_rows_unexpected_exception_raises_value_error(self) -> None:
        with patch(
            "app.crud.assessment.batch.openpyxl.load_workbook",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ValueError, match="Failed to parse XLSX dataset rows"):
                _parse_excel_rows(b"bad")


class TestBatchHelpers:
    def test_build_text_prompt_template_and_concat(self) -> None:
        row = {"q": " What? ", "ctx": "Context"}
        templated = _build_text_prompt(row, ["q", "ctx"], "Q:{q}\nC:{ctx}")
        assert "Q:" in templated
        assert "What?" in templated
        concatenated = _build_text_prompt(row, ["q", "ctx"], None)
        assert "What?" in concatenated
        assert concatenated.endswith("\nContext")

    def test_split_and_direct_urls(self) -> None:
        urls = split_attachment_urls(" https://a.com\nhttps://b.com , https://c.com ")
        assert urls == ["https://a.com", "https://b.com", "https://c.com"]
        image_url = to_direct_attachment_url(
            "https://drive.google.com/file/d/abc123/view?usp=sharing", "image"
        )
        assert "googleusercontent.com" in image_url
        pdf_url = to_direct_attachment_url(
            "https://drive.google.com/open?id=abc123", "pdf"
        )
        assert "drive.google.com/uc" in pdf_url

    def test_data_url_and_mime_guessers(self) -> None:
        mime, payload = split_data_url("data:image/png;base64,AAAA")
        assert mime == "image/png"
        assert payload == "AAAA"
        none_mime, raw = split_data_url("rawbase64")
        assert none_mime is None
        assert raw == "rawbase64"
        assert _guess_image_mime_from_url("https://x/y/file.jpeg") == "image/jpeg"
        assert _guess_image_mime_from_url("https://x/y/file.unknown") is None

    def test_base64_guess_and_decode(self) -> None:
        png_head = "iVBORw0KGgoAAAANSUhEUg=="
        assert _guess_image_mime_from_base64(png_head) == "image/png"
        assert _decode_base64_prefix("###") == b""

    def testresolve_image_mime_and_payload(self) -> None:
        mime, payload = resolve_image_mime_and_payload("https://x/y/file.webp", "url")
        assert mime == "image/webp"
        assert payload.endswith("file.webp")
        mime2, payload2 = resolve_image_mime_and_payload(
            "data:image/jpeg;base64,AAAA", "base64"
        )
        assert mime2 == "image/jpeg"
        assert payload2 == "AAAA"

    def testresolve_attachment_values(self) -> None:
        image_url_att = AssessmentAttachment(column="img", type="image", format="url")
        image_b64_att = AssessmentAttachment(
            column="img", type="image", format="base64"
        )
        pdf_url_att = AssessmentAttachment(column="pdf", type="pdf", format="url")
        pdf_b64_att = AssessmentAttachment(column="pdf", type="pdf", format="base64")

        values = resolve_attachment_values(
            "https://example.com/a.png,https://example.com/b.png", image_url_att
        )
        assert len(values) == 2
        assert values[0]["type"] == "input_image"

        values = resolve_attachment_values("data:image/png;base64,AAAA", image_b64_att)
        assert values[0]["image_url"].startswith("data:image/png;base64,")

        values = resolve_attachment_values("https://example.com/a.pdf", pdf_url_att)
        assert values[0]["type"] == "input_file"
        assert "file_url" in values[0]

        values = resolve_attachment_values(
            "data:application/pdf;base64,AAAA", pdf_b64_att
        )
        assert values[0]["file_data"].startswith("data:application/pdf;base64,")

    def test_build_openai_and_google_jsonl(self) -> None:
        rows = [{"q": "What is 2+2?", "img": "https://example.com/a.png"}]
        attachments = [AssessmentAttachment(column="img", type="image", format="url")]

        openai_jsonl = build_openai_jsonl(
            rows=rows,
            text_columns=["q"],
            attachments=attachments,
            prompt_template=None,
            openai_params={"model": "gpt-4.1-mini"},
        )
        assert len(openai_jsonl) == 1
        assert openai_jsonl[0]["custom_id"] == "row_0"

        google_jsonl = build_google_jsonl(
            rows=rows,
            text_columns=["q"],
            attachments=attachments,
            prompt_template=None,
            google_params={"temperature": 0.2, "instructions": "system"},
        )
        assert len(google_jsonl) == 1
        assert google_jsonl[0]["key"] == "row_0"
        assert google_jsonl[0]["request"]["systemInstruction"] == {
            "parts": [{"text": "system"}]
        }


class TestDetectItemType:
    """Per-item image/pdf detection for mixed-content attachment columns."""

    def test_data_url_pdf(self) -> None:
        assert (
            detect_item_type("data:application/pdf;base64,JVBERi0=", "base64", "image")
            == "pdf"
        )

    def test_data_url_image(self) -> None:
        assert (
            detect_item_type("data:image/png;base64,AAAA", "base64", "pdf") == "image"
        )

    def test_base64_magic_pdf(self) -> None:
        payload = base64.b64encode(b"%PDF-1.7 body").decode()
        assert detect_item_type(payload, "base64", "image") == "pdf"

    def test_base64_magic_png(self) -> None:
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 8).decode()
        assert detect_item_type(payload, "base64", "pdf") == "image"

    def test_base64_unknown_falls_back(self) -> None:
        payload = base64.b64encode(b"not a known magic").decode()
        assert detect_item_type(payload, "base64", "pdf") == "pdf"

    def test_mixed_fallback_resolves_to_image(self) -> None:
        """'mixed' is never a returned type; inconclusive detection -> image."""
        payload = base64.b64encode(b"not a known magic").decode()
        assert detect_item_type(payload, "base64", "mixed") == "image"

    def test_url_extension_pdf_case_insensitive(self) -> None:
        assert detect_item_type("https://x.com/a/scan.PDF", "url", "image", {}) == "pdf"

    def test_url_extension_image(self) -> None:
        assert detect_item_type("https://x.com/a/p.jpg", "url", "pdf", {}) == "image"

    def test_url_no_extension_probes_bytes(self) -> None:
        """Extensionless URL (Drive-style) is probed; magic bytes win over fallback."""
        url = "https://drive.google.com/file/d/ABC123/view"
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.raise_for_status = MagicMock()
        resp.iter_content = MagicMock(return_value=iter([b"%PDF-1.7"]))
        with patch(
            "app.services.assessment.utils.attachments.requests.get",
            return_value=resp,
        ) as mock_get:
            assert detect_item_type(url, "url", "image", {}) == "pdf"
        # Drive share URL is probed through the download endpoint.
        assert "uc?export=download&id=ABC123" in mock_get.call_args.args[0]

    def test_url_probe_uses_content_type_when_no_magic(self) -> None:
        url = "https://example.com/file"
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.raise_for_status = MagicMock()
        resp.iter_content = MagicMock(return_value=iter([b"\x00\x01\x02\x03"]))
        resp.headers = {"Content-Type": "application/pdf; charset=binary"}
        with patch(
            "app.services.assessment.utils.attachments.requests.get",
            return_value=resp,
        ):
            assert detect_item_type(url, "url", "image", {}) == "pdf"

    def test_url_probe_failure_falls_back(self) -> None:
        import requests as _requests

        url = "https://example.com/file"
        with patch(
            "app.services.assessment.utils.attachments.requests.get",
            side_effect=_requests.RequestException("boom"),
        ):
            assert detect_item_type(url, "url", "image", {}) == "image"

    def test_cache_skips_second_probe(self) -> None:
        url = "https://drive.google.com/file/d/XYZ/view"
        cache: dict[str, str] = {}
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.raise_for_status = MagicMock()
        resp.iter_content = MagicMock(return_value=iter([b"%PDF-1.7"]))
        with patch(
            "app.services.assessment.utils.attachments.requests.get",
            return_value=resp,
        ) as mock_get:
            assert detect_item_type(url, "url", "image", cache) == "pdf"
            assert detect_item_type(url, "url", "image", cache) == "pdf"
        assert mock_get.call_count == 1

    def test_mixed_column_resolves_both_types(self) -> None:
        """One column, two URLs with extensions -> one image, one pdf object."""
        att = AssessmentAttachment(column="docs", type="image", format="url")
        value = "https://x.com/a/photo.jpg, https://x.com/b/report.pdf"
        resolved = resolve_attachment_values(value, att, {})
        types = [obj["type"] for obj in resolved]
        assert types == ["input_image", "input_file"]


class TestAttachmentMagicAndMime:
    def test_image_magic_all_formats(self) -> None:
        from app.services.assessment.utils.attachments import _image_mime_from_magic

        assert _image_mime_from_magic(b"\x89PNG\r\n\x1a\n") == "image/png"
        assert _image_mime_from_magic(b"\xff\xd8\xff") == "image/jpeg"
        assert _image_mime_from_magic(b"GIF89a") == "image/gif"
        assert _image_mime_from_magic(b"GIF87a") == "image/gif"
        assert _image_mime_from_magic(b"BM....") == "image/bmp"
        assert _image_mime_from_magic(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"
        assert _image_mime_from_magic(b"II*\x00") == "image/tiff"
        assert _image_mime_from_magic(b"MM\x00*") == "image/tiff"
        assert _image_mime_from_magic(b"nope") is None

    def test_type_from_magic_pdf_and_none(self) -> None:
        from app.services.assessment.utils.attachments import _type_from_magic

        assert _type_from_magic(b"%PDF-1.7") == "pdf"
        assert _type_from_magic(b"\x89PNG\r\n\x1a\n") == "image"
        assert _type_from_magic(b"random") is None

    def test_guess_image_mime_from_url_variants(self) -> None:
        from app.services.assessment.utils.attachments import _guess_image_mime_from_url

        assert _guess_image_mime_from_url("http://x/a.PNG") == "image/png"
        assert _guess_image_mime_from_url("http://x/a.jpeg") == "image/jpeg"
        assert _guess_image_mime_from_url("http://x/a.webp") == "image/webp"
        assert _guess_image_mime_from_url("http://x/a.txt") is None

    def test_resolve_image_mime_data_url(self) -> None:
        from app.services.assessment.utils.attachments import (
            resolve_image_mime_and_payload,
        )

        mime, payload = resolve_image_mime_and_payload(
            "data:image/webp;base64,AAAA", "base64"
        )
        assert mime == "image/webp"
        assert payload == "AAAA"

    def test_decode_base64_prefix_empty(self) -> None:
        from app.services.assessment.utils.attachments import _decode_base64_prefix

        assert _decode_base64_prefix("   ") is None
