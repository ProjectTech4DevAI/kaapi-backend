"""Tests for assessment/batch.py provider routing in submit_assessment_batch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import io
import pytest
from openpyxl import Workbook
from openpyxl.utils.exceptions import InvalidFileException

from app.assessment.batch import (
    _build_text_prompt,
    _decode_base64_prefix,
    _guess_image_mime_from_base64,
    _guess_image_mime_from_url,
    _load_dataset_rows,
    _parse_excel_rows,
    _resolve_attachment_values,
    _resolve_image_mime_and_payload,
    _split_attachment_urls,
    _split_data_url,
    _to_direct_attachment_url,
    build_google_jsonl,
    build_openai_jsonl,
    submit_assessment_batch,
)
from app.assessment.models import AssessmentAttachment


def _make_run() -> MagicMock:
    run = MagicMock()
    run.id = 99
    run.run_name = "exp-v1"
    return run


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
            completion=SimpleNamespace(provider="openai-native", params={})
        )
        batch_job = MagicMock()
        batch_job.id = 1
        batch_job.total_items = 1

        with patch(
            "app.assessment.batch._load_dataset_rows",
            return_value=[{"question": "q1"}],
        ), patch(
            "app.assessment.batch.map_kaapi_to_openai_params",
            return_value=({}, []),
        ), patch(
            "app.assessment.batch.build_openai_jsonl",
            return_value=[{"custom_id": "row_0"}],
        ), patch(
            "app.utils.get_openai_client",
            return_value=MagicMock(),
        ), patch(
            "app.assessment.batch.OpenAIBatchProvider",
            return_value=MagicMock(),
        ), patch(
            "app.assessment.batch.start_batch_job",
            return_value=batch_job,
        ) as start_batch:
            result = submit_assessment_batch(
                session=session,
                run=run,
                dataset=dataset,
                config_blob=config_blob,
                assessment_input={"text_columns": ["question"], "attachments": []},
                organization_id=1,
                project_id=1,
            )

        assert result.id == 1
        assert start_batch.call_args.kwargs["provider_name"] == "openai"

    def test_google_native_routes_to_google_batch(self) -> None:
        session = MagicMock()
        run = _make_run()
        dataset = _make_dataset()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(provider="google-native", params={})
        )
        batch_job = MagicMock()
        batch_job.id = 2
        batch_job.total_items = 1
        gemini_client = MagicMock()
        gemini_client.client = MagicMock()

        with patch(
            "app.assessment.batch._load_dataset_rows",
            return_value=[{"question": "q1"}],
        ), patch(
            "app.assessment.batch.map_kaapi_to_google_params",
            return_value=({"model": "gemini-2.5-pro"}, []),
        ), patch(
            "app.assessment.batch.build_google_jsonl",
            return_value=[{"key": "row_0"}],
        ), patch(
            "app.core.batch.client.GeminiClient"
        ) as gemini_cls, patch(
            "app.core.batch.GeminiBatchProvider",
            return_value=MagicMock(),
        ), patch(
            "app.assessment.batch.start_batch_job",
            return_value=batch_job,
        ) as start_batch:
            gemini_cls.from_credentials.return_value = gemini_client
            result = submit_assessment_batch(
                session=session,
                run=run,
                dataset=dataset,
                config_blob=config_blob,
                assessment_input={"text_columns": ["question"], "attachments": []},
                organization_id=1,
                project_id=1,
            )

        assert result.id == 2
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
        with patch(
            "app.assessment.batch.get_cloud_storage", return_value=storage
        ), patch(
            "app.assessment.batch._parse_excel_rows",
            return_value=expected,
        ) as parse_excel:
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

        with patch("app.assessment.batch.get_cloud_storage", return_value=storage):
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
        with patch("app.assessment.batch.openpyxl.load_workbook", return_value=fake_wb):
            assert _parse_excel_rows(b"irrelevant") == []
        fake_wb.close.assert_called_once()

    def test_parse_excel_rows_returns_empty_when_header_missing(self) -> None:
        fake_ws = MagicMock()
        fake_ws.iter_rows.return_value = iter([])
        fake_wb = MagicMock()
        fake_wb.active = fake_ws
        with patch("app.assessment.batch.openpyxl.load_workbook", return_value=fake_wb):
            assert _parse_excel_rows(b"irrelevant") == []
        fake_wb.close.assert_called_once()

    def test_parse_excel_rows_invalid_file_exception_re_raises(self) -> None:
        with patch(
            "app.assessment.batch.openpyxl.load_workbook",
            side_effect=InvalidFileException("bad xlsx"),
        ):
            with pytest.raises(InvalidFileException):
                _parse_excel_rows(b"bad")

    def test_parse_excel_rows_unexpected_exception_raises_value_error(self) -> None:
        with patch(
            "app.assessment.batch.openpyxl.load_workbook",
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
        urls = _split_attachment_urls(" https://a.com\nhttps://b.com , https://c.com ")
        assert urls == ["https://a.com", "https://b.com", "https://c.com"]
        image_url = _to_direct_attachment_url(
            "https://drive.google.com/file/d/abc123/view?usp=sharing", "image"
        )
        assert "googleusercontent.com" in image_url
        pdf_url = _to_direct_attachment_url(
            "https://drive.google.com/open?id=abc123", "pdf"
        )
        assert "drive.google.com/uc" in pdf_url

    def test_data_url_and_mime_guessers(self) -> None:
        mime, payload = _split_data_url("data:image/png;base64,AAAA")
        assert mime == "image/png"
        assert payload == "AAAA"
        none_mime, raw = _split_data_url("rawbase64")
        assert none_mime is None
        assert raw == "rawbase64"
        assert _guess_image_mime_from_url("https://x/y/file.jpeg") == "image/jpeg"
        assert _guess_image_mime_from_url("https://x/y/file.unknown") is None

    def test_base64_guess_and_decode(self) -> None:
        png_head = "iVBORw0KGgoAAAANSUhEUg=="
        assert _guess_image_mime_from_base64(png_head) == "image/png"
        assert _decode_base64_prefix("###") == b""

    def test_resolve_image_mime_and_payload(self) -> None:
        mime, payload = _resolve_image_mime_and_payload("https://x/y/file.webp", "url")
        assert mime == "image/webp"
        assert payload.endswith("file.webp")
        mime2, payload2 = _resolve_image_mime_and_payload(
            "data:image/jpeg;base64,AAAA", "base64"
        )
        assert mime2 == "image/jpeg"
        assert payload2 == "AAAA"

    def test_resolve_attachment_values(self) -> None:
        image_url_att = AssessmentAttachment(column="img", type="image", format="url")
        image_b64_att = AssessmentAttachment(
            column="img", type="image", format="base64"
        )
        pdf_url_att = AssessmentAttachment(column="pdf", type="pdf", format="url")
        pdf_b64_att = AssessmentAttachment(column="pdf", type="pdf", format="base64")

        values = _resolve_attachment_values(
            "https://example.com/a.png,https://example.com/b.png", image_url_att
        )
        assert len(values) == 2
        assert values[0]["type"] == "input_image"

        values = _resolve_attachment_values("data:image/png;base64,AAAA", image_b64_att)
        assert values[0]["image_url"].startswith("data:image/png;base64,")

        values = _resolve_attachment_values("https://example.com/a.pdf", pdf_url_att)
        assert values[0]["type"] == "input_file"
        assert "file_url" in values[0]

        values = _resolve_attachment_values(
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
        assert google_jsonl[0]["metadata"]["key"] == "row_0"
