"""Tests for assessment/batch.py provider routing in submit_assessment_batch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openpyxl.utils.exceptions import InvalidFileException

from app.assessment.batch import (
    _load_dataset_rows,
    _parse_excel_rows,
    submit_assessment_batch,
)


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
