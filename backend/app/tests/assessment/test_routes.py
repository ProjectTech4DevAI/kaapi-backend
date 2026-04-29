"""Tests for assessment/routes.py."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.assessment.models import AssessmentCreate, AssessmentExportRow
from app.assessment.routes import (
    _dataset_to_response,
    create_evaluation,
    delete_dataset,
    export_assessment_results,
    export_assessment_run_results,
    get_assessment_manager,
    get_dataset,
    get_evaluation,
    list_assessment_managers,
    list_datasets,
    list_evaluations,
    retry_assessment_evaluation,
    retry_assessment_manager,
    stream_assessment_events,
)


def _auth_context() -> SimpleNamespace:
    return SimpleNamespace(
        organization_=SimpleNamespace(id=1),
        project_=SimpleNamespace(id=1),
    )


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        name="ds",
        description="d",
        dataset_metadata={"total_items_count": 2, "file_extension": ".csv"},
        object_store_url="s3://x",
    )


def _assessment() -> SimpleNamespace:
    return SimpleNamespace(
        id=10,
        experiment_name="exp",
        dataset_id=7,
        dataset_name="ds",
        status="processing",
        total_runs=1,
        pending_runs=0,
        processing_runs=1,
        completed_runs=0,
        failed_runs=0,
        run_stats=[],
        error_message=None,
        organization_id=1,
        project_id=1,
        inserted_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def _run() -> SimpleNamespace:
    return SimpleNamespace(
        id=22,
        assessment_id=10,
        run_name="exp",
        dataset_name="ds",
        dataset_id=7,
        config_id=UUID("00000000-0000-0000-0000-000000000001"),
        config_version=1,
        status="completed",
        total_items=1,
        error_message=None,
        organization_id=1,
        project_id=1,
        input=None,
        inserted_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def _row(run_id: int = 22) -> AssessmentExportRow:
    return AssessmentExportRow(
        assessment_id=10,
        experiment_name="exp",
        dataset_id=7,
        dataset_name="ds",
        run_id=run_id,
        run_name="exp",
        run_status="completed",
        config_id=None,
        config_version=1,
        row_id="row_0",
        result_status="passed",
        input_data={"q": "x"},
        output='{"score":1}',
        error=None,
        response_id="r",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        updated_at=datetime(2024, 1, 1),
    )


class TestRouteHelpers:
    def test_dataset_to_response(self) -> None:
        resp = _dataset_to_response(_dataset(), signed_url="signed")
        assert resp.dataset_id == 7
        assert resp.signed_url == "signed"


class TestDatasetRoutes:
    def test_list_datasets(self) -> None:
        with patch(
            "app.assessment.routes.list_evaluation_datasets", return_value=[_dataset()]
        ):
            resp = list_datasets(session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True
        assert len(resp.data or []) == 1

    def test_get_dataset_not_found(self) -> None:
        with patch("app.assessment.routes.get_dataset_by_id", return_value=None):
            with pytest.raises(HTTPException, match="not found"):
                get_dataset(1, session=MagicMock(), auth_context=_auth_context())

    def test_get_dataset_with_signed_url(self) -> None:
        storage = MagicMock()
        storage.get_signed_url.return_value = "signed-url"
        with patch("app.assessment.routes.get_dataset_by_id", return_value=_dataset()), patch(
            "app.assessment.routes.get_cloud_storage", return_value=storage
        ):
            resp = get_dataset(
                7,
                session=MagicMock(),
                auth_context=_auth_context(),
                include_signed_url=True,
            )
        assert resp.success is True
        assert resp.data is not None
        assert resp.data.signed_url == "signed-url"

    def test_delete_dataset_success_and_error(self) -> None:
        with patch("app.assessment.routes.get_dataset_by_id", return_value=_dataset()), patch(
            "app.assessment.routes.delete_dataset_crud", return_value=None
        ):
            resp = delete_dataset(7, session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True

        with patch("app.assessment.routes.get_dataset_by_id", return_value=_dataset()), patch(
            "app.assessment.routes.delete_dataset_crud",
            return_value="cannot delete",
        ):
            with pytest.raises(HTTPException, match="cannot delete"):
                delete_dataset(7, session=MagicMock(), auth_context=_auth_context())


class TestEvaluationRoutes:
    def test_create_evaluation(self) -> None:
        request = AssessmentCreate(
            experiment_name="exp",
            dataset_id=7,
            configs=[
                {
                    "config_id": "00000000-0000-0000-0000-000000000001",
                    "config_version": 1,
                }
            ],
        )
        result = SimpleNamespace(
            assessment_id=10,
            experiment_name="exp",
            dataset_id=7,
            dataset_name="ds",
            num_configs=1,
            runs=[],
        )
        with patch("app.assessment.routes.start_assessment", return_value=result):
            resp = create_evaluation(request, session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True

    def test_retry_endpoints(self) -> None:
        result = SimpleNamespace(
            assessment_id=10,
            experiment_name="exp",
            dataset_id=7,
            dataset_name="ds",
            num_configs=1,
            runs=[],
        )
        with patch("app.assessment.routes.get_assessment_by_id", return_value=_assessment()), patch(
            "app.assessment.routes.retry_assessment", return_value=result
        ):
            resp = retry_assessment_manager(10, session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True

        with patch("app.assessment.routes.get_assessment_run_by_id", return_value=_run()), patch(
            "app.assessment.routes.retry_assessment_run", return_value=result
        ):
            resp = retry_assessment_evaluation(
                22, session=MagicMock(), auth_context=_auth_context()
            )
        assert resp.success is True

    @pytest.mark.asyncio
    async def test_stream_assessment_events(self) -> None:
        async def gen():
            yield "event: x\ndata: {}\n\n"

        with patch(
            "app.assessment.routes.assessment_event_broker.subscribe",
            new=AsyncMock(return_value=gen()),
        ):
            response = await stream_assessment_events(
                _session=MagicMock(),
                _auth_context=_auth_context(),
            )
        assert isinstance(response, StreamingResponse)


class TestManagerAndRunRoutes:
    def test_list_and_get_managers(self) -> None:
        with patch("app.assessment.routes.list_assessments", return_value=[_assessment()]):
            resp = list_assessment_managers(
                session=MagicMock(),
                auth_context=_auth_context(),
            )
        assert resp.success is True
        assert len(resp.data or []) == 1

        with patch("app.assessment.routes.get_assessment_by_id", return_value=_assessment()):
            resp = get_assessment_manager(
                10,
                session=MagicMock(),
                auth_context=_auth_context(),
            )
        assert resp.success is True

        with patch("app.assessment.routes.get_assessment_by_id", return_value=None):
            with pytest.raises(HTTPException, match="not found"):
                get_assessment_manager(10, session=MagicMock(), auth_context=_auth_context())

    def test_list_and_get_runs(self) -> None:
        with patch("app.assessment.routes.list_assessment_runs", return_value=[_run()]):
            resp = list_evaluations(session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True

        with patch("app.assessment.routes.get_assessment_run_by_id", return_value=_run()):
            resp = get_evaluation(22, session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True

        with patch("app.assessment.routes.get_assessment_run_by_id", return_value=None):
            with pytest.raises(HTTPException, match="not found"):
                get_evaluation(22, session=MagicMock(), auth_context=_auth_context())


class TestExportRoutes:
    def test_export_assessment_results_json_and_zip(self) -> None:
        run1 = _run()
        run2 = _run()
        run2.id = 23
        run2.config_version = 2
        with patch("app.assessment.routes.get_assessment_by_id", return_value=_assessment()), patch(
            "app.assessment.routes.list_assessment_runs", return_value=[run1, run2]
        ), patch(
            "app.assessment.routes.load_export_rows_for_run",
            side_effect=[[ _row(run_id=22)], [_row(run_id=23)]],
        ), patch(
            "app.assessment.routes.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.assessment.routes.build_json_export_rows",
            return_value=[{"x": 1}],
        ):
            json_resp = export_assessment_results(
                10,
                session=MagicMock(),
                auth_context=_auth_context(),
                export_format="json",
            )
        assert json_resp.success is True

        with patch("app.assessment.routes.get_assessment_by_id", return_value=_assessment()), patch(
            "app.assessment.routes.list_assessment_runs", return_value=[run1, run2]
        ), patch(
            "app.assessment.routes.load_export_rows_for_run",
            side_effect=[[ _row(run_id=22)], [_row(run_id=23)]],
        ), patch(
            "app.assessment.routes.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.assessment.routes.serialize_export_rows",
            return_value=(b"csv", "text/csv"),
        ), patch(
            "app.assessment.routes.generate_timestamped_filename",
            return_value="out.zip",
        ):
            zip_resp = export_assessment_results(
                10,
                session=MagicMock(),
                auth_context=_auth_context(),
                export_format="csv",
            )
        assert isinstance(zip_resp, StreamingResponse)

    def test_export_assessment_run_results_json_and_file(self) -> None:
        run = _run()
        with patch("app.assessment.routes.get_assessment_run_by_id", return_value=run), patch(
            "app.assessment.routes.get_assessment_by_id", return_value=_assessment()
        ), patch(
            "app.assessment.routes.load_export_rows_for_run", return_value=[_row()]
        ), patch(
            "app.assessment.routes.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.assessment.routes.build_json_export_rows",
            return_value=[{"x": 1}],
        ):
            json_resp = export_assessment_run_results(
                22,
                session=MagicMock(),
                auth_context=_auth_context(),
                export_format="json",
            )
        assert json_resp.success is True

        with patch("app.assessment.routes.get_assessment_run_by_id", return_value=run), patch(
            "app.assessment.routes.get_assessment_by_id", return_value=_assessment()
        ), patch(
            "app.assessment.routes.load_export_rows_for_run", return_value=[_row()]
        ), patch(
            "app.assessment.routes.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.assessment.routes.build_export_response",
            return_value=StreamingResponse(iter([b"x"])),
        ):
            file_resp = export_assessment_run_results(
                22,
                session=MagicMock(),
                auth_context=_auth_context(),
                export_format="csv",
            )
        assert isinstance(file_resp, StreamingResponse)

    def test_export_not_found(self) -> None:
        with patch("app.assessment.routes.get_assessment_by_id", return_value=None):
            with pytest.raises(HTTPException, match="not found"):
                export_assessment_results(
                    10,
                    session=MagicMock(),
                    auth_context=_auth_context(),
                )
        with patch("app.assessment.routes.get_assessment_run_by_id", return_value=None):
            with pytest.raises(HTTPException, match="not found"):
                export_assessment_run_results(
                    22,
                    session=MagicMock(),
                    auth_context=_auth_context(),
                )
