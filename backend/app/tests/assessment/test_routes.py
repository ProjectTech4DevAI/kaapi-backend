"""Tests for assessment route endpoints (split into datasets/assessments/runs)."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.api.routes.assessment.assessments import (
    export_assessment_results,
    get_assessment,
    list_assessments,
    retry_assessment,
)
from app.api.routes.assessment.datasets import (
    _dataset_to_response,
    delete_dataset,
    get_dataset,
    list_datasets,
)
from app.api.routes.assessment.runs import (
    create_assessment_runs,
    export_assessment_run_results,
    get_assessment_run,
    list_assessment_runs,
    retry_assessment_run,
)
from app.models.assessment import (
    AssessmentConfigRef,
    AssessmentExportRow,
    AssessmentRunCreate,
    InputBinding,
)

ASSESSMENT_UUID = UUID("00000000-0000-0000-0000-000000000010")

# ─── Fixtures ────────────────────────────────────────────────────────────────


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
        status="processing",
        organization_id=1,
        project_id=1,
        inserted_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def _run() -> SimpleNamespace:
    return SimpleNamespace(
        id=22,
        assessment_id=10,
        config_id=UUID("00000000-0000-0000-0000-000000000001"),
        config_version=1,
        status="completed",
        total_items=1,
        error_message=None,
        input=None,
        execution=None,
        post_processing_config=None,
        batch_job_id=None,
        inserted_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def _row(execution_id: int = 22) -> AssessmentExportRow:
    return AssessmentExportRow(
        assessment_id=ASSESSMENT_UUID,
        experiment_name="exp",
        execution_id=execution_id,
        execution_status="COMPLETED",
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


# ─── Helpers ─────────────────────────────────────────────────────────────────


class TestRouteHelpers:
    def test_dataset_to_response(self) -> None:
        resp = _dataset_to_response(_dataset(), signed_url="signed")
        assert resp.dataset_id == 7
        assert resp.signed_url == "signed"


# ─── Datasets ────────────────────────────────────────────────────────────────


class TestDatasetRoutes:
    def test_list_datasets(self) -> None:
        with patch(
            "app.api.routes.assessment.datasets.list_assessment_datasets",
            return_value=[_dataset()],
        ):
            resp = list_datasets(session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True
        assert len(resp.data or []) == 1

    def test_get_dataset_not_found(self) -> None:
        with patch(
            "app.api.routes.assessment.datasets.get_assessment_dataset_by_id",
            side_effect=HTTPException(
                status_code=404,
                detail="Dataset 1 not found or not accessible",
            ),
        ):
            with pytest.raises(HTTPException, match="not found"):
                get_dataset(1, session=MagicMock(), auth_context=_auth_context())

    def test_get_dataset_with_signed_url(self) -> None:
        storage = MagicMock()
        storage.get_signed_url.return_value = "signed-url"
        with patch(
            "app.api.routes.assessment.datasets.get_assessment_dataset_by_id",
            return_value=_dataset(),
        ), patch(
            "app.api.routes.assessment.datasets.get_cloud_storage", return_value=storage
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

    def test_get_dataset_with_limit_rows_includes_preview(self) -> None:
        with patch(
            "app.api.routes.assessment.datasets.get_assessment_dataset_by_id",
            return_value=_dataset(),
        ), patch(
            "app.api.routes.assessment.datasets.preview_assessment_dataset",
            return_value=(["a", "b"], [["1", "2"], ["3", "4"]]),
        ) as preview_mock:
            resp = get_dataset(
                7,
                session=MagicMock(),
                auth_context=_auth_context(),
                limit_rows=2,
            )
        preview_mock.assert_called_once()
        assert resp.data is not None
        assert resp.data.preview is not None
        assert resp.data.preview.headers == ["a", "b"]
        assert resp.data.preview.returned_rows == 2
        assert resp.data.preview.truncated is True

    def test_get_dataset_without_limit_rows_skips_preview(self) -> None:
        with patch(
            "app.api.routes.assessment.datasets.get_assessment_dataset_by_id",
            return_value=_dataset(),
        ), patch(
            "app.api.routes.assessment.datasets.preview_assessment_dataset"
        ) as preview_mock:
            resp = get_dataset(7, session=MagicMock(), auth_context=_auth_context())
        preview_mock.assert_not_called()
        assert resp.data is not None
        assert resp.data.preview is None

    def test_delete_dataset_success_and_error(self) -> None:
        with patch(
            "app.api.routes.assessment.datasets.get_assessment_dataset_by_id",
            return_value=_dataset(),
        ), patch(
            "app.api.routes.assessment.datasets.delete_assessment_dataset",
            return_value=None,
        ):
            resp = delete_dataset(7, session=MagicMock(), auth_context=_auth_context())
        assert resp.success is True

        with patch(
            "app.api.routes.assessment.datasets.get_assessment_dataset_by_id",
            return_value=_dataset(),
        ), patch(
            "app.api.routes.assessment.datasets.delete_assessment_dataset",
            return_value="cannot delete",
        ):
            with pytest.raises(HTTPException, match="cannot delete"):
                delete_dataset(7, session=MagicMock(), auth_context=_auth_context())


# ─── Runs — POST + retry ─────────────────────────────────────────────────────


class TestRunRoutes:
    def test_create_assessment_runs(self) -> None:
        request = AssessmentRunCreate(
            experiment_name="exp",
            dataset_id=7,
            input_binding=InputBinding(prompt="p", text_columns=[], attachments=[]),
            configs=[
                AssessmentConfigRef(
                    id=UUID("00000000-0000-0000-0000-000000000001"), version=1
                )
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
        with patch(
            "app.api.routes.assessment.runs.start_assessment", return_value=result
        ):
            resp = create_assessment_runs(
                request, session=MagicMock(), auth_context=_auth_context()
            )
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
        with patch(
            "app.api.routes.assessment.assessments.get_assessment_by_id",
            return_value=_assessment(),
        ), patch(
            "app.api.routes.assessment.assessments.retry_assessment_service",
            return_value=result,
        ):
            resp = retry_assessment(
                10, session=MagicMock(), auth_context=_auth_context()
            )
        assert resp.success is True

        with patch(
            "app.api.routes.assessment.runs.get_run_by_id",
            return_value=_run(),
        ), patch(
            "app.api.routes.assessment.runs.retry_run",
            return_value=result,
        ):
            resp = retry_assessment_run(
                22, session=MagicMock(), auth_context=_auth_context()
            )
        assert resp.success is True


# ─── Assessments (parents) — list/get + Runs list/get ───────────────────────


class TestAssessmentAndRunRoutes:
    def test_list_and_get_assessments(self) -> None:
        public_stub = MagicMock()
        with patch(
            "app.api.routes.assessment.assessments.list_assessments_crud",
            return_value=[_assessment()],
        ), patch(
            "app.api.routes.assessment.assessments._build_assessment_public",
            return_value=public_stub,
        ):
            resp = list_assessments(
                session=MagicMock(),
                auth_context=_auth_context(),
            )
        assert resp.success is True
        assert len(resp.data or []) == 1

        with patch(
            "app.api.routes.assessment.assessments.get_assessment_by_id",
            return_value=_assessment(),
        ), patch(
            "app.api.routes.assessment.assessments._build_assessment_public",
            return_value=public_stub,
        ):
            resp = get_assessment(
                10,
                session=MagicMock(),
                auth_context=_auth_context(),
            )
        assert resp.success is True

        with patch(
            "app.api.routes.assessment.assessments.get_assessment_by_id",
            side_effect=HTTPException(
                status_code=404, detail="Assessment 10 not found or not accessible"
            ),
        ):
            with pytest.raises(HTTPException, match="not found"):
                get_assessment(10, session=MagicMock(), auth_context=_auth_context())

    def test_list_and_get_runs(self) -> None:
        public_stub = MagicMock()
        with patch(
            "app.api.routes.assessment.runs.list_runs",
            return_value=[_run()],
        ), patch(
            "app.api.routes.assessment.runs._build_run_public",
            return_value=public_stub,
        ):
            resp = list_assessment_runs(
                session=MagicMock(), auth_context=_auth_context()
            )
        assert resp.success is True

        with patch(
            "app.api.routes.assessment.runs.get_run_by_id",
            return_value=_run(),
        ), patch(
            "app.api.routes.assessment.runs._build_run_public",
            return_value=public_stub,
        ):
            resp = get_assessment_run(
                22, session=MagicMock(), auth_context=_auth_context()
            )
        assert resp.success is True

        with patch(
            "app.api.routes.assessment.runs.get_run_by_id",
            side_effect=HTTPException(
                status_code=404, detail="Assessment run 22 not found or not accessible"
            ),
        ):
            with pytest.raises(HTTPException, match="not found"):
                get_assessment_run(
                    22, session=MagicMock(), auth_context=_auth_context()
                )


# ─── Export endpoints ────────────────────────────────────────────────────────


class TestExportRoutes:
    def test_export_assessment_results_delegates_to_util(self) -> None:
        """Parent export route delegates JSON/single-file/ZIP packaging to utils."""
        with patch(
            "app.api.routes.assessment.assessments.get_assessment_by_id",
            return_value=_assessment(),
        ), patch(
            "app.api.routes.assessment.assessments.get_assessment_runs_for_assessment",
            return_value=[_run()],
        ), patch(
            "app.api.routes.assessment.assessments.build_assessment_results_response",
            return_value="stub-response",
        ) as build:
            result = export_assessment_results(
                10,
                session=MagicMock(),
                auth_context=_auth_context(),
                export_format="json",
            )
        assert result == "stub-response"
        assert build.call_args.kwargs["export_format"] == "json"

    def test_export_assessment_run_results_json_and_file(self) -> None:
        run = _run()
        with patch(
            "app.api.routes.assessment.runs.get_run_by_id",
            return_value=run,
        ), patch(
            "app.api.routes.assessment.runs.get_assessment_by_id",
            return_value=_assessment(),
        ), patch(
            "app.api.routes.assessment.runs.load_export_rows_for_run",
            return_value=[_row()],
        ), patch(
            "app.api.routes.assessment.runs.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.api.routes.assessment.runs.build_json_export_rows",
            return_value=[{"x": 1}],
        ):
            json_resp = export_assessment_run_results(
                22,
                session=MagicMock(),
                auth_context=_auth_context(),
                export_format="json",
            )
        assert json_resp.success is True

        with patch(
            "app.api.routes.assessment.runs.get_run_by_id",
            return_value=run,
        ), patch(
            "app.api.routes.assessment.runs.get_assessment_by_id",
            return_value=_assessment(),
        ), patch(
            "app.api.routes.assessment.runs.load_export_rows_for_run",
            return_value=[_row()],
        ), patch(
            "app.api.routes.assessment.runs.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.api.routes.assessment.runs.build_export_response",
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
        with patch(
            "app.api.routes.assessment.assessments.get_assessment_by_id",
            side_effect=HTTPException(
                status_code=404, detail="Assessment 10 not found or not accessible"
            ),
        ):
            with pytest.raises(HTTPException, match="not found"):
                export_assessment_results(
                    10,
                    session=MagicMock(),
                    auth_context=_auth_context(),
                )
        with patch(
            "app.api.routes.assessment.runs.get_run_by_id",
            side_effect=HTTPException(
                status_code=404, detail="Assessment run 22 not found or not accessible"
            ),
        ):
            with pytest.raises(HTTPException, match="not found"):
                export_assessment_run_results(
                    22,
                    session=MagicMock(),
                    auth_context=_auth_context(),
                )


# ─── New: util-level test for the extracted ZIP/single-file logic ──────────


class TestBuildAssessmentResultsResponse:
    """Verify the extracted util builds the right shape for json / single-file / zip."""

    def test_json_returns_apiresponse(self) -> None:
        from app.services.assessment.utils.export import (
            build_assessment_results_response,
        )

        with patch(
            "app.services.assessment.utils.export.load_export_rows_for_run",
            return_value=[_row()],
        ), patch(
            "app.services.assessment.utils.export.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.services.assessment.utils.export.build_json_export_rows",
            return_value=[{"x": 1}],
        ):
            resp = build_assessment_results_response(
                session=MagicMock(),
                assessment=_assessment(),
                runs=[_run()],
                export_format="json",
            )
        assert resp.success is True

    def test_csv_multi_run_returns_zip(self) -> None:
        from app.services.assessment.utils.export import (
            build_assessment_results_response,
        )

        run1 = _run()
        run2 = _run()
        run2.id = 23
        run2.config_version = 2

        with patch(
            "app.services.assessment.utils.export.load_export_rows_for_run",
            side_effect=[[_row(execution_id=22)], [_row(execution_id=23)]],
        ), patch(
            "app.services.assessment.utils.export.sort_export_rows",
            side_effect=lambda rows: rows,
        ), patch(
            "app.services.assessment.utils.export.serialize_export_rows",
            return_value=(b"csv", "text/csv"),
        ), patch(
            "app.services.assessment.utils.export.generate_timestamped_filename",
            return_value="out.zip",
        ):
            resp = build_assessment_results_response(
                session=MagicMock(),
                assessment=_assessment(),
                runs=[run1, run2],
                export_format="csv",
            )
        assert isinstance(resp, StreamingResponse)
        assert resp.media_type == "application/zip"


# ─── Public builders — real logic (no builder mock) ─────────────────────────

CONFIG_UUID = UUID("00000000-0000-0000-0000-000000000001")


def _run_row(status: object, error: str | None = None) -> SimpleNamespace:
    """Child-run stub with real enum status for the count/stat helpers."""
    return SimpleNamespace(
        id=22,
        assessment_id=ASSESSMENT_UUID,
        config_id=CONFIG_UUID,
        config_version=1,
        status=status,
        total_items=3,
        error_message=error,
        updated_at=datetime(2024, 1, 1),
    )


def _dispatch_session(parent: object, dataset: object) -> MagicMock:
    """Session whose .get() returns the parent Assessment / dataset by model type."""
    from app.models.assessment import Assessment
    from app.models.evaluation import EvaluationDataset

    session = MagicMock()

    def _get(model: type, _key: object) -> object:
        if model is Assessment:
            return parent
        if model is EvaluationDataset:
            return dataset
        return None

    session.get.side_effect = _get
    return session


class TestBuildAssessmentPublic:
    """_build_assessment_public must surface derived counts/stats/dataset info."""

    def test_populates_derived_fields(self) -> None:
        from app.api.routes.assessment.assessments import _build_assessment_public
        from app.models.assessment import AssessmentStatus

        assessment = SimpleNamespace(
            id=ASSESSMENT_UUID,
            experiment_name="exp",
            dataset_id=7,
            status=AssessmentStatus.COMPLETED_WITH_ERRORS,
            organization_id=1,
            project_id=1,
            inserted_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )
        runs = [
            _run_row(AssessmentStatus.COMPLETED),
            _run_row(AssessmentStatus.FAILED, "boom"),
        ]
        session = MagicMock()
        session.get.return_value = SimpleNamespace(id=7, name="ds")

        with patch(
            "app.api.routes.assessment.assessments.get_assessment_runs_for_assessment",
            return_value=runs,
        ):
            public = _build_assessment_public(session, assessment)

        assert public.dataset_id == 7
        assert public.dataset_name == "ds"
        assert public.counts.total == 2
        assert public.counts.completed == 1
        assert public.counts.failed == 1
        assert len(public.run_stats) == 2
        assert public.error_message == "1 of 2 run(s) failed"


class TestBuildRunPublic:
    """_build_run_public must read runtime state from the execution bag, not columns."""

    def test_sources_from_bag_and_parent(self) -> None:
        from app.api.routes.assessment.runs import _build_run_public
        from app.models.assessment import AssessmentStatus, Stage, StageStatus

        run = SimpleNamespace(
            id=22,
            assessment_id=ASSESSMENT_UUID,
            config_id=CONFIG_UUID,
            config_version=1,
            status=AssessmentStatus.PROCESSING,
            total_items=10,
            error_message=None,
            execution={
                "stage": "L2_ASSESSMENT",
                "stage_status": "PROCESSING",
                "pipeline": {"steps": ["a"]},
                "prefilter_total_rows": 10,
                "prefilter_total_passed": 7,
                "prefilter_total_rejected": 3,
            },
            post_processing_config={"sort": "score"},
            inserted_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )
        parent = SimpleNamespace(
            experiment_name="exp",
            dataset_id=7,
            input={"prompt": "p"},
        )
        session = _dispatch_session(parent, SimpleNamespace(id=7, name="ds"))

        public = _build_run_public(session, run)

        assert public.stage == Stage.L2_ASSESSMENT
        assert public.stage_status == StageStatus.PROCESSING
        assert public.pipeline == {"steps": ["a"]}
        assert public.prefilter_total_rows == 10
        assert public.prefilter_total_passed == 7
        assert public.prefilter_total_rejected == 3
        assert public.input == {"prompt": "p"}
        assert public.post_processing_config == {"sort": "score"}
        assert public.experiment_name == "exp"
        assert public.dataset_id == 7
        assert public.dataset_name == "ds"

    def test_surfaces_cost_from_bag(self) -> None:
        from app.api.routes.assessment.runs import _build_run_public
        from app.models.assessment import AssessmentStatus

        run = SimpleNamespace(
            id=22,
            assessment_id=ASSESSMENT_UUID,
            config_id=CONFIG_UUID,
            config_version=1,
            status=AssessmentStatus.COMPLETED,
            total_items=10,
            error_message=None,
            execution={
                "cost": {
                    "pre_filter": {"topic_relevance": 0.5, "total": 0.5},
                    "assessment": 1.25,
                    "total": 1.75,
                    "currency": "USD",
                }
            },
            post_processing_config=None,
            inserted_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )
        parent = SimpleNamespace(experiment_name="exp", dataset_id=7, input=None)
        session = _dispatch_session(parent, SimpleNamespace(id=7, name="ds"))

        public = _build_run_public(session, run)

        assert public.cost is not None
        assert public.cost.pre_filter is not None
        assert public.cost.pre_filter.topic_relevance == 0.5
        assert public.cost.pre_filter.total == 0.5
        assert public.cost.assessment == 1.25
        assert public.cost.total == 1.75
        assert public.cost.currency == "USD"

    def test_empty_execution_bag_does_not_raise(self) -> None:
        """Regression: a run with no execution bag must not 500 (was AttributeError)."""
        from app.api.routes.assessment.runs import _build_run_public
        from app.models.assessment import AssessmentStatus

        run = SimpleNamespace(
            id=22,
            assessment_id=ASSESSMENT_UUID,
            config_id=CONFIG_UUID,
            config_version=1,
            status=AssessmentStatus.PENDING,
            total_items=0,
            error_message=None,
            execution=None,
            post_processing_config=None,
            inserted_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )
        parent = SimpleNamespace(experiment_name="exp", dataset_id=7, input=None)
        session = _dispatch_session(parent, SimpleNamespace(id=7, name="ds"))

        public = _build_run_public(session, run)

        assert public.stage is None
        assert public.stage_status is None
        assert public.pipeline is None
        assert public.prefilter_total_rows is None
        assert public.input is None
        assert public.post_processing_config is None
        assert public.cost is None
