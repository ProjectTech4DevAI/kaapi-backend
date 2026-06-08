"""Tests for assessment/service.py orchestration behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.models.assessment import (
    AssessmentConfigRef,
    AssessmentCreate,
    Stage,
    StageStatus,
)
from app.models.config.config import ConfigTag
from app.services.assessment.service import (
    _build_retry_request,
    resume_assessment_run,
    retry_assessment,
    retry_assessment_run,
    start_assessment,
)


def _make_request(provider_config_id: UUID) -> AssessmentCreate:
    return AssessmentCreate(
        experiment_name="exp-1",
        dataset_id=7,
        prompt_template="Answer: {question}",
        system_instruction="Assess strictly",
        text_columns=["question"],
        attachments=[],
        configs=[
            AssessmentConfigRef(config_id=provider_config_id, config_version=1),
        ],
    )


def _make_dataset() -> MagicMock:
    dataset = MagicMock()
    dataset.id = 7
    dataset.name = "dataset-1"
    return dataset


def _make_run() -> MagicMock:
    run = MagicMock()
    run.id = 11
    run.assessment_id = 21
    run.config_id = UUID("00000000-0000-0000-0000-000000000001")
    run.config_version = 1
    run.status = "processing"
    return run


def _assessment_config_crud_patch():
    """Patch ConfigCrud so the bare tag-check in start_assessment short-circuits.

    Returns a config tagged for assessment use, which the check accepts.
    """
    crud = MagicMock()
    crud.read_one.return_value = SimpleNamespace(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        tag=ConfigTag.ASSESSMENT,
    )
    return patch("app.services.assessment.service.ConfigCrud", return_value=crud)


class TestStartAssessment:
    def test_dataset_not_found(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        with patch(
            "app.services.assessment.service.get_assessment_dataset_by_id",
            side_effect=HTTPException(
                status_code=404,
                detail="Dataset 7 not found or not accessible",
            ),
        ):
            with pytest.raises(HTTPException, match="not found"):
                start_assessment(
                    session=session,
                    request=request,
                    organization_id=1,
                    project_id=1,
                )

    def test_config_resolution_failure(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        with (
            patch(
                "app.services.assessment.service.get_assessment_dataset_by_id",
                return_value=_make_dataset(),
            ),
            patch(
                "app.services.assessment.service.resolve_evaluation_config",
                return_value=(None, "missing"),
            ),
            _assessment_config_crud_patch(),
        ):
            with pytest.raises(HTTPException, match="Failed to resolve config"):
                start_assessment(
                    session=session,
                    request=request,
                    organization_id=1,
                    project_id=1,
                )

    def test_rejects_unsupported_provider(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        config_blob = SimpleNamespace(completion=SimpleNamespace(provider="anthropic"))

        with (
            patch(
                "app.services.assessment.service.get_assessment_dataset_by_id",
                return_value=_make_dataset(),
            ),
            patch(
                "app.services.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch(
                "app.services.assessment.service.create_assessment"
            ) as create_assessment,
            _assessment_config_crud_patch(),
        ):
            with pytest.raises(
                HTTPException, match="not supported for batch assessment"
            ):
                start_assessment(
                    session=session,
                    request=request,
                    organization_id=1,
                    project_id=1,
                )
        create_assessment.assert_not_called()

    def test_google_provider_is_supported(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        dataset = _make_dataset()
        assessment = MagicMock()
        assessment.id = 21
        run = _make_run()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(provider="google", params={"model": "gemini"})
        )

        with (
            patch(
                "app.services.assessment.service.get_assessment_dataset_by_id",
                return_value=dataset,
            ),
            patch(
                "app.services.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch(
                "app.services.assessment.service.create_assessment",
                return_value=assessment,
            ),
            patch(
                "app.services.assessment.service.create_assessment_run",
                return_value=run,
            ),
            patch("app.celery.tasks.job_execution.run_assessment_pipeline") as dispatch,
            patch("app.services.assessment.service.recompute_assessment_status"),
            _assessment_config_crud_patch(),
        ):
            response = start_assessment(
                session=session,
                request=request,
                organization_id=1,
                project_id=1,
            )

        # Google is an accepted provider — no rejection, one Celery task dispatched.
        assert response.num_configs == 1
        dispatch.delay.assert_called_once()
        assert dispatch.delay.call_args.kwargs["run_id"] == 11

    def test_defaults_missing_provider_to_openai(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        dataset = _make_dataset()
        assessment = MagicMock()
        assessment.id = 21
        run = _make_run()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(provider=None, params={"model": "gpt-4.1-mini"})
        )

        with (
            patch(
                "app.services.assessment.service.get_assessment_dataset_by_id",
                return_value=dataset,
            ),
            patch(
                "app.services.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch(
                "app.services.assessment.service.create_assessment",
                return_value=assessment,
            ),
            patch(
                "app.services.assessment.service.create_assessment_run",
                return_value=run,
            ) as create_run,
            patch("app.celery.tasks.job_execution.run_assessment_pipeline") as dispatch,
            patch("app.services.assessment.service.recompute_assessment_status"),
            _assessment_config_crud_patch(),
        ):
            response = start_assessment(
                session=session,
                request=request,
                organization_id=1,
                project_id=1,
            )

        assert response.assessment_id == 21
        assert response.num_configs == 1
        assert response.runs[0].run_id == 11
        assessment_input = create_run.call_args.kwargs["assessment_input"]
        assert assessment_input["system_instruction"] == "Assess strictly"
        dispatch.delay.assert_called_once()

    def test_rejects_default_tagged_config(self) -> None:
        """Configs explicitly tagged 'default' must be rejected for assessment."""
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))

        crud = MagicMock()
        crud.read_one.return_value = SimpleNamespace(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            tag=ConfigTag.DEFAULT,
        )

        with (
            patch(
                "app.services.assessment.service.get_assessment_dataset_by_id",
                return_value=_make_dataset(),
            ),
            patch("app.services.assessment.service.ConfigCrud", return_value=crud),
            patch(
                "app.services.assessment.service.resolve_evaluation_config"
            ) as resolve,
        ):
            with pytest.raises(
                HTTPException,
                match="cannot be used for assessment",
            ):
                start_assessment(
                    session=session,
                    request=request,
                    organization_id=1,
                    project_id=1,
                )
        # Tag check must fire BEFORE config resolution.
        resolve.assert_not_called()

    def test_dispatches_one_celery_task_per_config(self) -> None:
        """Batch submission moved to the Celery task; start_assessment only
        creates runs and dispatches one task per resolved config."""
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        dataset = _make_dataset()
        assessment = MagicMock()
        assessment.id = 21
        run = _make_run()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(
                provider="openai", params={"model": "gpt-4.1-mini"}
            )
        )

        with (
            patch(
                "app.services.assessment.service.get_assessment_dataset_by_id",
                return_value=dataset,
            ),
            patch(
                "app.services.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch(
                "app.services.assessment.service.create_assessment",
                return_value=assessment,
            ),
            patch(
                "app.services.assessment.service.create_assessment_run",
                return_value=run,
            ),
            patch("app.celery.tasks.job_execution.run_assessment_pipeline") as dispatch,
            patch("app.services.assessment.service.recompute_assessment_status"),
            _assessment_config_crud_patch(),
        ):
            response = start_assessment(
                session=session,
                request=request,
                organization_id=1,
                project_id=1,
            )
        assert response.num_configs == 1
        dispatch.delay.assert_called_once()
        assert dispatch.delay.call_args.kwargs["run_id"] == 11
        assert dispatch.delay.call_args.kwargs["organization_id"] == 1
        assert dispatch.delay.call_args.kwargs["project_id"] == 1


class TestRetryHelpers:
    def test_build_retry_request_errors_and_success(self) -> None:
        with pytest.raises(HTTPException, match="No assessment runs"):
            _build_retry_request(experiment_name="exp", dataset_id=1, runs=[])

        run = MagicMock()
        run.input = None
        with pytest.raises(HTTPException, match="missing for retry"):
            _build_retry_request(experiment_name="exp", dataset_id=1, runs=[run])

        run2 = MagicMock()
        run2.id = 1
        run2.input = {"prompt_template": "p", "text_columns": ["q"], "attachments": []}
        run2.config_id = None
        run2.config_version = None
        with pytest.raises(HTTPException, match="Config reference is missing"):
            _build_retry_request(experiment_name="exp", dataset_id=1, runs=[run2])

        run3 = MagicMock()
        run3.id = 2
        run3.input = {
            "prompt_template": "p",
            "system_instruction": "sys",
            "text_columns": ["q"],
            "attachments": [],
            "output_schema": {"type": "object"},
        }
        run3.config_id = UUID("00000000-0000-0000-0000-000000000001")
        run3.config_version = 1
        req = _build_retry_request(experiment_name="exp", dataset_id=1, runs=[run3])
        assert req.experiment_name == "exp"
        assert req.system_instruction == "sys"
        assert len(req.configs) == 1

    def test_retry_assessment_wrappers(self) -> None:
        session = MagicMock()
        assessment = MagicMock()
        assessment.id = 21
        assessment.experiment_name = "exp"
        assessment.dataset_id = 7
        run = MagicMock()
        run.assessment_id = 21
        run.assessment = assessment
        run.input = {"prompt_template": "p", "text_columns": [], "attachments": []}
        run.config_id = UUID("00000000-0000-0000-0000-000000000001")
        run.config_version = 1

        result = SimpleNamespace(
            assessment_id=1,
            experiment_name="exp",
            dataset_id=7,
            dataset_name="ds",
            num_configs=1,
            runs=[],
        )

        with (
            patch(
                "app.services.assessment.service.get_assessment_runs_for_assessment",
                return_value=[run],
            ),
            patch(
                "app.services.assessment.service.start_assessment", return_value=result
            ),
        ):
            resp = retry_assessment(session, assessment, 1, 1)
        assert resp.assessment_id == 1

        with patch(
            "app.services.assessment.service.start_assessment", return_value=result
        ):
            resp2 = retry_assessment_run(session, run, 1, 1)
        assert resp2.assessment_id == 1


class TestResumeAssessmentRun:
    def _failed_run(self, stage: str) -> MagicMock:
        run = MagicMock()
        run.id = 11
        run.assessment_id = 21
        run.config_id = UUID("00000000-0000-0000-0000-000000000001")
        run.config_version = 1
        run.status = "failed"
        run.stage = stage
        run.stage_status = StageStatus.FAILED
        run.pipeline = {
            "stages": [
                {"stage": Stage.PRE_FILTER_TOPIC_RELEVANCE, "order": 1},
                {"stage": Stage.PRE_FILTER_DUPLICATE_DETECTION, "order": 2},
                {"stage": Stage.L2_ASSESSMENT, "order": 3},
            ]
        }
        run.assessment = SimpleNamespace(id=21, experiment_name="exp", dataset_id=7)
        return run

    def test_rejects_non_failed_run(self) -> None:
        run = self._failed_run(Stage.L2_ASSESSMENT)
        run.stage_status = StageStatus.PROCESSING
        with pytest.raises(HTTPException) as exc:
            resume_assessment_run(MagicMock(), run, 1, 1)
        assert exc.value.status_code == 400

    def test_rejects_stage_not_in_pipeline(self) -> None:
        run = self._failed_run(Stage.FAILED)
        with pytest.raises(HTTPException) as exc:
            resume_assessment_run(MagicMock(), run, 1, 1)
        assert exc.value.status_code == 400

    def test_resumes_in_place_from_failed_stage(self) -> None:
        run = self._failed_run(Stage.L2_ASSESSMENT)
        session = MagicMock()

        with (
            patch(
                "app.services.assessment.service.get_assessment_dataset_by_id",
                return_value=_make_dataset(),
            ),
            patch("app.services.assessment.service.recompute_assessment_status"),
            patch("app.celery.tasks.job_execution.run_assessment_pipeline") as dispatch,
        ):
            resp = resume_assessment_run(session, run, 1, 1)

        # Same run, reset to PENDING at the same (failed) stage, re-dispatched.
        assert run.stage == Stage.L2_ASSESSMENT
        assert run.stage_status == StageStatus.PENDING
        assert run.status == "processing"
        assert run.error_message is None
        dispatch.delay.assert_called_once()
        assert dispatch.delay.call_args.kwargs["run_id"] == 11
        assert resp.assessment_id == 21
        assert resp.num_configs == 1
