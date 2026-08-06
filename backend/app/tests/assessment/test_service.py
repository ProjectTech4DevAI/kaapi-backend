"""Tests for assessment/service.py orchestration behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.crud.assessment import core as assessment_core
from app.crud.assessment.core import _read_exec
from app.models.assessment import (
    AssessmentConfigRef,
    AssessmentRunCreate,
    AssessmentStatus,
    InputBinding,
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

ASSESSMENT_ID = UUID("00000000-0000-0000-0000-000000000021")
CONFIG_ID = UUID("00000000-0000-0000-0000-000000000001")


def _make_request(provider_config_id: UUID) -> AssessmentRunCreate:
    return AssessmentRunCreate(
        experiment_name="exp-1",
        dataset_id=7,
        input_binding=InputBinding(
            prompt="Answer: {question}",
            text_columns=["question"],
            attachments=[],
        ),
        configs=[
            AssessmentConfigRef(id=provider_config_id, version=1),
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
    run.assessment_id = ASSESSMENT_ID
    run.config_id = CONFIG_ID
    run.config_version = 1
    run.status = AssessmentStatus.PROCESSING
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
        config_blob = SimpleNamespace(completion=SimpleNamespace(provider="sarvamai"))

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
        assessment.id = ASSESSMENT_ID
        run = _make_run()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(
                provider="google-aistudio", params={"model": "gemini"}
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

        # Google is an accepted provider — no rejection, one Celery task dispatched.
        assert response.num_configs == 1
        dispatch.delay.assert_called_once()
        assert dispatch.delay.call_args.kwargs["run_id"] == 11

    @pytest.mark.parametrize(
        "provider",
        [
            "openai",
            "openai-native",
            "google-aistudio",
            "google-aistudio-native",
            "anthropic",
            "anthropic-native",
        ],
    )
    def test_supported_batch_providers_are_accepted(self, provider: str) -> None:
        """Every provider in _SUPPORTED_BATCH_PROVIDERS must pass validation."""
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        dataset = _make_dataset()
        assessment = MagicMock()
        assessment.id = ASSESSMENT_ID
        run = _make_run()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(provider=provider, params={"model": "m"})
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

    def test_anthropic_provider_is_supported(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        dataset = _make_dataset()
        assessment = MagicMock()
        assessment.id = ASSESSMENT_ID
        run = _make_run()
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(
                provider="anthropic", params={"model": "claude-opus-4-8"}
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

        # Anthropic is an accepted provider — no rejection, one Celery task dispatched.
        assert response.num_configs == 1
        dispatch.delay.assert_called_once()
        assert dispatch.delay.call_args.kwargs["run_id"] == 11

    def test_defaults_missing_provider_to_openai(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        dataset = _make_dataset()
        assessment = MagicMock()
        assessment.id = ASSESSMENT_ID
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

        assert response.assessment_id == ASSESSMENT_ID
        assert response.num_configs == 1
        assert response.runs[0].run_id == 11
        # Binding now lives on the parent via create_assessment(input_binding=...),
        # not threaded through create_assessment_run.
        assert create_run.call_args.kwargs["assessment_id"] == ASSESSMENT_ID
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
        assessment.id = ASSESSMENT_ID
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
        # The binding now comes in as the input_binding arg (parent assessment.input),
        # not off the child run.
        binding = {"prompt": "p", "text_columns": ["q"], "attachments": []}

        with pytest.raises(HTTPException, match="No assessment runs"):
            _build_retry_request(
                experiment_name="exp", dataset_id=1, input_binding=binding, runs=[]
            )

        run = MagicMock()
        with pytest.raises(HTTPException, match="missing for retry"):
            _build_retry_request(
                experiment_name="exp", dataset_id=1, input_binding=None, runs=[run]
            )

        run2 = MagicMock()
        run2.id = 1
        run2.config_id = None
        run2.config_version = None
        with pytest.raises(HTTPException, match="Config reference is missing"):
            _build_retry_request(
                experiment_name="exp", dataset_id=1, input_binding=binding, runs=[run2]
            )

        run3 = MagicMock()
        run3.id = 2
        run3.config_id = CONFIG_ID
        run3.config_version = 1
        req = _build_retry_request(
            experiment_name="exp", dataset_id=1, input_binding=binding, runs=[run3]
        )
        assert req.experiment_name == "exp"
        assert req.input_binding.prompt == "p"
        assert len(req.configs) == 1

    def test_retry_assessment_wrappers(self) -> None:
        session = MagicMock()
        assessment = MagicMock()
        assessment.id = ASSESSMENT_ID
        assessment.experiment_name = "exp"
        assessment.dataset_id = 7
        assessment.input = {"prompt": "p", "text_columns": [], "attachments": []}
        run = MagicMock()
        run.assessment_id = ASSESSMENT_ID
        run.assessment = assessment
        run.config_id = CONFIG_ID
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
        run.assessment_id = ASSESSMENT_ID
        run.config_id = CONFIG_ID
        run.config_version = 1
        run.status = AssessmentStatus.FAILED
        run.execution = {
            "stage": stage,
            "stage_status": StageStatus.FAILED,
            "pipeline": {
                "stages": [
                    {"stage": Stage.PRE_FILTER_TOPIC_RELEVANCE, "order": 1},
                    {"stage": Stage.PRE_FILTER_DUPLICATE_DETECTION, "order": 2},
                    {"stage": Stage.L2_ASSESSMENT, "order": 3},
                ]
            },
        }
        run.assessment = SimpleNamespace(
            id=ASSESSMENT_ID, experiment_name="exp", dataset_id=7
        )
        return run

    def test_rejects_non_failed_run(self) -> None:
        run = self._failed_run(Stage.L2_ASSESSMENT)
        run.execution["stage_status"] = StageStatus.PROCESSING
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
            patch.object(assessment_core, "flag_modified"),
            patch("app.celery.tasks.job_execution.run_assessment_pipeline") as dispatch,
        ):
            resp = resume_assessment_run(session, run, 1, 1)

        # Same run, reset to PENDING at the same (failed) stage, re-dispatched.
        assert _read_exec(run).get("stage") == Stage.L2_ASSESSMENT
        assert _read_exec(run).get("stage_status") == StageStatus.PENDING
        assert run.status == AssessmentStatus.PROCESSING
        assert run.error_message is None
        dispatch.delay.assert_called_once()
        assert dispatch.delay.call_args.kwargs["run_id"] == 11
        assert resp.assessment_id == ASSESSMENT_ID
        assert resp.num_configs == 1
