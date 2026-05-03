"""Tests for assessment/service.py orchestration behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.assessment.models import AssessmentConfigRef, AssessmentCreate
from app.assessment.service import (
    _build_retry_request,
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


def _untagged_config_crud_patch():
    """Patch ConfigCrud so the bare tag-check in start_assessment short-circuits.

    Returns a config whose ``tag`` is None — i.e. legacy/untagged — which the
    check accepts.
    """
    crud = MagicMock()
    crud.read_one.return_value = SimpleNamespace(
        id=UUID("00000000-0000-0000-0000-000000000001"), tag=None
    )
    return patch("app.assessment.service.ConfigCrud", return_value=crud)


class TestStartAssessment:
    def test_dataset_not_found(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        with patch("app.assessment.service.get_dataset_by_id", return_value=None):
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
                "app.assessment.service.get_dataset_by_id",
                return_value=_make_dataset(),
            ),
            patch(
                "app.assessment.service.resolve_evaluation_config",
                return_value=(None, "missing"),
            ),
            _untagged_config_crud_patch(),
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
                "app.assessment.service.get_dataset_by_id",
                return_value=_make_dataset(),
            ),
            patch(
                "app.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch("app.assessment.service.create_assessment") as create_assessment,
            _untagged_config_crud_patch(),
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
        batch_job = MagicMock()
        batch_job.id = 101
        batch_job.total_items = 3

        with (
            patch("app.assessment.service.get_dataset_by_id", return_value=dataset),
            patch(
                "app.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch(
                "app.assessment.service.create_assessment",
                return_value=assessment,
            ),
            patch(
                "app.assessment.service.create_assessment_run",
                return_value=run,
            ),
            patch(
                "app.assessment.service.submit_assessment_batch",
                return_value=batch_job,
            ) as submit_batch,
            patch(
                "app.assessment.service.update_assessment_run_status",
                return_value=run,
            ),
            patch("app.assessment.service.recompute_assessment_status"),
            _untagged_config_crud_patch(),
        ):
            response = start_assessment(
                session=session,
                request=request,
                organization_id=1,
                project_id=1,
            )

        assert response.num_configs == 1
        assert submit_batch.call_args.kwargs["config_blob"] is config_blob

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
        batch_job = MagicMock()
        batch_job.id = 101
        batch_job.total_items = 3

        with (
            patch("app.assessment.service.get_dataset_by_id", return_value=dataset),
            patch(
                "app.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch(
                "app.assessment.service.create_assessment",
                return_value=assessment,
            ),
            patch(
                "app.assessment.service.create_assessment_run",
                return_value=run,
            ) as create_run,
            patch(
                "app.assessment.service.submit_assessment_batch",
                return_value=batch_job,
            ) as submit_batch,
            patch(
                "app.assessment.service.update_assessment_run_status",
                return_value=run,
            ),
            patch("app.assessment.service.recompute_assessment_status"),
            _untagged_config_crud_patch(),
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
        assert (
            submit_batch.call_args.kwargs["assessment_input"]["system_instruction"]
            == "Assess strictly"
        )
        submit_batch.assert_called_once()

    def test_rejects_default_tagged_config(self) -> None:
        """Configs explicitly tagged 'default' must be rejected for assessment."""
        from app.models.config.config import ConfigTag

        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))

        crud = MagicMock()
        crud.read_one.return_value = SimpleNamespace(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            tag=ConfigTag.DEFAULT,
        )

        with (
            patch(
                "app.assessment.service.get_dataset_by_id",
                return_value=_make_dataset(),
            ),
            patch("app.assessment.service.ConfigCrud", return_value=crud),
            patch("app.assessment.service.resolve_evaluation_config") as resolve,
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

    def test_batch_submission_failure_marks_run_failed(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        dataset = _make_dataset()
        assessment = MagicMock()
        assessment.id = 21
        run = _make_run()
        run.status = "failed"
        config_blob = SimpleNamespace(
            completion=SimpleNamespace(
                provider="openai", params={"model": "gpt-4.1-mini"}
            )
        )

        with (
            patch("app.assessment.service.get_dataset_by_id", return_value=dataset),
            patch(
                "app.assessment.service.resolve_evaluation_config",
                return_value=(config_blob, None),
            ),
            patch(
                "app.assessment.service.create_assessment",
                return_value=assessment,
            ),
            patch(
                "app.assessment.service.create_assessment_run",
                return_value=run,
            ),
            patch(
                "app.assessment.service.submit_assessment_batch",
                side_effect=RuntimeError("submit failed"),
            ),
            patch(
                "app.assessment.service.update_assessment_run_status",
                return_value=run,
            ) as update_run,
            patch("app.assessment.service.recompute_assessment_status"),
            _untagged_config_crud_patch(),
        ):
            response = start_assessment(
                session=session,
                request=request,
                organization_id=1,
                project_id=1,
            )
        assert response.num_configs == 1
        assert update_run.called


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
                "app.assessment.service.get_assessment_runs_for_assessment",
                return_value=[run],
            ),
            patch("app.assessment.service.start_assessment", return_value=result),
        ):
            resp = retry_assessment(session, assessment, 1, 1)
        assert resp.assessment_id == 1

        with patch("app.assessment.service.start_assessment", return_value=result):
            resp2 = retry_assessment_run(session, run, 1, 1)
        assert resp2.assessment_id == 1
