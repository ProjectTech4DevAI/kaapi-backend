"""Tests for assessment/service.py orchestration behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.assessment.models import AssessmentConfigRef, AssessmentCreate
from app.assessment.service import start_assessment


def _make_request(provider_config_id: UUID) -> AssessmentCreate:
    return AssessmentCreate(
        experiment_name="exp-1",
        dataset_id=7,
        prompt_template="Answer: {question}",
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


class TestStartAssessment:
    def test_rejects_unsupported_provider(self) -> None:
        session = MagicMock()
        request = _make_request(UUID("00000000-0000-0000-0000-000000000001"))
        config_blob = SimpleNamespace(completion=SimpleNamespace(provider="google"))

        with patch(
            "app.assessment.service.get_dataset_by_id",
            return_value=_make_dataset(),
        ), patch(
            "app.assessment.service.resolve_evaluation_config",
            return_value=(config_blob, None),
        ), patch(
            "app.assessment.service.create_assessment"
        ) as create_assessment:
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

        with patch(
            "app.assessment.service.get_dataset_by_id", return_value=dataset
        ), patch(
            "app.assessment.service.resolve_evaluation_config",
            return_value=(config_blob, None),
        ), patch(
            "app.assessment.service.create_assessment",
            return_value=assessment,
        ), patch(
            "app.assessment.service.create_assessment_run",
            return_value=run,
        ), patch(
            "app.assessment.service.submit_assessment_batch",
            return_value=batch_job,
        ) as submit_batch, patch(
            "app.assessment.service.update_assessment_run_status",
            return_value=run,
        ), patch(
            "app.assessment.service.recompute_assessment_status"
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
        submit_batch.assert_called_once()
