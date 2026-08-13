"""Tests for the assessment API-client CRUD (app/crud/assessment/api.py) and the
API-client models (derive_method, assessment_blob validators). Real test session."""

import pytest
from pydantic import ValidationError

from app.crud.assessment import api
from app.models.assessment import (
    AssessmentMethod,
    AssessmentStatus,
    BatchInput,
    ResponseInput,
    derive_method,
)
from app.models.batch_job import BatchJob, BatchJobType
from app.models.config.assessment_blob import (
    DEFAULT_PREFILTER_MODEL,
    AssessmentConfigBlob,
    InputColumn,
    TopicRelevanceFilter,
)
from app.models.config.config import ConfigTag
from app.models.job import Job, JobType
from app.tests.utils.auth import get_user_test_auth_context
from app.tests.utils.test_data import create_test_config
from app.tests.utils.utils import random_lower_string


def _config(db, project_id):
    blob = AssessmentConfigBlob.model_validate(
        {
            "assessment": {
                "provider": "openai",
                "type": "text",
                "params": {
                    "model": "gpt-4o",
                    "input_schema": {"a": {"type": "text"}},
                },
            }
        }
    )
    return create_test_config(
        db,
        project_id=project_id,
        name=f"assess-{random_lower_string()}",
        config_blob=blob,
        tag=ConfigTag.ASSESSMENT,
    )


class TestApiCrudRoundTrip:
    def test_create_save_link_list_update(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _config(db, auth.project_id)

        assessment = api.create_assessment(
            session=db,
            method=AssessmentMethod.BATCH,
            input={"query": "q", "data": [{"a": "1"}]},
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        assert assessment.status == AssessmentStatus.PENDING

        execution = api.create_execution(
            session=db,
            assessment_id=assessment.id,
            config_id=config.id,
            config_version=1,
            total_items=1,
        )
        assert execution.status == AssessmentStatus.PENDING

        state = {
            "pipeline": [{"stage": "assessment", "kind": "ASSESSMENT"}],
            "stage": "assessment",
            "stage_status": "PENDING",
            "stage_batches": {},
            "stage_output_urls": {},
            "verdicts": {},
            "counters": {},
            "gate_passed": [True],
            "provider": "openai",
            "model": "gpt-4o",
            "input_schema": None,
            "callback_url": "https://hook/cb",
            "request_metadata": None,
        }
        api.save_execution_state(session=db, execution=execution, state=state)
        db.refresh(execution)
        assert execution.execution["provider"] == "openai"
        assert execution.execution["gate_passed"] == [True]

        job = BatchJob(
            provider="openai",
            job_type=BatchJobType.ASSESSMENT.value,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        api.set_execution_batch_job(
            session=db, execution=execution, batch_job_id=job.id
        )
        db.refresh(execution)
        assert execution.batch_job_id == job.id

        listed = api.list_executions(session=db, assessment_id=assessment.id)
        assert [e.id for e in listed] == [execution.id]

        api.update_status(session=db, obj=execution, status=AssessmentStatus.COMPLETED)
        api.update_status(session=db, obj=assessment, status=AssessmentStatus.COMPLETED)
        db.refresh(execution)
        db.refresh(assessment)
        assert execution.status == AssessmentStatus.COMPLETED
        assert assessment.status == AssessmentStatus.COMPLETED

    def test_set_assessment_job_links_job(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment = api.create_assessment(
            session=db,
            method=AssessmentMethod.RESPONSE,
            input={"query": "q"},
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        job = Job(job_type=JobType.ASSESSMENT, project_id=auth.project_id)
        db.add(job)
        db.commit()
        db.refresh(job)
        api.set_assessment_job(session=db, assessment=assessment, job_id=job.id)
        db.refresh(assessment)
        assert assessment.job_id == job.id

    def test_list_executions_orders_by_id(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _config(db, auth.project_id)
        assessment = api.create_assessment(
            session=db,
            method=AssessmentMethod.BATCH,
            input={"query": "q", "data": [{"a": "1"}]},
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        first = api.create_execution(
            session=db,
            assessment_id=assessment.id,
            config_id=config.id,
            config_version=1,
            total_items=1,
        )
        second = api.create_execution(
            session=db,
            assessment_id=assessment.id,
            config_id=config.id,
            config_version=1,
            total_items=1,
        )
        listed = api.list_executions(session=db, assessment_id=assessment.id)
        assert [e.id for e in listed] == [first.id, second.id]


class TestDeriveMethod:
    def test_response_input(self) -> None:
        assert (
            derive_method(ResponseInput(query="hi"), None) == AssessmentMethod.RESPONSE
        )

    def test_batch_input(self) -> None:
        assert (
            derive_method(BatchInput(query="q", data=[{"a": "1"}]), None)
            == AssessmentMethod.BATCH
        )

    def test_dataset_only_is_run(self) -> None:
        assert derive_method(None, 7) == AssessmentMethod.RUN

    def test_nothing_raises(self) -> None:
        with pytest.raises(ValueError, match="Provide inline"):
            derive_method(None, None)


class TestPreFilterValidators:
    def test_instructions_in_params_accepted(self) -> None:
        flt = TopicRelevanceFilter.model_validate(
            {
                "provider": "openai",
                "params": {"model": "gpt-4o", "instructions": "Is this on topic?"},
                "stop_on_fail": True,
            }
        )
        assert flt.params["instructions"] == "Is this on topic?"
        assert flt.stop_on_fail is True

    def test_missing_instructions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopicRelevanceFilter.model_validate(
                {"provider": "openai", "params": {"model": "gpt-4o"}}
            )

    def test_empty_instructions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopicRelevanceFilter.model_validate(
                {
                    "provider": "openai",
                    "params": {"model": "gpt-4o", "instructions": ""},
                }
            )

    def test_default_model_gets_effort_summary_and_drops_temperature(self) -> None:
        flt = TopicRelevanceFilter.model_validate(
            {
                "provider": "openai",
                "params": {
                    "model": DEFAULT_PREFILTER_MODEL,
                    "instructions": "on topic?",
                    "temperature": 0.7,
                },
            }
        )
        assert flt.params["effort"] == "high"
        assert flt.params["summary"] == "auto"
        assert "temperature" not in flt.params

    def test_non_default_model_keeps_temperature(self) -> None:
        flt = TopicRelevanceFilter.model_validate(
            {
                "provider": "openai",
                "params": {
                    "model": "gpt-4o",
                    "instructions": "on topic?",
                    "temperature": 0.7,
                },
            }
        )
        assert flt.params["temperature"] == 0.7
        assert "effort" not in flt.params


def _assessment_params(**extra) -> dict:
    params = {"model": "gpt-4o", "input_schema": {"a": {"type": "text"}}}
    params.update(extra)
    return {"assessment": {"provider": "openai", "type": "text", "params": params}}


class TestAssessmentCompletionConfigValidator:
    def test_temperature_dropped_when_not_user_set(self) -> None:
        blob = AssessmentConfigBlob.model_validate(_assessment_params())
        assert "temperature" not in blob.assessment.params

    def test_user_set_temperature_kept(self) -> None:
        blob = AssessmentConfigBlob.model_validate(_assessment_params(temperature=0.4))
        assert blob.assessment.params["temperature"] == 0.4


class TestInputSchemaValidators:
    def test_missing_input_schema_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AssessmentConfigBlob.model_validate(
                {
                    "assessment": {
                        "provider": "openai",
                        "type": "text",
                        "params": {"model": "gpt-4o"},
                    }
                }
            )

    def test_empty_input_schema_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AssessmentConfigBlob.model_validate(
                {
                    "assessment": {
                        "provider": "openai",
                        "type": "text",
                        "params": {"model": "gpt-4o", "input_schema": {}},
                    }
                }
            )

    def test_column_without_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AssessmentConfigBlob.model_validate(
                {
                    "assessment": {
                        "provider": "openai",
                        "type": "text",
                        "params": {
                            "model": "gpt-4o",
                            "input_schema": {"a": {"format": "url"}},
                        },
                    }
                }
            )

    def test_typed_columns_accepted(self) -> None:
        blob = AssessmentConfigBlob.model_validate(
            _assessment_params(
                input_schema={
                    "name": {"type": "text"},
                    "sheet": {"type": "pdf", "format": "url"},
                }
            )
        )
        assert set(blob.assessment.params["input_schema"]) == {"name", "sheet"}


class TestInputColumn:
    def test_type_required(self) -> None:
        with pytest.raises(ValidationError):
            InputColumn.model_validate({"format": "url"})

    def test_valid_type_accepted(self) -> None:
        col = InputColumn.model_validate({"type": "image", "format": "url"})
        assert col.type == "image"
        assert col.format == "url"
