"""Tests for the RESPONSE runner (app/services/assessment/api/response.py)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from gevent import Timeout

from app.crud.assessment import api
from app.models.assessment import AssessmentMethod, AssessmentStatus
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.services.assessment.api import response
from app.services.assessment.api.batch import build_pipeline
from app.tests.assessment.test_api_batch import _SessionCtx, _bag_for, _blob_dict
from app.tests.utils.auth import get_user_test_auth_context
from app.tests.utils.test_data import create_test_config
from app.tests.utils.utils import random_lower_string

_RUN_STAGE = "app.services.assessment.api.response._run_stage"
_DELIVER = "app.services.assessment.api.response.deliver"
_PROVIDER = "app.services.assessment.api.response.get_llm_provider"


class _Dumpable:
    """Stand-in for an SDK response object: ``model_dump`` returns the given dict."""

    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, **_):
        return self._payload


def _fake_client(provider: str, text: str = "ok", raise_exc: Exception | None = None):
    def _create(**_):
        if raise_exc:
            raise raise_exc
        if provider == "openai":
            return _Dumpable({"id": "r1", "output_text": text})
        if provider == "anthropic":
            return _Dumpable({"id": "m1", "content": [{"type": "text", "text": text}]})
        return _Dumpable({"candidates": [{"content": {"parts": [{"text": text}]}}]})

    if provider == "openai":
        return SimpleNamespace(responses=SimpleNamespace(create=_create))
    if provider == "anthropic":
        return SimpleNamespace(messages=SimpleNamespace(create=_create))
    genai = SimpleNamespace(models=SimpleNamespace(generate_content=_create))
    return SimpleNamespace(genai=genai) if provider == "google-gcp" else genai


def _blob(**kwargs) -> AssessmentConfigBlob:
    return AssessmentConfigBlob.model_validate(_blob_dict(**kwargs))


def _blob_with(*, pre_filter: bool, assessment: bool) -> AssessmentConfigBlob:
    raw = _blob_dict(topic_relevance=True if pre_filter else None)
    if not assessment:
        raw.pop("assessment")
    return AssessmentConfigBlob.model_validate(raw)


def _seed_response(db, auth, blob):
    config = create_test_config(
        db,
        project_id=auth.project_id,
        name=f"assess-{random_lower_string()}",
        config_blob=blob,
        tag=ConfigTag.ASSESSMENT,
    )
    assessment = api.create_assessment(
        session=db,
        method=AssessmentMethod.RESPONSE,
        input={"data": {"a": "one"}},
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )
    execution = api.create_execution(
        session=db,
        assessment_id=assessment.id,
        config_id=config.id,
        config_version=1,
        total_items=1,
    )
    api.save_execution_state(
        session=db,
        execution=execution,
        state=_bag_for(build_pipeline(blob), total=1),
    )
    api.update_status(session=db, obj=assessment, status=AssessmentStatus.PROCESSING)
    api.update_status(session=db, obj=execution, status=AssessmentStatus.PROCESSING)
    return assessment, execution


def _stage_output(
    stage: str, *, verdict: bool = True, error: str | None = None
) -> dict:
    if stage == "assessment":
        return {"output": '{"score": 1}', "error": None}
    out = f'{{"verdict": {str(verdict).lower()}, "reasoning": "r"}}'
    return {"output": None if error else out, "error": error}


def _run(db, execution_id: int, stage_output) -> list:
    delivered: list = []
    with (
        patch(_RUN_STAGE, side_effect=lambda **kw: stage_output(kw["stage"])),
        patch(_DELIVER, side_effect=lambda **kw: delivered.append(kw)),
        patch.object(response, "Session", lambda _: _SessionCtx(db)),
    ):
        response.run_response_assessment(execution_id=execution_id)
    return delivered


class TestRunStage:
    """One provider call per stage, built from the batch request builders."""

    @pytest.mark.parametrize(
        "provider", ["openai", "anthropic", "google-aistudio", "google-gcp"]
    )
    def test_each_provider_round_trips_through_parse_one(self, db, provider) -> None:
        auth = get_user_test_auth_context(db)
        raw = _blob_dict(provider=provider)
        raw["assessment"]["params"]["instructions"] = "Grade it."
        blob = AssessmentConfigBlob.model_validate(raw)
        with patch(
            _PROVIDER,
            return_value=SimpleNamespace(client=_fake_client(provider, text="graded")),
        ):
            out = response._run_stage(
                session=db,
                blob=blob,
                stage="assessment",
                row={"a": "one"},
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert out == {
            "output": "graded",
            "error": None,
            "usage": None,
            "response_id": {"openai": "r1", "anthropic": "m1"}.get(provider),
        }

    def test_provider_exception_becomes_the_row_error(self, db) -> None:
        auth = get_user_test_auth_context(db)
        with patch(
            _PROVIDER,
            return_value=SimpleNamespace(
                client=_fake_client("openai", raise_exc=RuntimeError("rate limited"))
            ),
        ):
            out = response._run_stage(
                session=db,
                blob=_blob(),
                stage="assessment",
                row={"a": "one"},
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert out["output"] is None
        assert out["error"] == "rate limited"


class TestRunResponseAssessment:
    def test_gate_then_assessment_completes_and_delivers(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob(topic_relevance=True))

        (call,) = _run(db, execution.id, _stage_output)

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.COMPLETED
        assert execution.execution["stage"] == "assessment"
        assert execution.execution["stage_status"] == "COMPLETED"
        assert execution.execution["counters"]["topic_relevance"]["passed"] == 1
        output = call["result"].output
        assert output.assessment == {"score": 1}
        assert output.pre_filter.topic_relevance.verdict is True
        assert call["failure_message"] is None

    def test_rejected_by_the_gate_skips_the_assessment(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob(topic_relevance=True))

        (call,) = _run(
            db, execution.id, lambda stage: _stage_output(stage, verdict=False)
        )

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.COMPLETED
        assert execution.execution["gate_passed"] == [False]
        assert execution.execution["stage"] == "topic_relevance"
        output = call["result"].output
        assert output.assessment is None
        assert output.pre_filter.topic_relevance.verdict is False

    def test_failed_pre_filter_call_fails_open_and_is_reported(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob(topic_relevance=True))

        def stage_output(stage):
            if stage == "topic_relevance":
                return _stage_output(stage, error="boom")
            return _stage_output(stage)

        (call,) = _run(db, execution.id, stage_output)

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.COMPLETED_WITH_ERRORS
        assert execution.error_message == "topic_relevance: boom"
        verdict = call["result"].output.pre_filter.topic_relevance
        assert verdict.verdict is True
        assert verdict.error == "boom"
        assert call["result"].output.assessment == {"score": 1}

    def test_failed_assessment_call_marks_the_run_failed(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob())

        (call,) = _run(
            db, execution.id, lambda stage: {"output": None, "error": "provider down"}
        )

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.FAILED
        assert execution.error_message == "assessment: provider down"
        assert execution.execution["stage_status"] == "FAILED"
        assert call["result"].error == "provider down"

    def test_unexpected_exception_is_a_failure_with_the_stage(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob())

        def explode(stage):
            raise RuntimeError("kaboom")

        (call,) = _run(db, execution.id, explode)

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.FAILED
        assert execution.error_message == "assessment: kaboom"
        assert execution.execution["error"] == "assessment: kaboom"
        assert call["result"] is None
        assert call["failure_message"] == "kaboom"

    def test_gevent_timeout_is_a_failure_with_a_callback(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob())

        def time_out(stage):
            raise Timeout(600)

        (call,) = _run(db, execution.id, time_out)

        db.refresh(parent)
        assert parent.status == AssessmentStatus.FAILED
        assert execution.error_message == "assessment: 600 seconds"
        assert call["result"] is None

    def test_missing_config_version_fails_before_any_stage(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob())
        execution.config_version = 99
        db.add(execution)
        db.commit()

        (call,) = _run(db, execution.id, _stage_output)

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.FAILED
        assert execution.execution["stage_status"] == "PENDING"
        assert call["result"] is None

    def test_terminal_run_is_not_rerun(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(db, auth, _blob())
        api.update_status(session=db, obj=execution, status=AssessmentStatus.COMPLETED)

        delivered = _run(db, execution.id, _stage_output)

        assert delivered == []

    def test_missing_run_is_a_no_op(self, db) -> None:
        delivered = _run(db, 10**9, _stage_output)

        assert delivered == []


class TestOptionalStages:
    @pytest.mark.parametrize(
        ("pre_filter", "assessment", "called", "stage"),
        [
            (True, False, ["topic_relevance"], "topic_relevance"),
            (False, True, ["assessment"], "assessment"),
            (True, True, ["topic_relevance", "assessment"], "assessment"),
        ],
        ids=["pre_filter_only", "assessment_only", "both"],
    )
    def test_runs_only_the_configured_stages(
        self, db, pre_filter, assessment, called, stage
    ) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(
            db, auth, _blob_with(pre_filter=pre_filter, assessment=assessment)
        )
        delivered = _run(db, execution.id, _stage_output)

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.COMPLETED
        assert execution.execution["stage"] == stage
        assert list(execution.execution.get("verdicts", {})) == (
            ["topic_relevance"] if pre_filter else []
        )
        (call,) = delivered
        output = call["result"].output
        assert (output.assessment is not None) is assessment
        assert (output.pre_filter is not None) is pre_filter
        assert sorted(execution.execution.get("counters", {})) == sorted(
            s for s in called if s != "assessment"
        )

    def test_pre_filter_only_rejected_row_keeps_its_verdict(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(
            db, auth, _blob_with(pre_filter=True, assessment=False)
        )
        (call,) = _run(
            db, execution.id, lambda stage: _stage_output(stage, verdict=False)
        )

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.COMPLETED
        assert execution.execution["gate_passed"] == [False]
        output = call["result"].output
        assert output.assessment is None
        assert output.pre_filter.topic_relevance.verdict is False

    def test_pre_filter_only_failed_call_is_reported(self, db) -> None:
        auth = get_user_test_auth_context(db)
        parent, execution = _seed_response(
            db, auth, _blob_with(pre_filter=True, assessment=False)
        )
        (call,) = _run(
            db, execution.id, lambda stage: _stage_output(stage, error="boom")
        )

        db.refresh(parent)
        db.refresh(execution)
        assert parent.status == AssessmentStatus.COMPLETED_WITH_ERRORS
        assert execution.error_message == "topic_relevance: boom"
        verdict = call["result"].output.pre_filter.topic_relevance
        assert verdict.error == "boom"
        assert verdict.verdict is True  # fails open, as in BATCH
