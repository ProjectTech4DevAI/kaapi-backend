"""Tests for the BATCH API-client staged pipeline (app/services/assessment/api/batch.py).

Pure helpers are exercised directly; the DB-driven state machine
(run_batch_stage / _submit_stage / _poll_outcome / _finalize / _fail) runs against
the transactional ``db`` session with only external provider/webhook seams mocked.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.crud.assessment import api
from app.models.assessment import (
    Assessment,
    AssessmentAttachment,
    AssessmentMethod,
    AssessmentStatus,
    BatchInput,
    StageStatus,
)
from app.models.batch_job import BatchJob, BatchJobType
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.services.assessment.api import batch as batch_service
from app.services.assessment.api.batch import (
    ApiStage,
    StageKind,
    _build_batch_provider,
    _err_str,
    _fail,
    _finalize,
    _openai_output_text,
    _parse_one,
    _parse_verdict,
    _poll_outcome,
    _prefilter_for_stage,
    _record_stage,
    _row_index,
    _row_subset,
    _stage_kind,
    _stage_params,
    _stage_prompt,
    _stage_provider_model,
    _submit_provider_batch,
    _submit_stage,
    build_pipeline,
    build_rows,
    is_supported_provider,
    next_stage,
    parse_batch_results,
    run_batch_stage,
)
from app.services.assessment.api.results import build_result
from app.tests.utils.auth import get_user_test_auth_context
from app.tests.utils.test_data import create_test_config
from app.tests.utils.utils import random_lower_string

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
}


# Pre-filter criteria now live in params.instructions (mandatory), no top-level prompt/content.
TOPIC_CRITERIA = "Is this on topic?"


def _blob_dict(
    *,
    topic_relevance: bool | None = None,
    provider: str = "openai",
    model: str = "gpt-4o",
    input_schema: dict | None = None,
    submission: str = "assess {a}",
    topic_submission: str | None = None,
) -> dict:
    """Assessment config blob dict. ``topic_relevance`` is the filter's ``stop_on_fail``
    value, or None to omit the filter entirely. ``topic_submission`` sets the pre-filter's
    own optional prompt template."""
    pre_filters: dict = {}
    if topic_relevance is not None:
        params: dict = {"model": "gpt-4o", "instructions": TOPIC_CRITERIA}
        if topic_submission is not None:
            params["submission"] = topic_submission
        pre_filters["topic_relevance"] = {
            "provider": "openai",
            "params": params,
            "stop_on_fail": topic_relevance,
        }
    assessment_params: dict = {
        "model": model,
        "json_output_schema": OUTPUT_SCHEMA,
        "submission": submission,
    }
    blob: dict = {
        "input_schema": input_schema or {"a": {"type": "text"}},
        "assessment": {
            "provider": provider,
            "type": "text",
            "params": assessment_params,
        },
    }
    if pre_filters:
        blob["pre_filters"] = pre_filters
    return blob


def _blob(**kwargs) -> AssessmentConfigBlob:
    return AssessmentConfigBlob.model_validate(_blob_dict(**kwargs))


class TestIsSupportedProvider:
    def test_supported_and_unsupported(self) -> None:
        assert is_supported_provider("openai") is True
        assert is_supported_provider("google") is True
        assert is_supported_provider("anthropic") is True
        assert is_supported_provider("sarvamai") is False


class TestBuildPipeline:
    def test_none_prefilters_is_assessment_only(self) -> None:
        pipeline = build_pipeline(None)
        assert pipeline == [
            {"stage": ApiStage.ASSESSMENT.value, "kind": StageKind.ASSESSMENT.value}
        ]

    def test_gate_precedes_assessment(self) -> None:
        blob = _blob(topic_relevance=True)  # stop_on_fail=True -> GATE
        pipeline = build_pipeline(blob.pre_filters)
        assert [s["stage"] for s in pipeline] == [
            ApiStage.TOPIC_RELEVANCE.value,
            ApiStage.ASSESSMENT.value,
        ]
        assert [s["kind"] for s in pipeline] == [
            StageKind.GATE.value,
            StageKind.ASSESSMENT.value,
        ]

    def test_passthrough_precedes_assessment(self) -> None:
        blob = _blob(topic_relevance=False)  # stop_on_fail=False -> PASS_THROUGH
        pipeline = build_pipeline(blob.pre_filters)
        assert [s["stage"] for s in pipeline] == [
            ApiStage.TOPIC_RELEVANCE.value,
            ApiStage.ASSESSMENT.value,
        ]
        assert [s["kind"] for s in pipeline] == [
            StageKind.PASS_THROUGH.value,
            StageKind.ASSESSMENT.value,
        ]


class TestNextStageAndKind:
    def test_next_stage_advances_then_none(self) -> None:
        pipeline = build_pipeline(_blob(topic_relevance=True).pre_filters)
        assert (
            next_stage(pipeline, ApiStage.TOPIC_RELEVANCE.value)
            == ApiStage.ASSESSMENT.value
        )
        assert next_stage(pipeline, ApiStage.ASSESSMENT.value) is None

    def test_stage_kind_lookup_and_missing_raises(self) -> None:
        pipeline = build_pipeline(_blob(topic_relevance=True).pre_filters)
        assert (
            _stage_kind(pipeline, ApiStage.TOPIC_RELEVANCE.value)
            == StageKind.GATE.value
        )
        with pytest.raises(ValueError, match="not in pipeline"):
            _stage_kind(pipeline, "nonexistent")


class TestBuildRows:
    def test_text_only_rows(self) -> None:
        batch_input = BatchInput(data=[{"a": "1", "b": "2"}])
        rows, text_columns, attachments = build_rows(batch_input)
        assert rows == [{"a": "1", "b": "2"}]
        assert set(text_columns) == {"a", "b"}
        assert attachments == []

    def test_attachment_column_is_split_out(self) -> None:
        batch_input = BatchInput(data=[{"text": "hi", "img": "https://x/a.png"}])
        input_columns = {"img": {"type": "image", "format": "url"}}
        rows, text_columns, attachments = build_rows(batch_input, input_columns)
        assert text_columns == ["text"]
        assert len(attachments) == 1
        assert isinstance(attachments[0], AssessmentAttachment)
        assert attachments[0].column == "img"
        assert attachments[0].type == "image"

    def test_base64_attachment_rejected(self) -> None:
        batch_input = BatchInput(data=[{"img": "data"}])
        input_columns = {"img": {"type": "image", "format": "base64"}}
        with pytest.raises(ValueError, match="url-format"):
            build_rows(batch_input, input_columns)

    def test_empty_submissions(self) -> None:
        batch_input = BatchInput.model_construct(data=[])
        rows, text_columns, attachments = build_rows(batch_input)
        assert rows == []
        assert text_columns == []


class TestStagePrompt:
    def test_assessment_uses_config_submission(self) -> None:
        blob = _blob(submission="assess {a} now")
        assert _stage_prompt(blob, ApiStage.ASSESSMENT.value) == "assess {a} now"

    def test_prefilter_returns_its_own_submission(self) -> None:
        blob = _blob(topic_relevance=True, topic_submission="rank {a}")
        assert _stage_prompt(blob, ApiStage.TOPIC_RELEVANCE.value) == "rank {a}"

    def test_prefilter_without_submission_is_none(self) -> None:
        blob = _blob(topic_relevance=True)
        assert _stage_prompt(blob, ApiStage.TOPIC_RELEVANCE.value) is None


class TestPrefilterForStage:
    def test_returns_configured_filter(self) -> None:
        blob = _blob(topic_relevance=True)
        tr = _prefilter_for_stage(blob, ApiStage.TOPIC_RELEVANCE.value)
        assert tr.params["instructions"] == TOPIC_CRITERIA

    def test_missing_filter_raises(self) -> None:
        blob = _blob()  # no pre-filters
        with pytest.raises(ValueError, match="No pre-filter configured"):
            _prefilter_for_stage(blob, ApiStage.TOPIC_RELEVANCE.value)


class TestStageParams:
    def test_assessment_renames_output_schema_and_drops_input_schema(self) -> None:
        blob = _blob(input_schema={"a": {"type": "text"}})
        params = _stage_params(blob, ApiStage.ASSESSMENT.value)
        assert params["output_schema"] == OUTPUT_SCHEMA
        assert "json_output_schema" not in params
        assert "input_schema" not in params

    def test_prefilter_layers_verdict_schema_and_appends_gate_directive(self) -> None:
        blob = _blob(topic_relevance=True)
        params = _stage_params(blob, ApiStage.TOPIC_RELEVANCE.value)
        assert params["output_schema"] == batch_service.PREFILTER_VERDICT_SCHEMA
        assert (
            params["instructions"]
            == f"{TOPIC_CRITERIA}\n\n{batch_service._PREFILTER_INSTRUCTION}"
        )

    def test_submission_is_not_a_provider_param(self) -> None:
        # submission is a prompt template, popped from both branches' provider params.
        blob = _blob(
            submission="assess {a}",
            topic_relevance=True,
            topic_submission="rank {a}",
        )
        assert "submission" not in _stage_params(blob, ApiStage.ASSESSMENT.value)
        assert "submission" not in _stage_params(blob, ApiStage.TOPIC_RELEVANCE.value)


class TestStageProviderModel:
    def test_assessment_uses_config_provider_model(self) -> None:
        blob = _blob(provider="anthropic", model="claude-sonnet-4-6")
        assert _stage_provider_model(blob, ApiStage.ASSESSMENT.value) == (
            "anthropic",
            "claude-sonnet-4-6",
        )

    def test_prefilter_uses_own_provider_model(self) -> None:
        blob = _blob(topic_relevance=True)
        assert _stage_provider_model(blob, ApiStage.TOPIC_RELEVANCE.value) == (
            "openai",
            "gpt-4o",
        )


class TestRowIndex:
    def test_valid_and_invalid_keys(self) -> None:
        assert _row_index("row_5") == 5
        assert _row_index("row_0") == 0
        assert _row_index("item_3") is None
        assert _row_index("row_abc") is None
        assert _row_index(None) is None


class TestErrStr:
    def test_dict_with_message(self) -> None:
        assert _err_str({"message": "boom"}) == "boom"

    def test_dict_without_message_stringifies(self) -> None:
        assert _err_str({"code": 500}) == "{'code': 500}"

    def test_plain_value(self) -> None:
        assert _err_str("oops") == "oops"


class TestOpenAIOutputText:
    def test_plain_string(self) -> None:
        assert _openai_output_text("hello") == "hello"

    def test_message_list_concatenated(self) -> None:
        output = [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "foo"},
                    {"type": "output_text", "text": "bar"},
                    {"type": "refusal", "text": "ignored"},
                ],
            }
        ]
        assert _openai_output_text(output) == "foobar"

    def test_non_message_items_ignored(self) -> None:
        assert _openai_output_text([{"type": "reasoning"}]) == ""


class TestParseOne:
    def test_top_level_error(self) -> None:
        result = _parse_one({"error": {"message": "bad"}}, "openai")
        assert result["output"] is None
        assert result["error"] == "bad"

    def test_openai_success(self) -> None:
        result = _parse_one(
            {
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": "answer",
                        "usage": {"input_tokens": 1},
                        "id": "resp_1",
                    },
                }
            },
            "openai",
        )
        assert result["output"] == "answer"
        assert result["error"] is None
        assert result["response_id"] == "resp_1"

    def test_openai_http_error_status(self) -> None:
        result = _parse_one(
            {
                "response": {
                    "status_code": 500,
                    "body": {"error": {"message": "server error"}, "id": "resp_2"},
                }
            },
            "openai",
        )
        assert result["output"] is None
        assert result["error"] == "server error"
        assert result["response_id"] == "resp_2"

    def test_openai_empty_output(self) -> None:
        result = _parse_one({"response": {"status_code": 200, "body": {}}}, "openai")
        assert result["output"] is None
        assert result["error"] == "Empty response output"

    def test_anthropic_success(self) -> None:
        result = _parse_one(
            {
                "response": {
                    "content": [
                        {"type": "text", "text": "hi "},
                        {"type": "text", "text": "there"},
                        {"type": "tool_use"},
                    ],
                    "usage": {"input_tokens": 2},
                    "id": "msg_1",
                }
            },
            "anthropic",
        )
        assert result["output"] == "hi there"
        assert result["response_id"] == "msg_1"

    def test_anthropic_empty(self) -> None:
        result = _parse_one({"response": {"content": []}}, "anthropic")
        assert result["output"] is None
        assert result["error"] == "Empty response"

    def test_google_success(self) -> None:
        with patch(
            "app.services.assessment.api.batch.extract_text_from_response_dict",
            return_value="gemini text",
        ):
            result = _parse_one({"response": {"candidates": []}}, "google")
        assert result["output"] == "gemini text"

    def test_google_empty_response(self) -> None:
        result = _parse_one({"response": None}, "google")
        assert result["output"] is None
        assert result["error"] == "Empty response"

    def test_google_gcp_parses_like_google(self) -> None:
        with patch(
            "app.services.assessment.api.batch.extract_text_from_response_dict",
            return_value="vertex text",
        ):
            result = _parse_one({"response": {"candidates": []}}, "google-gcp")
        assert result["output"] == "vertex text"

    def test_google_aistudio_parses_like_google(self) -> None:
        with patch(
            "app.services.assessment.api.batch.extract_text_from_response_dict",
            return_value="aistudio text",
        ):
            result = _parse_one({"response": {"candidates": []}}, "google-aistudio")
        assert result["output"] == "aistudio text"
        assert result["error"] is None

    def test_unknown_provider(self) -> None:
        result = _parse_one({"response": {}}, "cohere")
        assert result["output"] is None
        assert result["error"] == "Unknown provider cohere"


class TestParseBatchResults:
    def test_indexes_by_row_key_and_skips_bad_keys(self) -> None:
        raw = [
            {"key": "row_0", "response": None},
            {"custom_id": "row_2", "response": None},
            {"key": "not_a_row", "response": None},
        ]
        parsed = parse_batch_results(raw, "google")
        assert set(parsed.keys()) == {0, 2}


class TestParseVerdict:
    def test_empty_output_fails_open(self) -> None:
        assert _parse_verdict(None) == {"verdict": True, "reasoning": ""}

    def test_valid_json(self) -> None:
        out = json.dumps({"verdict": False, "reasoning": "off topic"})
        assert _parse_verdict(out) == {"verdict": False, "reasoning": "off topic"}

    def test_unparseable_fails_open(self) -> None:
        assert _parse_verdict("not json {") == {"verdict": True, "reasoning": ""}

    def test_missing_verdict_key_defaults_true(self) -> None:
        assert _parse_verdict(json.dumps({"reasoning": "x"})) == {
            "verdict": True,
            "reasoning": "x",
        }


class TestRecordStage:
    def _bag(self, total: int) -> dict:
        return {"gate_passed": [True] * total, "verdicts": {}, "counters": {}}

    def test_assessment_kind_is_noop(self) -> None:
        bag = self._bag(2)
        _record_stage(bag, ApiStage.ASSESSMENT.value, StageKind.ASSESSMENT.value, {})
        assert bag["verdicts"] == {}

    def test_gate_marks_failed_rows(self) -> None:
        bag = self._bag(2)
        parsed = {
            0: {"output": json.dumps({"verdict": True, "reasoning": ""})},
            1: {"output": json.dumps({"verdict": False, "reasoning": "no"})},
        }
        _record_stage(bag, ApiStage.TOPIC_RELEVANCE.value, StageKind.GATE.value, parsed)
        assert bag["gate_passed"] == [True, False]
        counters = bag["counters"][ApiStage.TOPIC_RELEVANCE.value]
        assert counters == {"total": 2, "passed": 1, "rejected": 1}

    def test_pass_through_records_but_does_not_gate(self) -> None:
        bag = self._bag(2)
        parsed = {
            0: {"output": json.dumps({"verdict": False, "reasoning": "dup"})},
            1: {"output": json.dumps({"verdict": True, "reasoning": ""})},
        }
        _record_stage(
            bag,
            ApiStage.TOPIC_RELEVANCE.value,
            StageKind.PASS_THROUGH.value,
            parsed,
        )
        assert bag["gate_passed"] == [True, True]
        assert bag["counters"][ApiStage.TOPIC_RELEVANCE.value]["rejected"] == 1


class TestRowSubset:
    def test_assessment_only_gate_passed(self) -> None:
        bag = {"gate_passed": [True, False, True]}
        assert _row_subset(
            bag, ApiStage.ASSESSMENT.value, StageKind.ASSESSMENT.value, 3
        ) == [0, 2]

    def test_prefilter_all_rows(self) -> None:
        bag = {"gate_passed": [True, False, True]}
        assert _row_subset(
            bag, ApiStage.TOPIC_RELEVANCE.value, StageKind.GATE.value, 3
        ) == [0, 1, 2]


class _SessionCtx:
    """Wrap the transactional test session so ``with Session(engine)`` in the
    service reuses it (and never closes it) instead of opening a real connection."""

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *_exc):
        return False


def _make_batch_job(db, *, org_id, project_id, **kwargs) -> BatchJob:
    job = BatchJob(
        provider="openai",
        job_type=BatchJobType.ASSESSMENT.value,
        organization_id=org_id,
        project_id=project_id,
        total_items=kwargs.pop("total_items", 1),
        **kwargs,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _seed_assessment(db, *, org_id, project_id, config_id, bag, data, status=None):
    batch_input = BatchInput(data=data)
    assessment = api.create_assessment(
        session=db,
        method=AssessmentMethod.BATCH,
        input=batch_input.model_dump(mode="json"),
        organization_id=org_id,
        project_id=project_id,
    )
    execution = api.create_execution(
        session=db,
        assessment_id=assessment.id,
        config_id=config_id,
        config_version=1,
        total_items=len(data),
    )
    api.save_execution_state(session=db, execution=execution, state=bag)
    if status is not None:
        api.update_status(session=db, obj=assessment, status=status)
        api.update_status(session=db, obj=execution, status=status)
    return assessment, execution


def _make_config_id(db, project_id):
    config = create_test_config(
        db,
        project_id=project_id,
        name=f"assess-{random_lower_string()}",
        config_blob=_blob(),
        tag=ConfigTag.ASSESSMENT,
    )
    return config.id


def _bag_for(pipeline, *, total, callback="https://hook.example/cb", provider="openai"):
    return {
        "pipeline": pipeline,
        "stage": pipeline[0]["stage"],
        "stage_status": StageStatus.PENDING.value,
        "stage_batches": {},
        "stage_output_urls": {},
        "verdicts": {},
        "counters": {},
        "gate_passed": [True] * total,
        "provider": provider,
        "model": "gpt-4o",
        "input_schema": None,
        "callback_url": callback,
        "request_metadata": {"ref": "x"},
    }


class TestRunBatchStageFullWalk:
    def test_gate_then_assessment_to_completed_callback(self, db) -> None:
        auth = get_user_test_auth_context(db)
        org_id, project_id = auth.organization_id, auth.project_id
        blob = _blob(topic_relevance=True)  # gate + assessment pipeline
        config = create_test_config(
            db,
            project_id=project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(blob.pre_filters)
        bag = _bag_for(pipeline, total=2)
        assessment, execution = _seed_assessment(
            db,
            org_id=org_id,
            project_id=project_id,
            config_id=config.id,
            bag=bag,
            data=[{"a": "one"}, {"a": "two"}],
            status=AssessmentStatus.PROCESSING,
        )

        def fake_start(**_kwargs):
            return _make_batch_job(
                db,
                org_id=org_id,
                project_id=project_id,
                provider_status="completed",
                provider_output_file_id="file_out",
                raw_output_url="s3://bucket/out.jsonl",
            )

        gate_results = [
            {
                "custom_id": "row_0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps({"verdict": True, "reasoning": "ok"})
                    },
                },
            },
            {
                "custom_id": "row_1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps(
                            {"verdict": False, "reasoning": "off"}
                        )
                    },
                },
            },
        ]

        from app.models.assessment import (
            AssessmentBatchResult,
            AssessmentCounts,
        )

        clean_result = AssessmentBatchResult(
            total_items=2, counts=AssessmentCounts(assessed=1, filtered=1), items=[]
        )
        delivered: list = []

        def call_tick():
            return run_batch_stage(
                execution_id=execution.id,
                organization_id=org_id,
                project_id=project_id,
            )

        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch(
                "app.services.assessment.api.batch.start_batch_job",
                side_effect=fake_start,
            ),
            patch(
                "app.services.assessment.api.batch.poll_batch_status",
                return_value={"request_counts": {"completed": 2, "failed": 0}},
            ),
            patch(
                "app.services.assessment.api.batch.process_completed_batch",
                return_value=(gate_results, {}),
            ),
            patch(
                "app.services.assessment.api.batch.get_openai_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=clean_result,
            ) as build_result,
            patch(
                "app.services.assessment.api.callbacks.deliver",
                side_effect=lambda **kw: delivered.append(kw) or True,
            ),
        ):
            first = call_tick()  # PENDING -> submit gate
            assert first == {"requeue": True}
            ticks = [first]
            for _ in range(5):
                r = call_tick()
                ticks.append(r)
                if not r["requeue"]:
                    break

        assert ticks[-1] == {"requeue": False}
        db.refresh(execution)
        db.refresh(assessment)
        assert execution.status == AssessmentStatus.COMPLETED
        assert assessment.status == AssessmentStatus.COMPLETED
        bag_after = execution.execution
        assert bag_after["gate_passed"] == [True, False]
        tr_verdicts = bag_after["verdicts"][ApiStage.TOPIC_RELEVANCE.value]
        assert tr_verdicts["0"]["verdict"] is True
        assert tr_verdicts["1"]["verdict"] is False
        assert build_result.called
        assert len(delivered) == 1


class TestRunBatchStageEarlyReturns:
    def _tick(self, db, execution_id, org_id, project_id):
        with patch(
            "app.services.assessment.api.batch.Session",
            lambda _e: _SessionCtx(db),
        ):
            return run_batch_stage(
                execution_id=execution_id,
                organization_id=org_id,
                project_id=project_id,
            )

    def test_execution_not_found(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assert self._tick(db, 99999999, auth.organization_id, auth.project_id) == {
            "requeue": False
        }

    def test_terminal_status_returns_no_requeue(self, db) -> None:
        auth = get_user_test_auth_context(db)
        blob = _blob()
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(None)
        _, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=_bag_for(pipeline, total=1),
            data=[{"a": "1"}],
            status=AssessmentStatus.COMPLETED,
        )
        assert self._tick(db, execution.id, auth.organization_id, auth.project_id) == {
            "requeue": False
        }

    def test_uninitialised_bag_returns_no_requeue(self, db) -> None:
        auth = get_user_test_auth_context(db)
        blob = _blob()
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        _, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag={"stage": None, "pipeline": []},
            data=[{"a": "1"}],
        )
        assert self._tick(db, execution.id, auth.organization_id, auth.project_id) == {
            "requeue": False
        }

    def test_processing_batch_missing_fails(self, db) -> None:
        auth = get_user_test_auth_context(db)
        blob = _blob()
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(None)
        bag = _bag_for(pipeline, total=1, callback="")
        bag["stage_status"] = StageStatus.PROCESSING.value
        bag["stage_batches"] = {}  # no batch job recorded for the stage
        assessment, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=bag,
            data=[{"a": "1"}],
            status=AssessmentStatus.PROCESSING,
        )
        assert self._tick(db, execution.id, auth.organization_id, auth.project_id) == {
            "requeue": False
        }
        db.refresh(execution)
        assert execution.status == AssessmentStatus.FAILED
        assert "batch not found" in execution.error_message


class TestPollOutcome:
    def _provider(self):
        return MagicMock()

    def test_processing_when_status_pending(self, db) -> None:
        auth = get_user_test_auth_context(db)
        job = _make_batch_job(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            provider_status="in_progress",
        )
        with patch(
            "app.services.assessment.api.batch.poll_batch_status",
            return_value={},
        ):
            outcome, results = _poll_outcome(db, self._provider(), job)
        assert outcome == "processing"
        assert results is None

    def test_failed_status(self, db) -> None:
        auth = get_user_test_auth_context(db)
        job = _make_batch_job(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            provider_status="failed",
        )
        with patch(
            "app.services.assessment.api.batch.poll_batch_status",
            return_value={},
        ):
            outcome, _ = _poll_outcome(db, self._provider(), job)
        assert outcome == "failed"

    def test_success_but_all_failed_counts(self, db) -> None:
        auth = get_user_test_auth_context(db)
        job = _make_batch_job(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            provider_status="completed",
            provider_output_file_id="f",
        )
        with patch(
            "app.services.assessment.api.batch.poll_batch_status",
            return_value={"request_counts": {"completed": 0, "failed": 3}},
        ):
            outcome, _ = _poll_outcome(db, self._provider(), job)
        assert outcome == "failed"

    def test_success_output_not_ready(self, db) -> None:
        auth = get_user_test_auth_context(db)
        job = _make_batch_job(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            provider_status="completed",
        )  # no provider_output_file_id yet
        with patch(
            "app.services.assessment.api.batch.poll_batch_status",
            return_value={"request_counts": {"completed": 1}},
        ):
            outcome, _ = _poll_outcome(db, self._provider(), job)
        assert outcome == "processing"


class TestBuildBatchProvider:
    def test_openai(self, db) -> None:
        auth = get_user_test_auth_context(db)
        with patch(
            "app.services.assessment.api.batch.get_openai_client",
            return_value=MagicMock(),
        ):
            provider = _build_batch_provider(
                session=db,
                provider_name="openai",
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert provider is not None

    def test_google(self, db) -> None:
        auth = get_user_test_auth_context(db)
        gemini = MagicMock()
        gemini.client = MagicMock()
        with patch("app.services.assessment.api.batch.GeminiClient") as gemini_cls:
            gemini_cls.from_credentials.return_value = gemini
            provider = _build_batch_provider(
                session=db,
                provider_name="google",
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert provider is not None

    def test_google_gcp_routes_to_vertex(self, db) -> None:
        auth = get_user_test_auth_context(db)
        sentinel = MagicMock()
        with (
            patch(
                "app.services.assessment.api.batch.get_provider_credential",
                return_value={"gcs_bucket": "b", "sa_key": {}},
            ),
            patch(
                "app.services.assessment.api.batch.GoogleGCPBatchProvider.from_credentials",
                return_value=sentinel,
            ) as vertex_from_cred,
        ):
            provider = _build_batch_provider(
                session=db,
                provider_name="google-gcp",
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert provider is sentinel
        vertex_from_cred.assert_called_once()

    def test_google_gcp_missing_credential_raises_404(self, db) -> None:
        auth = get_user_test_auth_context(db)
        with patch(
            "app.services.assessment.api.batch.get_provider_credential",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc:
                _build_batch_provider(
                    session=db,
                    provider_name="google-gcp",
                    organization_id=auth.organization_id,
                    project_id=auth.project_id,
                )
        assert exc.value.status_code == 404

    def test_anthropic(self, db) -> None:
        auth = get_user_test_auth_context(db)
        with patch(
            "app.services.assessment.api.batch.get_anthropic_client",
            return_value=MagicMock(),
        ):
            provider = _build_batch_provider(
                session=db,
                provider_name="anthropic",
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert provider is not None

    def test_unsupported_raises(self, db) -> None:
        with pytest.raises(ValueError, match="Unsupported provider"):
            _build_batch_provider(
                session=db,
                provider_name="cohere",
                organization_id=1,
                project_id=1,
            )


class TestSubmitProviderBatch:
    def _kwargs(self, db, auth, provider_name, params):
        return {
            "session": db,
            "provider_name": provider_name,
            "model": "m",
            "rows": [{"a": "one"}],
            "text_columns": ["a"],
            "attachments": [],
            "prompt": "do it",
            "params": params,
            "row_indices": [0],
            "organization_id": auth.organization_id,
            "project_id": auth.project_id,
            "description": "assessment-1-assessment",
        }

    def test_google_branch(self, db) -> None:
        auth = get_user_test_auth_context(db)
        gemini = MagicMock()
        gemini.client = MagicMock()
        job = _make_batch_job(
            db, org_id=auth.organization_id, project_id=auth.project_id
        )
        with (
            patch("app.services.assessment.api.batch.GeminiClient") as gemini_cls,
            patch(
                "app.services.assessment.api.batch.start_batch_job",
                return_value=job,
            ) as start,
        ):
            gemini_cls.from_credentials.return_value = gemini
            result = _submit_provider_batch(
                **self._kwargs(db, auth, "google", {"model": "gemini-2.5-pro"})
            )
        assert result.id == job.id
        assert start.call_args.kwargs["provider_name"] == "google"

    def test_google_gcp_branch_routes_to_vertex(self, db) -> None:
        auth = get_user_test_auth_context(db)
        job = _make_batch_job(
            db, org_id=auth.organization_id, project_id=auth.project_id
        )
        with (
            patch(
                "app.services.assessment.api.batch.get_provider_credential",
                return_value={"gcs_bucket": "b", "sa_key": {}},
            ),
            patch(
                "app.services.assessment.api.batch.GoogleGCPBatchProvider.from_credentials",
                return_value=MagicMock(),
            ) as vertex_from_cred,
            patch(
                "app.services.assessment.api.batch.start_batch_job",
                return_value=job,
            ) as start,
        ):
            _submit_provider_batch(
                **self._kwargs(db, auth, "google-gcp", {"model": "gemini-2.5-pro"})
            )
        assert start.call_args.kwargs["provider_name"] == "google-gcp"
        vertex_from_cred.assert_called_once()
        # Vertex config omits the "models/" prefixed model (uses bare id).
        assert "model" not in start.call_args.kwargs["config"]

    def test_anthropic_branch_sets_max_tokens(self, db) -> None:
        auth = get_user_test_auth_context(db)
        job = _make_batch_job(
            db, org_id=auth.organization_id, project_id=auth.project_id
        )
        with (
            patch(
                "app.services.assessment.api.batch.get_anthropic_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.assessment.api.batch.start_batch_job",
                return_value=job,
            ) as start,
        ):
            _submit_provider_batch(
                **self._kwargs(db, auth, "anthropic", {"model": "claude-sonnet-4-6"})
            )
        assert start.call_args.kwargs["config"]["model"] == "claude-sonnet-4-6"

    def test_unsupported_provider_raises(self, db) -> None:
        auth = get_user_test_auth_context(db)
        with pytest.raises(ValueError, match="Unsupported provider"):
            _submit_provider_batch(**self._kwargs(db, auth, "cohere", {"model": "m"}))

    def test_empty_jsonl_raises(self, db) -> None:
        auth = get_user_test_auth_context(db)
        with (
            patch(
                "app.services.assessment.api.batch.get_openai_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.assessment.api.batch.build_openai_jsonl",
                return_value=[],
            ),
        ):
            with pytest.raises(ValueError, match="No batch rows built"):
                _submit_provider_batch(
                    **self._kwargs(db, auth, "openai", {"model": "gpt-4o"})
                )


class TestSubmitStageEmptySubset:
    def test_all_gated_out_skips_submit(self, db) -> None:
        auth = get_user_test_auth_context(db)
        blob = _blob()
        pipeline = build_pipeline(None)
        bag = _bag_for(pipeline, total=2)
        bag["gate_passed"] = [False, False]
        batch_input = BatchInput(data=[{"a": "1"}, {"a": "2"}])
        assessment = api.create_assessment(
            session=db,
            method=AssessmentMethod.BATCH,
            input=batch_input.model_dump(mode="json"),
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        execution = api.create_execution(
            session=db,
            assessment_id=assessment.id,
            config_id=_make_config_id(db, auth.project_id),
            config_version=1,
            total_items=2,
        )
        ok = _submit_stage(
            session=db,
            execution=execution,
            blob=blob,
            batch_input=batch_input,
            bag=bag,
            stage=ApiStage.ASSESSMENT.value,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        # Empty subset submits no provider batch and reports False so the driver
        # advances/finalizes instead of re-submitting a PENDING stage forever.
        assert ok is False
        assert bag["counters"][ApiStage.ASSESSMENT.value] == {
            "total": 0,
            "passed": 0,
            "rejected": 0,
        }


class TestRunBatchStageAllGatedOut:
    """BUG 1 regression: a gate that rejects every row must finalize the run, not
    livelock re-submitting an empty assessment stage forever."""

    def test_every_row_gated_out_reaches_terminal_and_delivers_once(self, db) -> None:
        auth = get_user_test_auth_context(db)
        org_id, project_id = auth.organization_id, auth.project_id
        blob = _blob(topic_relevance=True)  # one stop_on_fail gate + assessment
        config = create_test_config(
            db,
            project_id=project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(blob.pre_filters)
        bag = _bag_for(pipeline, total=2)
        assessment, execution = _seed_assessment(
            db,
            org_id=org_id,
            project_id=project_id,
            config_id=config.id,
            bag=bag,
            data=[{"a": "one"}, {"a": "two"}],
            status=AssessmentStatus.PROCESSING,
        )

        submitted_stages: list[str] = []

        def fake_start(**kwargs):
            submitted_stages.append(kwargs.get("config", {}).get("description", ""))
            return _make_batch_job(
                db,
                org_id=org_id,
                project_id=project_id,
                provider_status="completed",
                provider_output_file_id="file_out",
                raw_output_url="s3://bucket/out.jsonl",
            )

        # Both rows fail the gate -> gate_passed all False -> assessment subset empty.
        gate_results = [
            {
                "custom_id": f"row_{i}",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps(
                            {"verdict": False, "reasoning": "off"}
                        )
                    },
                },
            }
            for i in range(2)
        ]

        from app.models.assessment import AssessmentBatchResult, AssessmentCounts

        clean_result = AssessmentBatchResult(
            total_items=2, counts=AssessmentCounts(assessed=0, filtered=2), items=[]
        )
        delivered: list = []

        def call_tick():
            return run_batch_stage(
                execution_id=execution.id,
                organization_id=org_id,
                project_id=project_id,
            )

        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch(
                "app.services.assessment.api.batch.start_batch_job",
                side_effect=fake_start,
            ),
            patch(
                "app.services.assessment.api.batch.poll_batch_status",
                return_value={"request_counts": {"completed": 2, "failed": 0}},
            ),
            patch(
                "app.services.assessment.api.batch.process_completed_batch",
                return_value=(gate_results, {}),
            ),
            patch(
                "app.services.assessment.api.batch.get_openai_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=clean_result,
            ),
            patch(
                "app.services.assessment.api.callbacks.deliver",
                side_effect=lambda **kw: delivered.append(kw) or True,
            ),
        ):
            ticks = [call_tick()]  # PENDING -> submit gate
            assert ticks[0] == {"requeue": True}
            for _ in range(6):
                r = call_tick()
                ticks.append(r)
                if not r["requeue"]:
                    break

        # Terminal, not stuck PROCESSING, and the final tick did not ask for a requeue.
        assert ticks[-1] == {"requeue": False}
        db.refresh(execution)
        db.refresh(assessment)
        assert execution.status in (
            AssessmentStatus.COMPLETED,
            AssessmentStatus.COMPLETED_WITH_ERRORS,
        )
        assert assessment.status in (
            AssessmentStatus.COMPLETED,
            AssessmentStatus.COMPLETED_WITH_ERRORS,
        )
        assert execution.execution["gate_passed"] == [False, False]
        # Only the gate batch was ever submitted; the empty assessment stage was skipped.
        assert all("topic_relevance" in d for d in submitted_stages)
        assert len(delivered) == 1


class TestFinalizeAndFail:
    def _setup(self, db, auth, callback):
        pipeline = build_pipeline(None)
        bag = _bag_for(pipeline, total=1, callback=callback)
        return _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=_make_config_id(db, auth.project_id),
            bag=bag,
            data=[{"a": "1"}],
            status=AssessmentStatus.PROCESSING,
        )

    def _result(self, errors, total):
        from app.models.assessment import (
            AssessmentBatchResult,
            AssessmentOutput,
            AssessmentResult,
        )

        items = [
            AssessmentResult(
                output=AssessmentOutput(assessment="x"),
                error="boom" if i < errors else None,
            )
            for i in range(total)
        ]
        return AssessmentBatchResult(total_items=total, items=items)

    def test_finalize_all_errors_marks_failed(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = self._setup(db, auth, "")
        with (
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=self._result(errors=2, total=2),
            ),
            patch("app.services.assessment.api.callbacks.deliver") as deliver,
        ):
            _finalize(db, execution, assessment, dict(execution.execution))
        db.refresh(assessment)
        assert assessment.status == AssessmentStatus.FAILED
        deliver.assert_not_called()

    def test_finalize_partial_errors_marks_completed_with_errors_and_delivers(
        self, db
    ) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = self._setup(db, auth, "https://hook/cb")
        delivered: list = []
        with (
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=self._result(errors=1, total=2),
            ),
            patch(
                "app.services.assessment.api.callbacks.deliver",
                side_effect=lambda **kw: delivered.append(kw),
            ),
        ):
            _finalize(db, execution, assessment, dict(execution.execution))
        db.refresh(assessment)
        assert assessment.status == AssessmentStatus.COMPLETED_WITH_ERRORS
        assert len(delivered) == 1

    def test_fail_sets_failed_and_delivers(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = self._setup(db, auth, "https://hook/cb")
        delivered: list = []
        with (
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=self._result(errors=0, total=1),
            ),
            patch(
                "app.services.assessment.api.callbacks.deliver",
                side_effect=lambda **kw: delivered.append(kw),
            ),
        ):
            _fail(db, execution, assessment, dict(execution.execution), "kaboom")
        db.refresh(execution)
        db.refresh(assessment)
        assert execution.status == AssessmentStatus.FAILED
        assert execution.error_message == "kaboom"
        assert assessment.status == AssessmentStatus.FAILED
        assert len(delivered) == 1


class TestRunBatchStageProcessingBranches:
    def _processing_execution(self, db, auth, *, stage_pipeline_prefilter=False):
        blob = _blob(topic_relevance=True) if stage_pipeline_prefilter else _blob()
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(
            blob.pre_filters if stage_pipeline_prefilter else None
        )
        job = _make_batch_job(
            db, org_id=auth.organization_id, project_id=auth.project_id
        )
        bag = _bag_for(pipeline, total=2, callback="")
        bag["stage"] = pipeline[0]["stage"]
        bag["stage_status"] = StageStatus.PROCESSING.value
        bag["stage_batches"] = {pipeline[0]["stage"]: job.id}
        assessment, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=bag,
            data=[{"a": "one"}, {"a": "two"}],
            status=AssessmentStatus.PROCESSING,
        )
        return assessment, execution

    def _tick(self, db, execution, auth):
        return run_batch_stage(
            execution_id=execution.id,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )

    def test_poll_processing_requeues(self, db) -> None:
        auth = get_user_test_auth_context(db)
        _, execution = self._processing_execution(db, auth)
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch("app.services.assessment.api.batch._build_batch_provider"),
            patch(
                "app.services.assessment.api.batch._poll_outcome",
                return_value=("processing", None),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": True}

    def test_poll_error_retries(self, db) -> None:
        auth = get_user_test_auth_context(db)
        _, execution = self._processing_execution(db, auth)
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch("app.services.assessment.api.batch._build_batch_provider"),
            patch(
                "app.services.assessment.api.batch._poll_outcome",
                side_effect=RuntimeError("network hiccup"),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": True}

    def test_poll_failed_fails_run(self, db) -> None:
        auth = get_user_test_auth_context(db)
        _, execution = self._processing_execution(db, auth)
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch("app.services.assessment.api.batch._build_batch_provider"),
            patch(
                "app.services.assessment.api.batch._poll_outcome",
                return_value=("failed", None),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": False}
        db.refresh(execution)
        assert execution.status == AssessmentStatus.FAILED

    def test_advance_value_error_fails_run(self, db) -> None:
        auth = get_user_test_auth_context(db)
        _, execution = self._processing_execution(
            db, auth, stage_pipeline_prefilter=True
        )
        gate_results = [
            {
                "custom_id": "row_0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps({"verdict": True, "reasoning": ""})
                    },
                },
            },
            {
                "custom_id": "row_1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps({"verdict": True, "reasoning": ""})
                    },
                },
            },
        ]
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch("app.services.assessment.api.batch._build_batch_provider"),
            patch(
                "app.services.assessment.api.batch._poll_outcome",
                return_value=("completed", gate_results),
            ),
            patch(
                "app.services.assessment.api.batch._submit_stage",
                side_effect=ValueError("cannot submit next stage"),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": False}
        db.refresh(execution)
        assert execution.status == AssessmentStatus.FAILED
        assert execution.error_message == "cannot submit next stage"

    def test_pending_submit_value_error_fails_run(self, db) -> None:
        auth = get_user_test_auth_context(db)
        blob = _blob()
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(None)
        _, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=_bag_for(pipeline, total=1, callback=""),
            data=[{"a": "1"}],
            status=AssessmentStatus.PROCESSING,
        )
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch(
                "app.services.assessment.api.batch._submit_stage",
                side_effect=ValueError("bad stage build"),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": False}
        db.refresh(execution)
        assert execution.status == AssessmentStatus.FAILED
        assert execution.error_message == "bad stage build"

    def test_parent_assessment_missing(self, db) -> None:
        auth = get_user_test_auth_context(db)
        blob = _blob()
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(None)
        _, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=_bag_for(pipeline, total=1),
            data=[{"a": "1"}],
        )
        real_get = db.get

        def fake_get(model, ident, *a, **k):
            if model is Assessment:
                return None
            return real_get(model, ident, *a, **k)

        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch.object(db, "get", side_effect=fake_get),
        ):
            assert self._tick(db, execution, auth) == {"requeue": False}


class TestRunBatchStageFailurePaths:
    """BUG 2 regression: uncaught exceptions in blob resolution or stage submit must
    route to ``_fail`` (FAILED + webhook, no requeue), not strand the run in PROCESSING.
    Transient poll errors stay a requeue."""

    def _pending_assessment_execution(self, db, auth):
        blob = _blob()  # assessment-only pipeline, gate_passed all True
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(None)
        return _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=_bag_for(pipeline, total=1),
            data=[{"a": "one"}],
            status=AssessmentStatus.PROCESSING,
        )

    def _result(self):
        from app.models.assessment import AssessmentBatchResult, AssessmentCounts

        return AssessmentBatchResult(total_items=1, counts=AssessmentCounts(), items=[])

    def _tick(self, db, execution, auth):
        return run_batch_stage(
            execution_id=execution.id,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )

    def test_blob_resolution_error_fails_run(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = self._pending_assessment_execution(db, auth)
        delivered: list = []
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch(
                "app.services.assessment.api.batch._resolve_blob",
                side_effect=HTTPException(
                    status_code=404, detail="config version deleted"
                ),
            ),
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=self._result(),
            ),
            patch(
                "app.services.assessment.api.callbacks.deliver",
                side_effect=lambda **kw: delivered.append(kw),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": False}
        db.refresh(execution)
        db.refresh(assessment)
        assert execution.status == AssessmentStatus.FAILED
        assert assessment.status == AssessmentStatus.FAILED
        assert len(delivered) == 1

    def test_pending_submit_non_value_error_fails_run(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = self._pending_assessment_execution(db, auth)
        delivered: list = []
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch(
                "app.services.assessment.api.batch._submit_provider_batch",
                side_effect=RuntimeError("provider credentials missing"),
            ),
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=self._result(),
            ),
            patch(
                "app.services.assessment.api.callbacks.deliver",
                side_effect=lambda **kw: delivered.append(kw),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": False}
        db.refresh(execution)
        assert execution.status == AssessmentStatus.FAILED
        assert "provider credentials missing" in execution.error_message
        assert len(delivered) == 1

    def test_advance_submit_non_value_error_fails_run(self, db) -> None:
        # A gate stage completes, then submitting the next stage raises a non-ValueError.
        auth = get_user_test_auth_context(db)
        blob = _blob(topic_relevance=True)
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(blob.pre_filters)
        job = _make_batch_job(
            db, org_id=auth.organization_id, project_id=auth.project_id
        )
        bag = _bag_for(pipeline, total=2)
        bag["stage"] = pipeline[0]["stage"]
        bag["stage_status"] = StageStatus.PROCESSING.value
        bag["stage_batches"] = {pipeline[0]["stage"]: job.id}
        assessment, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=bag,
            data=[{"a": "one"}, {"a": "two"}],
            status=AssessmentStatus.PROCESSING,
        )
        gate_results = [
            {
                "custom_id": f"row_{i}",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps({"verdict": True, "reasoning": ""})
                    },
                },
            }
            for i in range(2)
        ]
        delivered: list = []
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch("app.services.assessment.api.batch._build_batch_provider"),
            patch(
                "app.services.assessment.api.batch._poll_outcome",
                return_value=("completed", gate_results),
            ),
            patch(
                "app.services.assessment.api.batch._submit_stage",
                side_effect=RuntimeError("provider blew up on next stage"),
            ),
            patch(
                "app.services.assessment.api.results.build_result",
                return_value=self._result(),
            ),
            patch(
                "app.services.assessment.api.callbacks.deliver",
                side_effect=lambda **kw: delivered.append(kw),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": False}
        db.refresh(execution)
        assert execution.status == AssessmentStatus.FAILED
        assert execution.error_message == "provider blew up on next stage"
        assert len(delivered) == 1

    def test_poll_exception_still_requeues_not_fails(self, db) -> None:
        auth = get_user_test_auth_context(db)
        blob = _blob()
        config = create_test_config(
            db,
            project_id=auth.project_id,
            name=f"assess-{random_lower_string()}",
            config_blob=blob,
            tag=ConfigTag.ASSESSMENT,
        )
        pipeline = build_pipeline(None)
        job = _make_batch_job(
            db, org_id=auth.organization_id, project_id=auth.project_id
        )
        bag = _bag_for(pipeline, total=1)
        bag["stage_status"] = StageStatus.PROCESSING.value
        bag["stage_batches"] = {pipeline[0]["stage"]: job.id}
        _, execution = _seed_assessment(
            db,
            org_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            bag=bag,
            data=[{"a": "one"}],
            status=AssessmentStatus.PROCESSING,
        )
        with (
            patch(
                "app.services.assessment.api.batch.Session",
                lambda _e: _SessionCtx(db),
            ),
            patch("app.services.assessment.api.batch._build_batch_provider"),
            patch(
                "app.services.assessment.api.batch._poll_outcome",
                side_effect=RuntimeError("transient provider hiccup"),
            ),
        ):
            assert self._tick(db, execution, auth) == {"requeue": True}
        db.refresh(execution)
        assert execution.status == AssessmentStatus.PROCESSING


def _openai_result(idx: int, *, output=None, status=200) -> dict:
    body: dict = {}
    if status >= 400:
        body["error"] = {"message": "server error"}
    elif output is not None:
        body["output_text"] = output
    return {
        "custom_id": f"row_{idx}",
        "response": {"status_code": status, "body": body},
    }


class TestBuildResult:
    """results.build_result assembled from the bag (verdicts) + stored assessment output.
    The cloud-storage read is the only external seam mocked."""

    def _seed(self, db, auth, *, data, bag):
        batch_input = BatchInput(data=data)
        assessment = api.create_assessment(
            session=db,
            method=AssessmentMethod.BATCH,
            input=batch_input.model_dump(mode="json"),
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        execution = api.create_execution(
            session=db,
            assessment_id=assessment.id,
            config_id=_make_config_id(db, auth.project_id),
            config_version=1,
            total_items=len(data),
        )
        api.save_execution_state(session=db, execution=execution, state=bag)
        return assessment

    def _bag(self, **overrides) -> dict:
        bag = {
            "gate_passed": [True],
            "verdicts": {},
            "stage_output_urls": {},
            "input_schema": {"a": {"type": "text"}},
            "provider": "openai",
        }
        bag.update(overrides)
        return bag

    def _storage_patch(self, jsonl: str):
        storage = MagicMock()
        storage.stream.return_value.read.return_value.decode.return_value = jsonl
        return patch(
            "app.services.assessment.api.results.get_cloud_storage",
            return_value=storage,
        )

    def test_structured_output_and_verdict(self, db) -> None:
        auth = get_user_test_auth_context(db)
        bag = self._bag(
            stage_output_urls={"assessment": "s3://b/out.jsonl"},
            verdicts={"topic_relevance": {"0": {"verdict": True, "reasoning": "ok"}}},
        )
        assessment = self._seed(db, auth, data=[{"a": "1"}], bag=bag)
        jsonl = json.dumps(_openai_result(0, output=json.dumps({"score": 7})))
        with self._storage_patch(jsonl):
            result = build_result(session=db, assessment=assessment)

        assert result.total_items == 1
        item = result.items[0]
        assert item.output.assessment == {"score": 7}
        assert item.output.pre_filter.topic_relevance.verdict is True
        assert item.error is None
        assert result.counts.assessed == 1
        assert result.counts.filtered == 0

    def test_gate_failed_row_has_null_assessment_and_is_filtered(self, db) -> None:
        auth = get_user_test_auth_context(db)
        bag = self._bag(
            gate_passed=[True, False],
            stage_output_urls={"assessment": "s3://b/out.jsonl"},
            verdicts={
                "topic_relevance": {
                    "0": {"verdict": True, "reasoning": ""},
                    "1": {"verdict": False, "reasoning": "off"},
                }
            },
        )
        assessment = self._seed(db, auth, data=[{"a": "1"}, {"a": "2"}], bag=bag)
        # Only the gate-passed row is present in the assessment output.
        jsonl = json.dumps(_openai_result(0, output=json.dumps({"score": 3})))
        with self._storage_patch(jsonl):
            result = build_result(session=db, assessment=assessment)

        assert result.items[1].output.assessment is None
        assert result.items[1].output.pre_filter.topic_relevance.verdict is False
        assert result.counts.assessed == 1
        assert result.counts.filtered == 1

    def test_error_output_counts_as_error(self, db) -> None:
        auth = get_user_test_auth_context(db)
        bag = self._bag(stage_output_urls={"assessment": "s3://b/out.jsonl"})
        assessment = self._seed(db, auth, data=[{"a": "1"}], bag=bag)
        jsonl = json.dumps(_openai_result(0, status=500))
        with self._storage_patch(jsonl):
            result = build_result(session=db, assessment=assessment)

        assert result.items[0].output.assessment is None
        assert result.items[0].error == "server error"
        assert result.counts.errors == 1
        assert result.counts.assessed == 0

    def test_free_text_output_kept_as_string(self, db) -> None:
        auth = get_user_test_auth_context(db)
        bag = self._bag(stage_output_urls={"assessment": "s3://b/out.jsonl"})
        assessment = self._seed(db, auth, data=[{"a": "1"}], bag=bag)
        jsonl = json.dumps(_openai_result(0, output="just prose"))
        with self._storage_patch(jsonl):
            result = build_result(session=db, assessment=assessment)

        assert result.items[0].output.assessment == "just prose"

    def test_storage_read_failure_flags_load_error(self, db) -> None:
        auth = get_user_test_auth_context(db)
        bag = self._bag(stage_output_urls={"assessment": "s3://b/out.jsonl"})
        assessment = self._seed(db, auth, data=[{"a": "1"}], bag=bag)
        storage = MagicMock()
        storage.stream.side_effect = RuntimeError("s3 down")
        with patch(
            "app.services.assessment.api.results.get_cloud_storage",
            return_value=storage,
        ):
            result = build_result(session=db, assessment=assessment)

        assert (
            result.items[0].error == "Assessment output could not be read from storage."
        )
        assert result.items[0].output.assessment is None

    def test_no_output_url_is_clean_empty(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment = self._seed(db, auth, data=[{"a": "1"}], bag=self._bag())
        result = build_result(session=db, assessment=assessment)

        assert result.items[0].output.assessment is None
        assert result.items[0].error is None
        assert result.counts.assessed == 0
