"""Tests for prefilter settings + pipeline stage ordering."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.crud.assessment import core as assessment_core
from app.crud.assessment.core import _read_exec
from app.models.assessment import AssessmentStatus, Stage, StageStatus
from app.services.assessment.prefilter import resolve_prefilter_settings
from app.services.assessment.stages import (
    advance_or_finalize,
    build_pipeline,
    build_prefilter_requests,
    next_stage,
    ordered_stages,
)

_FULL_INPUT = {
    "prefilter_config": {
        "topic_relevance": {"columns": ["Problem"], "prompt": "rubric"},
        "duplicate_detection": {"columns": ["Problem"]},
    }
}


class TestResolvePrefilterSettings:
    def test_both_enabled(self) -> None:
        cfg = resolve_prefilter_settings(_FULL_INPUT["prefilter_config"])
        assert cfg["tr_enabled"] is True
        assert cfg["dup_enabled"] is True

    def test_disabled_when_empty(self) -> None:
        cfg = resolve_prefilter_settings({})
        assert cfg["tr_enabled"] is False
        assert cfg["dup_enabled"] is False

    def test_tr_enabled_with_attachment_columns_only(self) -> None:
        cfg = resolve_prefilter_settings(
            {
                "topic_relevance": {
                    "columns": [],
                    "attachment_columns": ["Answer Sheet"],
                    "prompt": "rubric",
                }
            }
        )
        assert cfg["tr_enabled"] is True

    def test_tr_disabled_when_columns_empty_and_no_prompt(self) -> None:
        cfg = resolve_prefilter_settings(
            {"topic_relevance": {"attachment_columns": ["Answer Sheet"]}}
        )
        assert cfg["tr_enabled"] is False


class TestPipeline:
    def test_full_pipeline_order(self) -> None:
        pipeline = build_pipeline(_FULL_INPUT)
        assert ordered_stages(pipeline) == [
            Stage.PRE_FILTER_TOPIC_RELEVANCE,
            Stage.PRE_FILTER_DUPLICATE_DETECTION,
            Stage.L2_ASSESSMENT,
        ]

    def test_no_prefilter_is_l2_only(self) -> None:
        pipeline = build_pipeline({})
        assert ordered_stages(pipeline) == [Stage.L2_ASSESSMENT]
        assert next_stage(pipeline) == Stage.L2_ASSESSMENT

    def test_next_stage(self) -> None:
        pipeline = build_pipeline(_FULL_INPUT)
        assert next_stage(pipeline, Stage.PRE_FILTER_TOPIC_RELEVANCE) == (
            Stage.PRE_FILTER_DUPLICATE_DETECTION
        )
        assert next_stage(pipeline, Stage.L2_ASSESSMENT) is None


class TestAdvanceOrFinalize:
    def test_advances_to_next_pending_stage(self) -> None:
        run = SimpleNamespace(
            execution={
                "pipeline": build_pipeline(_FULL_INPUT),
                "stage": Stage.PRE_FILTER_TOPIC_RELEVANCE,
                "stage_status": StageStatus.COMPLETED,
            },
            status="processing",
        )
        with patch.object(assessment_core, "flag_modified"):
            nxt = advance_or_finalize(run)
        assert nxt == Stage.PRE_FILTER_DUPLICATE_DETECTION
        assert _read_exec(run).get("stage") == Stage.PRE_FILTER_DUPLICATE_DETECTION
        assert _read_exec(run).get("stage_status") == StageStatus.PENDING

    def test_finalizes_after_last_stage(self) -> None:
        run = SimpleNamespace(
            execution={
                "pipeline": build_pipeline({}),
                "stage": Stage.L2_ASSESSMENT,
                "stage_status": StageStatus.COMPLETED,
            },
            status="processing",
        )
        with patch.object(assessment_core, "flag_modified"):
            assert advance_or_finalize(run) is None
        assert _read_exec(run).get("stage") == Stage.COMPLETED
        assert _read_exec(run).get("stage_status") == StageStatus.COMPLETED
        assert run.status == AssessmentStatus.COMPLETED


class TestBuildPrefilterRequests:
    _CFG = {
        "tr_columns": ["Problem"],
        "tr_prompt": "rubric",
        "dup_columns": ["Problem"],
    }

    def test_topic_relevance_stage(self) -> None:
        lines = build_prefilter_requests(
            Stage.PRE_FILTER_TOPIC_RELEVANCE, [(0, {"Problem": "p"})], self._CFG
        )
        assert len(lines) == 1

    def test_duplicate_detection_stage(self) -> None:
        lines = build_prefilter_requests(
            Stage.PRE_FILTER_DUPLICATE_DETECTION, [(0, {"Problem": "p"})], self._CFG
        )
        assert len(lines) == 1

    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(ValueError):
            build_prefilter_requests("BOGUS", [(0, {"Problem": "p"})], self._CFG)
