"""Tests for prefilter settings + pipeline stage ordering."""

from app.models.assessment import Stage
from app.services.assessment.prefilter import resolve_prefilter_settings
from app.services.assessment.stages import (
    build_pipeline,
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
