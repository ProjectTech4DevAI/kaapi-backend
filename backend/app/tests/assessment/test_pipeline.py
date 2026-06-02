"""Tests for the prefilter pipeline orchestrator."""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from app.services.assessment.prefilter.pipeline import run_prefilter_pipeline


def _run() -> MagicMock:
    run = MagicMock()
    run.id = 99
    return run


def _tr(verdict: bool, decision: str = "ACCEPT") -> dict:
    return {
        "row_id": "row",
        "verdict": verdict,
        "decision": decision,
        "column_relevance": {"Problem": verdict},
        "reasoning": "r",
    }


def _patches(stack: ExitStack, *, tr_side=None, dup_return=None):
    """Patch the pipeline's external deps; return the TR mock."""
    client = MagicMock()
    stack.enter_context(
        patch(
            "app.services.assessment.prefilter.pipeline.GeminiClient.from_credentials",
            return_value=MagicMock(client=client),
        )
    )
    stack.enter_context(
        patch(
            "app.services.assessment.prefilter.pipeline.get_cloud_storage",
            return_value=MagicMock(),
        )
    )
    stack.enter_context(
        patch(
            "app.services.assessment.prefilter.pipeline.upload_jsonl_to_object_store",
            return_value="s3://prefilter.json",
        )
    )
    stack.enter_context(
        patch("app.crud.assessment.core.update_assessment_run_prefilter_stats")
    )
    tr_mock = stack.enter_context(
        patch("app.services.assessment.prefilter.pipeline.run_topic_relevance")
    )
    if tr_side is not None:
        tr_mock.side_effect = tr_side
    dup_mock = stack.enter_context(
        patch("app.services.assessment.prefilter.pipeline.run_duplicate_detection")
    )
    if dup_return is not None:
        dup_mock.return_value = dup_return
    return tr_mock, dup_mock


class TestRunL1Pipeline:
    def test_no_filters_configured_passthrough(self) -> None:
        rows = [{"Problem": "a"}, {"Problem": "b"}]
        passed, indices, results = run_prefilter_pipeline(
            run=_run(),
            rows=rows,
            prefilter_config={},
            session=MagicMock(),
            organization_id=1,
            project_id=1,
        )
        assert passed == rows
        assert indices == [0, 1]
        assert results == []

    def test_topic_relevance_filters_rejected_rows(self) -> None:
        rows = [{"Problem": "keep"}, {"Problem": "drop"}, {"Problem": "keep2"}]
        # idx 1 rejected.
        side = [_tr(True), _tr(False, "REJECT"), _tr(True)]
        with ExitStack() as stack:
            _patches(stack, tr_side=side)
            passed, indices, results = run_prefilter_pipeline(
                run=_run(),
                rows=rows,
                prefilter_config={
                    "topic_relevance": {"columns": ["Problem"], "prompt": "rubric"}
                },
                session=MagicMock(),
                organization_id=1,
                project_id=1,
            )
        assert indices == [0, 2]
        assert [r["Problem"] for r in passed] == ["keep", "keep2"]
        assert len(results) == 3
        assert results[1]["prefilter_passed"] is False

    def test_duplicate_detection_runs_on_passed_rows(self) -> None:
        rows = [{"Problem": "a", "Solution": "b"}]
        dup = {
            "row_id": "row_0",
            "verdict": "UNIQUE",
            "match_title": None,
            "source_url": None,
            "matching_sentence": None,
            "reason": "novel",
        }
        with ExitStack() as stack:
            tr_mock, dup_mock = _patches(stack, tr_side=[_tr(True)], dup_return=dup)
            _, _, results = run_prefilter_pipeline(
                run=_run(),
                rows=rows,
                prefilter_config={
                    "topic_relevance": {"columns": ["Problem"], "prompt": "rubric"},
                    "duplicate_detection": {"columns": ["Problem", "Solution"]},
                },
                session=MagicMock(),
                organization_id=1,
                project_id=1,
            )
        dup_mock.assert_called_once()
        assert results[0]["duplicate_detection"]["verdict"] == "UNIQUE"

    def test_attachment_columns_filtered_to_selection(self) -> None:
        from app.models.assessment import AssessmentAttachment

        rows = [{"Problem": "a", "Docs": "x", "Other": "y"}]
        atts = [
            AssessmentAttachment(column="Docs", type="image", format="url"),
            AssessmentAttachment(column="Other", type="image", format="url"),
        ]
        with ExitStack() as stack:
            tr_mock, _ = _patches(stack, tr_side=[_tr(True)])
            run_prefilter_pipeline(
                run=_run(),
                rows=rows,
                prefilter_config={
                    "topic_relevance": {
                        "columns": ["Problem"],
                        "prompt": "rubric",
                        "attachment_columns": ["Docs"],
                    }
                },
                session=MagicMock(),
                organization_id=1,
                project_id=1,
                attachments=atts,
            )
        # run_topic_relevance is called with only the selected attachment ("Docs").
        passed_atts = tr_mock.call_args.args[6]
        assert [a.column for a in passed_atts] == ["Docs"]
