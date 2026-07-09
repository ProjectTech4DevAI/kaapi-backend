"""Integration tests for the correctness judge inside the fast-eval scoring stage.

Drives `_stage3_score_and_trace` with a real DB (`db` fixture) and mocked HTTP
boundaries (OpenAI judge completion + Langfuse). Covers:

- FR-1 / FR-8: every scoreable row's trace carries both a Cosine Similarity and a
  Correctness score, and both land in EvaluationRun.score.summary_scores.
- FR-2: the Correctness value is in [0,1] with a non-empty reasoning comment.
- FR-3: zero-config (judge_config=None) still judges every scoreable row.
- FR-9: `update_traces_with_correctness_scores` writes a Correctness trace score
  per item, isolating a per-item failure.
- FR-10: a row whose judge call returns malformed output is flagged judge_failed,
  its cosine is untouched, and sibling rows + the run still complete.
- FR-11: EvaluationRun.cost gains a `judge` stage with token counts + USD.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, select

from app.crud.evaluations.fast import _stage3_score_and_trace
from app.crud.evaluations.langfuse import update_traces_with_correctness_scores
from app.crud.evaluations.score import (
    CORRECTNESS_SCORE_NAME,
    COSINE_SCORE_NAME,
    JUDGE_FAILED_REASON,
)
from app.models import EvaluationRun, Organization, Project
from app.models.evaluation import RunModeEnum
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)

_IDENTICAL_VEC = [1.0, 0.0, 0.0]


def _response_row(item_id: str, question: str, answer: str, ground_truth: str):
    return {
        "item_id": item_id,
        "question": question,
        "generated_output": answer,
        "ground_truth": ground_truth,
        "response_id": f"resp_{item_id}",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "question_id": None,
        "failed": False,
    }


def _embedding_row(item_id: str):
    return {
        "item_id": item_id,
        "output_embedding": list(_IDENTICAL_VEC),
        "ground_truth_embedding": list(_IDENTICAL_VEC),
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
        "failed": False,
    }


def _judge_reply(score: float, reasoning: str):
    import json

    return SimpleNamespace(
        output_text=json.dumps({"score": score, "reasoning": reasoning}),
        output=[],
        usage=SimpleNamespace(input_tokens=20, output_tokens=6, total_tokens=26),
    )


def _make_eval_run(db: Session, *, total_items: int) -> EvaluationRun:
    org = db.exec(select(Organization)).first()
    project = db.exec(select(Project).where(Project.organization_id == org.id)).first()
    dataset = create_test_evaluation_dataset(
        db=db, organization_id=org.id, project_id=project.id
    )
    config = create_test_config(db, project_id=project.id, use_kaapi_schema=True)
    eval_run = EvaluationRun(
        run_name=f"corr-{dataset.name}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=1,
        status="processing",
        run_mode=RunModeEnum.FAST.value,
        total_items=total_items,
        organization_id=org.id,
        project_id=project.id,
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    return eval_run


class _Stage3Harness:
    """Patches the non-judge externals of `_stage3_score_and_trace`.

    Langfuse trace creation, model resolution, and cost pricing are mocked; the
    judge itself (build_judge_params + judge_row) and cosine math run for real.
    """

    def __init__(self, trace_map: dict[str, str], estimate: dict[str, float] | None):
        self.trace_map = trace_map
        self.estimate = estimate

    def __enter__(self):
        self._patches = [
            patch(
                "app.crud.evaluations.fast.create_langfuse_dataset_run",
                return_value=self.trace_map,
            ),
            patch(
                "app.crud.evaluations.fast.resolve_model_from_config",
                return_value="gpt-4o",
            ),
            patch(
                "app.crud.evaluations.cost.estimate_model_cost",
                return_value=self.estimate,
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


class TestStage3Correctness:
    def test_fr1_fr8_scoreable_rows_get_both_cosine_and_correctness(
        self, db: Session
    ) -> None:
        eval_run = _make_eval_run(db, total_items=2)
        response_results = [
            _response_row("item-1", "Q1", "A1", "GT1"),
            _response_row("item-2", "Q2", "A2", "GT2"),
        ]
        embedding_results = [_embedding_row("item-1"), _embedding_row("item-2")]

        openai_client = MagicMock()
        openai_client.responses.create.return_value = _judge_reply(
            0.8, "Correct and complete."
        )
        langfuse = MagicMock()

        with _Stage3Harness(
            trace_map={"item-1": "trace-1", "item-2": "trace-2"},
            estimate={"input_cost": 0.01, "output_cost": 0.02},
        ):
            run, score, _writes, correctness_writes = _stage3_score_and_trace(
                session=db,
                openai_client=openai_client,
                eval_run=eval_run,
                langfuse=langfuse,
                response_results=response_results,
                embedding_results=embedding_results,
                judge_config=None,
                log_prefix="[t]",
            )

        # FR-1: every trace carries both score names.
        by_trace = {t["trace_id"]: t for t in score["traces"]}
        for trace_id in ("trace-1", "trace-2"):
            names = {s["name"] for s in by_trace[trace_id]["scores"]}
            assert COSINE_SCORE_NAME in names
            assert CORRECTNESS_SCORE_NAME in names

        # FR-2: correctness value in [0,1] with non-empty reasoning comment.
        correctness = next(
            s
            for s in by_trace["trace-1"]["scores"]
            if s["name"] == CORRECTNESS_SCORE_NAME
        )
        assert 0.0 <= correctness["value"] <= 1.0
        assert correctness["value"] == 0.8
        assert correctness["comment"] == "Correct and complete."
        assert correctness["data_type"] == "NUMERIC"

        # FR-8: both summaries present, next to each other.
        summary_names = {s["name"] for s in score["summary_scores"]}
        assert COSINE_SCORE_NAME in summary_names
        assert CORRECTNESS_SCORE_NAME in summary_names

        # Durable per-row correctness map is the resync source of truth.
        assert run.per_item_correctness == {"trace-1": 0.8, "trace-2": 0.8}
        assert {w["trace_id"] for w in correctness_writes} == {"trace-1", "trace-2"}

    def test_fr3_zero_config_judges_every_scoreable_row(self, db: Session) -> None:
        """No judge_config → the fallback model still scores each scoreable row."""
        eval_run = _make_eval_run(db, total_items=3)
        response_results = [
            _response_row(f"item-{i}", f"Q{i}", f"A{i}", f"GT{i}") for i in range(3)
        ]
        embedding_results = [_embedding_row(f"item-{i}") for i in range(3)]

        openai_client = MagicMock()
        openai_client.responses.create.return_value = _judge_reply(0.6, "ok")

        with _Stage3Harness(
            trace_map={f"item-{i}": f"trace-{i}" for i in range(3)},
            estimate={"input_cost": 0.0, "output_cost": 0.0},
        ):
            run, score, _writes, correctness_writes = _stage3_score_and_trace(
                session=db,
                openai_client=openai_client,
                eval_run=eval_run,
                langfuse=MagicMock(),
                response_results=response_results,
                embedding_results=embedding_results,
                judge_config=None,
                log_prefix="[t]",
            )

        # One judge call per scoreable row, all scored.
        assert openai_client.responses.create.call_count == 3
        assert len(run.per_item_correctness) == 3
        assert len(correctness_writes) == 3

    def test_fr10_malformed_judge_output_isolated_from_cosine_and_siblings(
        self, db: Session
    ) -> None:
        eval_run = _make_eval_run(db, total_items=2)
        response_results = [
            _response_row("item-1", "Q1", "A1", "GT1"),
            _response_row("item-2", "Q2", "A2", "GT2"),
        ]
        embedding_results = [_embedding_row("item-1"), _embedding_row("item-2")]

        # item-2's judge reply is non-JSON → that single row is flagged unscoreable.
        def _judge_side_effect(**kwargs: Any):
            if "Q2" in kwargs["input"]:
                return SimpleNamespace(
                    output_text="totally not json",
                    output=[],
                    usage=SimpleNamespace(
                        input_tokens=1, output_tokens=1, total_tokens=2
                    ),
                )
            return _judge_reply(0.9, "great")

        openai_client = MagicMock()
        openai_client.responses.create.side_effect = _judge_side_effect

        with _Stage3Harness(
            trace_map={"item-1": "trace-1", "item-2": "trace-2"},
            estimate={"input_cost": 0.01, "output_cost": 0.02},
        ):
            run, score, _writes, correctness_writes = _stage3_score_and_trace(
                session=db,
                openai_client=openai_client,
                eval_run=eval_run,
                langfuse=MagicMock(),
                response_results=response_results,
                embedding_results=embedding_results,
                judge_config=None,
                log_prefix="[t]",
            )

        by_trace = {t["trace_id"]: t for t in score["traces"]}

        # Failed row flagged judge_failed; sibling scored.
        assert run.unscoreable["trace-2"] == JUDGE_FAILED_REASON
        assert "trace-1" not in (run.unscoreable or {})
        assert run.per_item_correctness == {"trace-1": 0.9}

        # FR-10: the failed row keeps its real cosine score (judge failure never
        # touches cosine) and carries no Correctness entry.
        failed_names = {s["name"] for s in by_trace["trace-2"]["scores"]}
        assert CORRECTNESS_SCORE_NAME not in failed_names
        cosine = next(
            s for s in by_trace["trace-2"]["scores"] if s["name"] == COSINE_SCORE_NAME
        )
        assert cosine["value"] == pytest.approx(1.0, abs=0.01)
        assert not cosine.get("unscoreable")

        # Sibling still has both scores.
        assert {s["name"] for s in by_trace["trace-1"]["scores"]} == {
            COSINE_SCORE_NAME,
            CORRECTNESS_SCORE_NAME,
        }

    def test_fr11_cost_includes_judge_stage_with_tokens_and_usd(
        self, db: Session
    ) -> None:
        eval_run = _make_eval_run(db, total_items=2)
        response_results = [
            _response_row("item-1", "Q1", "A1", "GT1"),
            _response_row("item-2", "Q2", "A2", "GT2"),
        ]
        embedding_results = [_embedding_row("item-1"), _embedding_row("item-2")]

        openai_client = MagicMock()
        openai_client.responses.create.return_value = _judge_reply(0.7, "fine")

        with _Stage3Harness(
            trace_map={"item-1": "trace-1", "item-2": "trace-2"},
            estimate={"input_cost": 0.03, "output_cost": 0.05},
        ):
            run, _score, _writes, _corr = _stage3_score_and_trace(
                session=db,
                openai_client=openai_client,
                eval_run=eval_run,
                langfuse=MagicMock(),
                response_results=response_results,
                embedding_results=embedding_results,
                judge_config=None,
                log_prefix="[t]",
            )

        assert "judge" in run.cost
        judge_cost = run.cost["judge"]
        assert judge_cost["model"] == "gpt-4o-mini"  # EVAL_JUDGE_FALLBACK_MODEL
        # Two rows judged, each 20 input / 6 output tokens.
        assert judge_cost["input_tokens"] == 40
        assert judge_cost["output_tokens"] == 12
        assert judge_cost["total_tokens"] == 52
        assert judge_cost["cost_usd"] > 0


class TestUpdateTracesWithCorrectnessScores:
    """FR-9: per-item Correctness trace-score writes, failures isolated."""

    def test_writes_correctness_score_per_item(self) -> None:
        langfuse = MagicMock()
        per_item = [
            {"trace_id": "trace_1", "correctness": 0.9, "reasoning": "accurate"},
            {"trace_id": "trace_2", "correctness": 0.4, "reasoning": "partial"},
        ]

        failed = update_traces_with_correctness_scores(
            langfuse=langfuse, per_item_correctness=per_item
        )

        assert failed == []
        assert langfuse.create_score.call_count == 2
        calls = langfuse.create_score.call_args_list
        assert calls[0].kwargs["name"] == CORRECTNESS_SCORE_NAME
        assert calls[0].kwargs["trace_id"] == "trace_1"
        assert calls[0].kwargs["value"] == 0.9
        assert calls[0].kwargs["comment"] == "accurate"
        langfuse.flush.assert_called_once()

    def test_per_item_failure_isolated(self) -> None:
        langfuse = MagicMock()

        def _score_side_effect(**kwargs: Any) -> None:
            if kwargs.get("trace_id") == "trace_2":
                raise Exception("write failed")

        langfuse.create_score.side_effect = _score_side_effect
        per_item = [
            {"trace_id": "trace_1", "correctness": 0.9, "reasoning": "ok"},
            {"trace_id": "trace_2", "correctness": 0.4, "reasoning": "bad"},
            {"trace_id": "trace_3", "correctness": 0.5, "reasoning": "meh"},
        ]

        failed = update_traces_with_correctness_scores(
            langfuse=langfuse, per_item_correctness=per_item
        )

        # Only the bad trace is reported; siblings still written.
        assert failed == ["trace_2"]
        assert langfuse.create_score.call_count == 3
