"""End-to-end judge scoring on the v2 fast pipeline (`run_fast_evaluation`).

Drives the ground-truth + adherence-to-prompt slices of the three-metric SRD
through the real fast pipeline with a judged run (`is_judge_run=True`,
`langfuse=None` as v2 dispatches): FR-2 (trace scores in [0,1] + reasoning), FR-9
(zero-config uses the fallback model + built-in prompts), FR-14 (run-level summary
scores + per-row scores on the trace records), FR-15 (per-row isolation), FR-16
(judge cost stage), FR-18 (v1 never judges).

A v2 judged run drops cosine + embeddings entirely: no embedding API calls, no
"Cosine Similarity" trace/summary score, and `per_item_scores` stays NULL. Both
judge metrics come from ONE combined call per row, and a run whose config carries
no resolvable instructions silently drops the prompt metric. v1 (`is_judge_run`
False) is unchanged — cosine only, no judge.

External boundaries mocked: OpenAI (embeddings + the judge completion at
`_create_judge_response`), S3, `save_score`, model/cost resolution. DB is real.
"""

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.fast import (
    CHUNK_CONFIG_INDEX,
    CHUNK_CONFIG_RUN_ID,
    JOB_TYPE_EVALUATION_FAST_CHUNK,
    PROMPT_TEMPLATE_LABEL,
    run_fast_evaluation,
)
from app.crud.evaluations.judge import JUDGE_COST_STAGE
from app.crud.evaluations.score import (
    GROUND_TRUTH_SCORE_NAME,
    JUDGE_FAILED_REASON,
    PROMPT_SCORE_NAME,
)
from app.models import Config, EvaluationDataset, EvaluationRun
from app.models.batch_job import BatchJob
from app.models.evaluation import RunModeEnum
from app.models.llm.request import ConfigBlob, KaapiCompletionConfig, PromptTemplate
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)
from app.tests.utils.utils import random_lower_string

COSINE_SCORE_NAME = "Cosine Similarity"


def _make_dataset(*, db: Session, user_api_key: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        original_items_count=3,
        duplication_factor=1,
    )


def _make_text_config(
    db: Session,
    project_id: int,
    *,
    instructions: str | None = None,
    prompt_template: str | None = None,
) -> Config:
    params: dict[str, Any] = {"model": "gpt-4o-fast-eval-test", "temperature": 0.7}
    if instructions is not None:
        params["instructions"] = instructions
    blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params=params,
        ),
        prompt_template=(
            PromptTemplate(template=prompt_template) if prompt_template else None
        ),
    )
    return create_test_config(
        db=db, project_id=project_id, use_kaapi_schema=True, config_blob=blob
    )


def _make_run(
    *,
    db: Session,
    user_api_key: TestAuthContext,
    is_judge_run: bool,
    instructions: str | None = None,
    prompt_template: str | None = None,
    config_version: int = 1,
) -> EvaluationRun:
    dataset = _make_dataset(db=db, user_api_key=user_api_key)
    config = _make_text_config(
        db,
        user_api_key.project_id,
        instructions=instructions,
        prompt_template=prompt_template,
    )
    run = EvaluationRun(
        run_name=f"run-{random_lower_string()}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=config_version,
        status="pending",
        run_mode=RunModeEnum.FAST.value,
        total_items=0,
        is_judge_run=is_judge_run or None,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _resp_result(
    item_id: str, question: str, ground_truth: str = "golden"
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "question": question,
        "generated_output": f"generated for {question}",
        "ground_truth": ground_truth,
        "response_id": f"resp_{item_id}",
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        "question_id": item_id,
        "failed": False,
    }


def _fake_embedding_response():
    """Two identical unit vectors → cosine ≈ 1.0."""
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            SimpleNamespace(index=1, embedding=[1.0, 0.0, 0.0]),
        ],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )


def _judge_response(score: float, reasoning: str, *, usage=(12, 6, 18)):
    body = json.dumps({"ground_truth": {"score": score, "reasoning": reasoning}})
    return _raw_judge_response(body, usage=usage)


def _both_metrics_response(
    *,
    ground_truth: tuple[float, str] = (0.8, "conveys the same facts"),
    prompt: tuple[float, str] = (0.6, "answered in the wrong language"),
    usage=(12, 6, 18),
):
    gt_score, gt_reason = ground_truth
    prompt_score, prompt_reason = prompt
    body = json.dumps(
        {
            "ground_truth": {"score": gt_score, "reasoning": gt_reason},
            "prompt": {"score": prompt_score, "reasoning": prompt_reason},
        }
    )
    return _raw_judge_response(body, usage=usage)


def _raw_judge_response(text: str, *, usage=(12, 6, 18)):
    i, o, t = usage
    return SimpleNamespace(
        output_text=text,
        output=[],
        usage=SimpleNamespace(input_tokens=i, output_tokens=o, total_tokens=t),
    )


@pytest.fixture
def _s3_store() -> Iterator[dict[str, list[dict[str, Any]]]]:
    store: dict[str, list[dict[str, Any]]] = {}

    def _upload(*, filename, results, **_):
        url = f"s3://bucket/{filename}"
        store[url] = list(results)
        return url

    def _load(*, url, **_):
        return store[url]

    with (
        patch("app.crud.evaluations.fast._upload_unit_to_s3", side_effect=_upload),
        patch("app.crud.evaluations.fast._load_unit_from_s3", side_effect=_load),
    ):
        yield store


def _seed_chunk(
    *,
    db: Session,
    eval_run: EvaluationRun,
    results: list[dict[str, Any]],
    store: dict[str, list[dict[str, Any]]],
) -> None:
    url = f"s3://bucket/responses_{eval_run.id}_0.json"
    store[url] = results
    job = BatchJob(
        provider="openai",
        job_type=JOB_TYPE_EVALUATION_FAST_CHUNK,
        config={
            "model": "gpt-4o",
            CHUNK_CONFIG_RUN_ID: eval_run.id,
            CHUNK_CONFIG_INDEX: 0,
        },
        raw_output_url=url,
        total_items=len(results),
        organization_id=eval_run.organization_id,
        project_id=eval_run.project_id,
    )
    db.add(job)
    db.commit()


def _persist_score_into(db: Session):
    def _fake_save_score(*, eval_run_id, score, **_):
        run = db.get(EvaluationRun, eval_run_id)
        run.score = {"summary_scores": score["summary_scores"]}
        run.score_trace_url = f"s3://bucket/traces_{eval_run_id}.json"
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    return _fake_save_score


def _run_pipeline(
    *,
    db: Session,
    eval_run: EvaluationRun,
    judge_side_effect,
    mock_cost: bool = False,
) -> tuple[EvaluationRun, MagicMock]:
    """Run `run_fast_evaluation` for a judged run with all externals stubbed.

    `langfuse=None` mirrors what the v2 aggregate passes for a judged run, so refs
    key by item_id. The judge completion is driven by `judge_side_effect(params)`.
    Returns the run plus the OpenAI mock so callers can assert the embedding path
    was (v1) or was not (v2 judge) exercised.
    """
    fake_openai = MagicMock()
    fake_openai.embeddings.create.return_value = _fake_embedding_response()

    def _judge(_client, params):
        return judge_side_effect(params)

    ctx = [
        patch(
            "app.crud.evaluations.fast.resolve_model_from_config", return_value="gpt-4o"
        ),
        patch(
            "app.crud.evaluations.fast.save_score", side_effect=_persist_score_into(db)
        ),
        patch("app.crud.evaluations.judge._create_judge_response", side_effect=_judge),
    ]
    if mock_cost:
        ctx.append(
            patch(
                "app.crud.evaluations.cost.estimate_model_cost",
                return_value={"input_cost": 0.001, "output_cost": 0.002},
            )
        )

    with ctx[0], ctx[1], ctx[2]:
        if mock_cost:
            with ctx[3]:
                result = run_fast_evaluation(
                    session=db,
                    openai_client=fake_openai,
                    langfuse=None,
                    eval_run=eval_run,
                )
        else:
            result = run_fast_evaluation(
                session=db, openai_client=fake_openai, langfuse=None, eval_run=eval_run
            )
    return result, fake_openai


def _trace_by_ref(result: EvaluationRun) -> dict[str, dict[str, Any]]:
    return {t["trace_id"]: t for t in result.score["traces"]}


def _score_named(trace: dict[str, Any], name: str) -> dict[str, Any] | None:
    for s in trace["scores"]:
        if s["name"] == name:
            return s
    return None


def _metric_values(result: EvaluationRun, name: str) -> dict[str, float]:
    """{ref: value} for one metric across the run's trace scores.

    Replaces the removed per-row judge map columns: per-row judge scores now live
    only in the score unit's trace scores.
    """
    return {
        trace_id: score["value"]
        for trace_id, trace in _trace_by_ref(result).items()
        if (score := _score_named(trace, name)) is not None
    }


def _summary_named(result: EvaluationRun, name: str) -> dict[str, Any] | None:
    for s in result.score["summary_scores"]:
        if s["name"] == name:
            return s
    return None


class TestGroundTruthScoring:
    def test_ground_truth_score_is_the_only_scorer_no_cosine(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-2/FR-14 + v2 contract: each row carries ONLY a ground-truth score (no
        cosine); summary + per-row trace scores populated; per_item_scores stays NULL
        and no embeddings are computed for a judged run."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[
                _resp_result("item-1", "Q1", "golden-1"),
                _resp_result("item-2", "Q2", "golden-2"),
            ],
            store=_s3_store,
        )

        result, fake_openai = _run_pipeline(
            db=db,
            eval_run=eval_run,
            judge_side_effect=lambda _p: _judge_response(0.8, "conveys the same facts"),
        )

        assert result.status == "completed"
        # A judged run never embeds — cosine is gone entirely.
        fake_openai.embeddings.create.assert_not_called()

        traces = _trace_by_ref(result)
        assert set(traces) == {"item-1", "item-2"}
        for ref in ("item-1", "item-2"):
            gt = _score_named(traces[ref], GROUND_TRUTH_SCORE_NAME)
            assert _score_named(traces[ref], COSINE_SCORE_NAME) is None
            assert gt is not None
            assert 0.0 <= gt["value"] <= 1.0
            assert gt["value"] == pytest.approx(0.8, abs=0.01)
            assert gt["comment"] == "conveys the same facts"

        summary_names = {s["name"] for s in result.score["summary_scores"]}
        assert GROUND_TRUTH_SCORE_NAME in summary_names
        assert COSINE_SCORE_NAME not in summary_names
        gt_summary = next(
            s
            for s in result.score["summary_scores"]
            if s["name"] == GROUND_TRUTH_SCORE_NAME
        )
        assert gt_summary["avg"] == pytest.approx(0.8, abs=0.01)
        assert gt_summary["total_pairs"] == 2

        assert _metric_values(result, GROUND_TRUTH_SCORE_NAME) == {
            "item-1": 0.8,
            "item-2": 0.8,
        }

        run = db.get(EvaluationRun, result.id)
        # Cosine's durable per-row map is a v1-only artifact; a judged run leaves it NULL.
        assert run.per_item_scores is None

    def test_zero_config_judges_with_fallback_model_and_builtin_prompt(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-9: a judged run uses the fallback model and the built-in ground-truth
        prompt (as its instructions) — judging is system-config only."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[_resp_result("item-1", "Q1", "golden-1")],
            store=_s3_store,
        )

        captured: dict = {}

        def _capture(params):
            captured.update(params)
            return _judge_response(0.6, "partially correct")

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_capture)

        assert result.status == "completed"
        assert captured["model"] == settings.EVAL_JUDGE_MODEL
        # The judge is a reasoning model that rejects a custom temperature.
        assert "temperature" not in captured
        # The built-in ground-truth rubric drives the grade (as the call instructions).
        assert "Adherence to Ground Truth" in captured["instructions"]
        # The golden answer reaches the judge (FR-3).
        assert "golden-1" in captured["input"]

    def test_per_row_isolation_leaves_failed_row_unscoreable(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-15: a malformed judge reply for one row flags that row judge_failed and
        drops its ground-truth score, while its siblings and the run survive. Neither
        row carries a cosine score (v2 never embeds)."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[
                _resp_result("item-good", "Q-good", "golden-good"),
                _resp_result("item-bad", "Q-bad", "golden-bad"),
            ],
            store=_s3_store,
        )

        def _judge(params):
            if "Q-bad" in params["input"]:
                return _raw_judge_response("totally not json")
            return _judge_response(0.9, "correct")

        result, fake_openai = _run_pipeline(
            db=db, eval_run=eval_run, judge_side_effect=_judge
        )

        assert result.status == "completed"
        fake_openai.embeddings.create.assert_not_called()
        traces = _trace_by_ref(result)

        # Good row: ground truth only, no cosine.
        assert _score_named(traces["item-good"], GROUND_TRUTH_SCORE_NAME) is not None
        assert _score_named(traces["item-good"], COSINE_SCORE_NAME) is None

        # Bad row: ground truth dropped, and still no cosine placeholder for a judged run.
        assert _score_named(traces["item-bad"], COSINE_SCORE_NAME) is None
        assert _score_named(traces["item-bad"], GROUND_TRUTH_SCORE_NAME) is None

        assert set(_metric_values(result, GROUND_TRUTH_SCORE_NAME)) == {"item-good"}
        # The failed row is excluded from the aggregate, not counted as a zero.
        assert _summary_named(result, GROUND_TRUTH_SCORE_NAME)["total_pairs"] == 1

        run = db.get(EvaluationRun, result.id)
        assert run.unscoreable.get("item-bad") == JUDGE_FAILED_REASON
        assert run.per_item_scores is None

    def test_judge_cost_stage_tracked(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-16: EvaluationRun.cost gains a single judge stage with tokens + USD."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[
                _resp_result("item-1", "Q1", "golden-1"),
                _resp_result("item-2", "Q2", "golden-2"),
            ],
            store=_s3_store,
        )

        result, _ = _run_pipeline(
            db=db,
            eval_run=eval_run,
            judge_side_effect=lambda _p: _judge_response(
                0.7, "close", usage=(12, 6, 18)
            ),
            mock_cost=True,
        )

        run = db.get(EvaluationRun, result.id)
        stage = run.cost[JUDGE_COST_STAGE]
        # Two rows × (12 in, 6 out, 18 total) summed for the combined judge call.
        assert stage["input_tokens"] == 24
        assert stage["output_tokens"] == 12
        assert stage["total_tokens"] == 36
        # cost_usd from the mocked estimate: (0.001 + 0.002) rounded.
        assert stage["cost_usd"] == pytest.approx(0.003, abs=1e-9)


BOT_INSTRUCTIONS = "You are a farming helpline bot. Always answer in Hindi."


class TestAdherenceToPromptScoring:
    def test_both_metrics_come_from_one_call_per_row(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(
            db=db,
            user_api_key=user_api_key,
            is_judge_run=True,
            instructions=BOT_INSTRUCTIONS,
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[
                _resp_result("item-1", "Q1", "golden-1"),
                _resp_result("item-2", "Q2", "golden-2"),
            ],
            store=_s3_store,
        )

        calls: list[dict[str, Any]] = []

        def _judge(params):
            calls.append(params)
            return _both_metrics_response()

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        assert result.status == "completed"
        # One combined call grades both metrics — two rows must not cost four calls.
        assert len(calls) == 2
        for params in calls:
            assert "Adherence to Ground Truth" in params["instructions"]
            assert "Adherence to Prompt" in params["instructions"]
            assert "ground_truth, prompt" in params["input"]

        traces = _trace_by_ref(result)
        for ref in ("item-1", "item-2"):
            prompt_score = _score_named(traces[ref], PROMPT_SCORE_NAME)
            assert prompt_score is not None
            assert prompt_score["value"] == pytest.approx(0.6, abs=0.01)
            assert prompt_score["comment"] == "answered in the wrong language"
            assert _score_named(traces[ref], GROUND_TRUTH_SCORE_NAME) is not None

        summary_names = {s["name"] for s in result.score["summary_scores"]}
        assert summary_names == {GROUND_TRUTH_SCORE_NAME, PROMPT_SCORE_NAME}

    def test_bot_instructions_reach_the_input_never_the_grader_instructions(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(
            db=db,
            user_api_key=user_api_key,
            is_judge_run=True,
            instructions=BOT_INSTRUCTIONS,
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[_resp_result("item-1", "Q-farming", "golden-1")],
            store=_s3_store,
        )

        captured: dict[str, Any] = {}

        def _judge(params):
            captured.update(params)
            return _both_metrics_response()

        _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        assert (
            f"Assistant's configured instructions:\n{BOT_INSTRUCTIONS}"
            in captured["input"]
        )
        assert "Q-farming" in captured["input"]
        assert "generated for Q-farming" in captured["input"]
        # The evaluated bot's prompt must not become the grader's own system prompt,
        # or a malicious bot prompt could steer its own grade.
        assert BOT_INSTRUCTIONS not in captured["instructions"]

    def test_per_row_prompt_score_and_reasoning_land_in_the_trace_scores(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(
            db=db,
            user_api_key=user_api_key,
            is_judge_run=True,
            instructions=BOT_INSTRUCTIONS,
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[
                _resp_result("item-1", "Q1", "golden-1"),
                _resp_result("item-2", "Q2", "golden-2"),
            ],
            store=_s3_store,
        )

        result, _ = _run_pipeline(
            db=db,
            eval_run=eval_run,
            judge_side_effect=lambda _p: _both_metrics_response(
                ground_truth=(0.9, "same facts"), prompt=(0.25, "answered in English")
            ),
        )

        traces = _trace_by_ref(result)
        for ref in ("item-1", "item-2"):
            prompt_score = _score_named(traces[ref], PROMPT_SCORE_NAME)
            assert prompt_score == {
                "name": PROMPT_SCORE_NAME,
                "value": 0.25,
                "data_type": "NUMERIC",
                "comment": "answered in English",
            }
            assert _score_named(traces[ref], GROUND_TRUTH_SCORE_NAME)["value"] == 0.9

        prompt_summary = _summary_named(result, PROMPT_SCORE_NAME)
        assert prompt_summary["avg"] == 0.25
        assert prompt_summary["total_pairs"] == 2

    def test_prompt_template_is_appended_to_the_config_prompt_block(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(
            db=db,
            user_api_key=user_api_key,
            is_judge_run=True,
            instructions=BOT_INSTRUCTIONS,
            prompt_template="Farmer asks: {{input}}",
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[_resp_result("item-1", "Q1", "golden-1")],
            store=_s3_store,
        )

        captured: dict[str, Any] = {}

        def _judge(params):
            captured.update(params)
            return _both_metrics_response()

        _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        assert BOT_INSTRUCTIONS in captured["input"]
        assert "Farmer asks: {{input}}" in captured["input"]
        assert PROMPT_TEMPLATE_LABEL in captured["input"]

    def test_malformed_reply_drops_both_metrics_for_that_row_only(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(
            db=db,
            user_api_key=user_api_key,
            is_judge_run=True,
            instructions=BOT_INSTRUCTIONS,
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[
                _resp_result("item-good", "Q-good", "golden-good"),
                _resp_result("item-bad", "Q-bad", "golden-bad"),
            ],
            store=_s3_store,
        )

        def _judge(params):
            if "Q-bad" in params["input"]:
                return _raw_judge_response("totally not json")
            return _both_metrics_response()

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        assert result.status == "completed"
        traces = _trace_by_ref(result)
        assert _score_named(traces["item-good"], PROMPT_SCORE_NAME) is not None
        assert _score_named(traces["item-good"], GROUND_TRUTH_SCORE_NAME) is not None
        assert _score_named(traces["item-bad"], PROMPT_SCORE_NAME) is None
        assert _score_named(traces["item-bad"], GROUND_TRUTH_SCORE_NAME) is None

        assert set(_metric_values(result, PROMPT_SCORE_NAME)) == {"item-good"}
        assert set(_metric_values(result, GROUND_TRUTH_SCORE_NAME)) == {"item-good"}

        run = db.get(EvaluationRun, result.id)
        assert run.unscoreable.get("item-bad") == JUDGE_FAILED_REASON


class TestPromptMetricUnscoreable:
    """No resolvable instructions → the prompt metric drops for the whole run."""

    def _assert_only_ground_truth_scored(self, result: EvaluationRun) -> None:
        assert result.status == "completed"
        summary_names = {s["name"] for s in result.score["summary_scores"]}
        assert GROUND_TRUTH_SCORE_NAME in summary_names
        assert PROMPT_SCORE_NAME not in summary_names

        trace = _trace_by_ref(result)["item-1"]
        assert _score_named(trace, PROMPT_SCORE_NAME) is None
        assert _score_named(trace, GROUND_TRUTH_SCORE_NAME)["value"] == 0.8

    def test_config_without_instructions_drops_the_prompt_metric(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[_resp_result("item-1", "Q1", "golden-1")],
            store=_s3_store,
        )

        captured: dict[str, Any] = {}

        def _judge(params):
            captured.update(params)
            return _both_metrics_response()

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        self._assert_only_ground_truth_scored(result)
        # The dropped metric leaves no trace in the judge request either.
        assert "Adherence to Prompt" not in captured["instructions"]
        assert "Assistant's configured instructions" not in captured["input"]

    def test_empty_instructions_drop_the_prompt_metric(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(
            db=db, user_api_key=user_api_key, is_judge_run=True, instructions="   "
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[_resp_result("item-1", "Q1", "golden-1")],
            store=_s3_store,
        )

        result, _ = _run_pipeline(
            db=db,
            eval_run=eval_run,
            # The stub still offers a prompt score; gating, not the reply, drops it.
            judge_side_effect=lambda _p: _both_metrics_response(),
        )

        self._assert_only_ground_truth_scored(result)

    def test_unresolvable_config_version_drops_the_prompt_metric(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        eval_run = _make_run(
            db=db,
            user_api_key=user_api_key,
            is_judge_run=True,
            instructions=BOT_INSTRUCTIONS,
            config_version=999,  # no such version → resolution fails
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[_resp_result("item-1", "Q1", "golden-1")],
            store=_s3_store,
        )

        result, _ = _run_pipeline(
            db=db,
            eval_run=eval_run,
            # The stub still offers a prompt score; gating, not the reply, drops it.
            judge_side_effect=lambda _p: _both_metrics_response(),
        )

        self._assert_only_ground_truth_scored(result)


class TestV1PipelineUnchanged:
    def test_v1_run_produces_no_judge_metrics(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-18: a non-judge run scores cosine only — no judge trace or summary
        scores, and the judge completion is never called. Embeddings run and
        per_item_scores is populated exactly as before."""
        # Instructions resolve, so the prompt metric would be enabled if v1 judged.
        eval_run = _make_run(
            db=db,
            user_api_key=user_api_key,
            is_judge_run=False,
            instructions=BOT_INSTRUCTIONS,
        )
        _seed_chunk(
            db=db,
            eval_run=eval_run,
            results=[_resp_result("item-1", "Q1", "golden-1")],
            store=_s3_store,
        )

        judge_calls: list = []

        def _judge(params):
            judge_calls.append(params)
            return _judge_response(0.9, "should never run")

        result, fake_openai = _run_pipeline(
            db=db, eval_run=eval_run, judge_side_effect=_judge
        )

        assert result.status == "completed"
        assert judge_calls == []
        # v1 embeds every row (cosine input); the pair is identical → cosine ≈ 1.0.
        fake_openai.embeddings.create.assert_called()
        traces = _trace_by_ref(result)
        assert _score_named(traces["item-1"], GROUND_TRUTH_SCORE_NAME) is None
        assert _score_named(traces["item-1"], COSINE_SCORE_NAME) is not None

        assert _metric_values(result, GROUND_TRUTH_SCORE_NAME) == {}
        assert _metric_values(result, PROMPT_SCORE_NAME) == {}

        run = db.get(EvaluationRun, result.id)
        # v1 keeps its durable cosine per-row map.
        assert run.per_item_scores == {"item-1": pytest.approx(1.0, abs=0.01)}
        summary_names = {s["name"] for s in result.score["summary_scores"]}
        assert GROUND_TRUTH_SCORE_NAME not in summary_names
        assert PROMPT_SCORE_NAME not in summary_names
        assert COSINE_SCORE_NAME in summary_names
