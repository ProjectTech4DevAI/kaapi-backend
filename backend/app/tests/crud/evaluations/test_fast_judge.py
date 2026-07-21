"""End-to-end judge scoring on the v2 fast pipeline (`run_fast_evaluation`).

Drives the ground-truth slice of the three-metric SRD through the real fast
pipeline with a judged run (`is_judge_run=True`, `langfuse=None` as v2 dispatches):
FR-2 (a ground-truth trace score in [0,1] + reasoning), FR-9 (zero-config uses
the fallback model + built-in prompt), FR-14 (summary + durable per-row map),
FR-15 (per-row isolation), FR-16 (judge cost stage), FR-18 (v1 never judges).

A v2 judged run drops cosine + embeddings entirely: no embedding API calls, no
"Cosine Similarity" trace/summary score, and `per_item_scores` stays NULL. The
"Adherence to Ground Truth" judge score is the only scorer. v1 (`is_judge_run`
False) is unchanged — cosine only, no judge.

External boundaries mocked: OpenAI (embeddings + the judge completion at
`_create_judge_response`), S3, `save_score`, model/cost resolution. DB is real.
"""

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import openai
import pytest
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.fast import (
    CHUNK_CONFIG_INDEX,
    CHUNK_CONFIG_RUN_ID,
    JOB_TYPE_EVALUATION_FAST_CHUNK,
    _format_top_kb_matches,
    _responses_call_for_item,
    run_fast_evaluation,
    run_response_chunk,
)
from app.crud.evaluations.score import (
    GROUND_TRUTH_SCORE_NAME,
    JUDGE_FAILED_REASON,
    KNOWLEDGE_BASE_SCORE_NAME,
)
from app.models import Config, EvaluationDataset, EvaluationRun
from app.models.batch_job import BatchJob
from app.models.evaluation import RunModeEnum
from app.models.llm.request import ConfigBlob, KaapiCompletionConfig, TextLLMParams
from app.models.response import FileResultChunk
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


def _make_text_config(db: Session, project_id: int) -> Config:
    blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={"model": "gpt-4o-fast-eval-test", "temperature": 0.7},
        )
    )
    return create_test_config(
        db=db, project_id=project_id, use_kaapi_schema=True, config_blob=blob
    )


def _make_run(
    *,
    db: Session,
    user_api_key: TestAuthContext,
    is_judge_run: bool,
) -> EvaluationRun:
    dataset = _make_dataset(db=db, user_api_key=user_api_key)
    config = _make_text_config(db, user_api_key.project_id)
    run = EvaluationRun(
        run_name=f"run-{random_lower_string()}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=1,
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


class TestGroundTruthScoring:
    def test_ground_truth_score_is_the_only_scorer_no_cosine(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-2/FR-14 + v2 contract: each row carries ONLY a ground-truth score (no
        cosine); summary + durable per-row map populated; per_item_scores stays NULL
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

        run = db.get(EvaluationRun, result.id)
        assert run.per_item_ground_truth == {"item-1": 0.8, "item-2": 0.8}
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

        run = db.get(EvaluationRun, result.id)
        assert run.unscoreable.get("item-bad") == JUDGE_FAILED_REASON
        assert set(run.per_item_ground_truth) == {"item-good"}
        assert run.per_item_scores is None

    def test_judge_cost_stage_tracked(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-16: EvaluationRun.cost gains a ground_truth_judge stage with tokens + USD."""
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
        stage = run.cost["ground_truth_judge"]
        # Two rows × (12 in, 6 out, 18 total) summed for the combined judge call.
        assert stage["input_tokens"] == 24
        assert stage["output_tokens"] == 12
        assert stage["total_tokens"] == 36
        # cost_usd from the mocked estimate: (0.001 + 0.002) rounded.
        assert stage["cost_usd"] == pytest.approx(0.003, abs=1e-9)


class TestV1PipelineUnchanged:
    def test_v1_run_produces_no_judge_metrics(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """FR-18: a non-judge run scores cosine only — no ground-truth score, no
        per-row ground-truth map, and the judge completion is never called. Embeddings
        run and per_item_scores is populated exactly as before."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=False)
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

        run = db.get(EvaluationRun, result.id)
        assert run.per_item_ground_truth is None
        # v1 keeps its durable cosine per-row map.
        assert run.per_item_scores == {"item-1": pytest.approx(1.0, abs=0.01)}
        summary_names = {s["name"] for s in result.score["summary_scores"]}
        assert GROUND_TRUTH_SCORE_NAME not in summary_names
        assert COSINE_SCORE_NAME in summary_names


def _responses_item(item_id: str = "item-1") -> dict[str, Any]:
    return {
        "id": item_id,
        "input": {"question": "What is X?"},
        "expected_output": {"answer": "golden"},
        "metadata": {"question_id": 1},
    }


def _openai_response():
    return SimpleNamespace(
        output_text="generated answer",
        output=[],
        id="resp_1",
        usage=SimpleNamespace(input_tokens=5, output_tokens=5, total_tokens=10),
    )


class TestResponsesChunkCapture:
    """`_responses_call_for_item` flattens file_search hits into JSON-safe dicts."""

    def test_success_with_chunks_returns_serializable_plain_dicts(self):
        client = MagicMock()
        client.responses.create.return_value = _openai_response()
        with patch(
            "app.crud.evaluations.fast.get_file_search_results",
            return_value=[
                FileResultChunk(score=0.91, text="chunk A", filename="doc.pdf"),
                FileResultChunk(score=0.42, text="chunk B"),
            ],
        ):
            result = _responses_call_for_item(
                openai_client=client,
                base_params={"model": "gpt-4o"},
                item=_responses_item(),
            )

        assert result["failed"] is False
        # filename flows into the persisted unit so knowledge_base can name its matches.
        assert result["retrieved_chunks"] == [
            {"score": 0.91, "text": "chunk A", "filename": "doc.pdf"},
            {"score": 0.42, "text": "chunk B", "filename": None},
        ]
        json.dumps(result)  # the S3 unit must stay JSON-serializable

    def test_success_without_hits_returns_empty_chunks(self):
        client = MagicMock()
        client.responses.create.return_value = _openai_response()
        with patch(
            "app.crud.evaluations.fast.get_file_search_results", return_value=[]
        ):
            result = _responses_call_for_item(
                openai_client=client,
                base_params={"model": "gpt-4o"},
                item=_responses_item(),
            )

        assert result["failed"] is False
        assert result["retrieved_chunks"] == []

    def test_error_path_has_no_chunks(self):
        client = MagicMock()
        client.responses.create.side_effect = openai.OpenAIError("provider down")
        with patch("app.crud.evaluations.fast.get_file_search_results") as fake_search:
            result = _responses_call_for_item(
                openai_client=client,
                base_params={"model": "gpt-4o"},
                item=_responses_item(),
            )

        assert result["failed"] is True
        assert result["retrieved_chunks"] is None
        fake_search.assert_not_called()


class TestKnowledgeBaseScoring:
    def test_groundedness_scores_chunked_row_and_flags_chunkless_row_na(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """A chunked row gets a numeric KB score; a chunk-less-but-judged row gets an
        N/A KB placeholder (not a numeric score, not absent), while ground truth still
        scores every judged row."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        chunked = _resp_result("item-chunked", "Q1", "golden-1")
        chunked["retrieved_chunks"] = [{"score": 0.9, "text": "supporting chunk"}]
        plain = _resp_result("item-plain", "Q2", "golden-2")  # no retrieved_chunks
        _seed_chunk(db=db, eval_run=eval_run, results=[chunked, plain], store=_s3_store)

        def _judge(params):
            if KNOWLEDGE_BASE_SCORE_NAME in params["instructions"]:
                return _raw_judge_response(
                    json.dumps(
                        {
                            "ground_truth": {"score": 0.8, "reasoning": "gt"},
                            "knowledge_base": {"score": 0.6, "reasoning": "kb"},
                        }
                    )
                )
            return _judge_response(0.9, "gt only")

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        assert result.status == "completed"
        run = db.get(EvaluationRun, result.id)
        # ground_truth keeps its durable per-row column; knowledge_base has none —
        # its per-row score lives only in the score_trace_url trace unit below.
        assert set(run.per_item_ground_truth) == {"item-chunked", "item-plain"}

        traces = _trace_by_ref(result)
        kb_chunked = _score_named(traces["item-chunked"], KNOWLEDGE_BASE_SCORE_NAME)
        assert kb_chunked is not None
        assert kb_chunked["value"] == 0.6

        # The judged-but-chunkless row surfaces a human N/A placeholder that stays out
        # of the summary avg (it is not a numeric 0).
        kb_plain = _score_named(traces["item-plain"], KNOWLEDGE_BASE_SCORE_NAME)
        assert kb_plain["value"] == "N/A"
        assert kb_plain["data_type"] == "CATEGORICAL"
        assert kb_plain["unscoreable"] is True
        assert kb_plain["comment"] == "Knowledge base not queried."
        assert _score_named(traces["item-plain"], GROUND_TRUTH_SCORE_NAME) is not None

    def test_kb_scored_row_comment_names_top_matches(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """A scored KB row appends the top retrieved filenames to its comment."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        chunked = _resp_result("item-1", "Q1", "golden-1")
        chunked["retrieved_chunks"] = [
            {"score": 0.906, "text": "supporting chunk", "filename": "biu-1.pdf"},
            {"score": 0.42, "text": "weak chunk", "filename": "faq.pdf"},
        ]
        _seed_chunk(db=db, eval_run=eval_run, results=[chunked], store=_s3_store)

        def _judge(params):
            if KNOWLEDGE_BASE_SCORE_NAME in params["instructions"]:
                return _raw_judge_response(
                    json.dumps(
                        {
                            "ground_truth": {"score": 0.8, "reasoning": "gt"},
                            "knowledge_base": {"score": 0.7, "reasoning": "grounded"},
                        }
                    )
                )
            return _judge_response(0.9, "gt only")

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        kb = _score_named(_trace_by_ref(result)["item-1"], KNOWLEDGE_BASE_SCORE_NAME)
        assert kb["data_type"] == "NUMERIC"
        assert kb["value"] == 0.7
        # No relevance gate: every retrieved chunk names a match, low scores included.
        assert (
            kb["comment"]
            == "grounded | Top matches: biu-1.pdf (90.6%), faq.pdf (42.0%)"
        )

    def test_kb_scored_even_when_all_chunks_low_score(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """Gate removed: any retrieved chunk is judged, however weak its score."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        row = _resp_result("item-1", "Q1", "golden-1")
        row["retrieved_chunks"] = [
            {"score": 0.55, "text": "weak but relevant", "filename": "a.pdf"},
            {"score": 0.30, "text": "weaker", "filename": "b.pdf"},
        ]
        _seed_chunk(db=db, eval_run=eval_run, results=[row], store=_s3_store)

        def _judge(params):
            if KNOWLEDGE_BASE_SCORE_NAME in params["instructions"]:
                return _raw_judge_response(
                    json.dumps(
                        {
                            "ground_truth": {"score": 0.9, "reasoning": "gt"},
                            "knowledge_base": {"score": 0.6, "reasoning": "partial"},
                        }
                    )
                )
            return _judge_response(0.9, "gt only")

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        kb = _score_named(_trace_by_ref(result)["item-1"], KNOWLEDGE_BASE_SCORE_NAME)
        assert kb["data_type"] == "NUMERIC"
        assert kb["value"] == 0.6
        assert kb["comment"] == "partial | Top matches: a.pdf (55.0%), b.pdf (30.0%)"

    def test_kb_na_placeholder_stays_out_of_summary_avg(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """The KB summary avg reflects only genuinely-scored rows, never the N/A rows."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        scored = _resp_result("item-scored", "Q1", "golden-1")
        scored["retrieved_chunks"] = [
            {"score": 0.9, "text": "supporting", "filename": "kb.pdf"}
        ]
        dropped = _resp_result("item-dropped", "Q2", "golden-2")  # no chunks → N/A
        _seed_chunk(
            db=db, eval_run=eval_run, results=[scored, dropped], store=_s3_store
        )

        def _judge(params):
            if KNOWLEDGE_BASE_SCORE_NAME in params["instructions"]:
                return _raw_judge_response(
                    json.dumps(
                        {
                            "ground_truth": {"score": 0.8, "reasoning": "gt"},
                            "knowledge_base": {"score": 0.6, "reasoning": "grounded"},
                        }
                    )
                )
            return _judge_response(0.9, "gt only")

        result, _ = _run_pipeline(db=db, eval_run=eval_run, judge_side_effect=_judge)

        kb_summary = next(
            s
            for s in result.score["summary_scores"]
            if s["name"] == KNOWLEDGE_BASE_SCORE_NAME
        )
        # Only item-scored (0.6) is a real KB score; the N/A row never enters the avg.
        assert kb_summary["avg"] == 0.6
        assert kb_summary["total_pairs"] == 1

    def test_non_kb_metric_none_is_skipped_not_placeholdered(
        self, db: Session, user_api_key: TestAuthContext, _s3_store
    ):
        """A missing non-KB metric (ground_truth) yields no trace score at all — the
        N/A placeholder is a KB-only affordance."""
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        row = _resp_result("item-1", "Q1", "golden-1")
        row["retrieved_chunks"] = [
            {"score": 0.9, "text": "supporting", "filename": "kb.pdf"}
        ]
        _seed_chunk(db=db, eval_run=eval_run, results=[row], store=_s3_store)

        # Judge returns knowledge_base only; ground_truth is silently dropped in parsing.
        result, _ = _run_pipeline(
            db=db,
            eval_run=eval_run,
            judge_side_effect=lambda _p: _raw_judge_response(
                json.dumps({"knowledge_base": {"score": 0.7, "reasoning": "grounded"}})
            ),
        )

        trace = _trace_by_ref(result)["item-1"]
        assert _score_named(trace, KNOWLEDGE_BASE_SCORE_NAME) is not None
        # No ground_truth score is emitted — not even an N/A placeholder.
        assert _score_named(trace, GROUND_TRUTH_SCORE_NAME) is None


class TestFormatTopKbMatches:
    """`_format_top_kb_matches` — the human 'Top matches: ...' string for KB comments."""

    def test_formats_filename_and_percent_to_one_decimal(self) -> None:
        result = _format_top_kb_matches(
            [
                {"filename": "biu-1.pdf", "score": 0.906},
                {"filename": "faq.pdf", "score": 0.663},
            ]
        )
        assert result == "biu-1.pdf (90.6%), faq.pdf (66.3%)"

    def test_includes_all_chunks_regardless_of_score(self) -> None:
        result = _format_top_kb_matches(
            [
                {"filename": "hi.pdf", "score": 0.9},
                {"filename": "lo.pdf", "score": 0.5},
            ]
        )
        assert result == "hi.pdf (90.0%), lo.pdf (50.0%)"

    def test_caps_at_three_matches(self) -> None:
        chunks = [{"filename": f"f{i}.pdf", "score": 0.9 - i * 0.01} for i in range(5)]
        result = _format_top_kb_matches(chunks)
        assert result == "f0.pdf (90.0%), f1.pdf (89.0%), f2.pdf (88.0%)"

    def test_missing_filename_renders_unknown(self) -> None:
        assert _format_top_kb_matches([{"score": 0.9}]) == "unknown (90.0%)"
        assert (
            _format_top_kb_matches([{"filename": None, "score": 0.8}])
            == "unknown (80.0%)"
        )

    def test_empty_input_is_empty_string(self) -> None:
        assert _format_top_kb_matches([]) == ""


class TestFileSearchIncludeParam:
    """`run_response_chunk` requests file_search hits only when a file_search tool is present,
    and never overrides tool_choice (stays at the model default / auto)."""

    def _run_and_capture_base_params(
        self, *, db: Session, eval_run: EvaluationRun, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def _fake_call(*, openai_client, base_params, item):
            captured.update(base_params)
            return {"item_id": item["id"], "failed": False, "usage": {}}

        with (
            patch(
                "app.crud.evaluations.fast.map_kaapi_to_openai_params",
                return_value=({"model": "gpt-4o", "tools": tools}, []),
            ),
            patch(
                "app.crud.evaluations.fast._responses_call_for_item",
                side_effect=_fake_call,
            ),
            patch(
                "app.crud.evaluations.fast._upload_unit_to_s3",
                return_value="s3://bucket/chunk.json",
            ),
        ):
            run_response_chunk(
                session=db,
                openai_client=MagicMock(),
                eval_run=eval_run,
                config=TextLLMParams(model="gpt-4o"),
                dataset_items_slice=[{"id": "item-1"}],
                chunk_index=0,
                log_prefix="[test]",
            )
        return captured

    def test_file_search_present_sets_include_and_leaves_tool_choice_default(
        self, db: Session, user_api_key: TestAuthContext
    ):
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        base_params = self._run_and_capture_base_params(
            db=db, eval_run=eval_run, tools=[{"type": "file_search"}]
        )
        assert base_params["include"] == ["file_search_call.results"]
        assert "tool_choice" not in base_params

    def test_no_file_search_leaves_include_and_tool_choice_unset(
        self, db: Session, user_api_key: TestAuthContext
    ):
        eval_run = _make_run(db=db, user_api_key=user_api_key, is_judge_run=True)
        base_params = self._run_and_capture_base_params(
            db=db, eval_run=eval_run, tools=[]
        )
        assert "include" not in base_params
        assert "tool_choice" not in base_params
        assert "include" not in base_params
