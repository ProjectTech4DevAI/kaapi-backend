"""Tests for the fast (synchronous) evaluation path.

Covers FR-1 through FR-15 from the Fast Evaluation SRD plus the chunk/aggregate
split of the responses stage. External boundaries (OpenAI, Langfuse, S3, Celery
dispatch) are mocked; the DB is real (`db` fixture).
"""

import random
from collections.abc import Iterator
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import openai
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.core.util import now
from app.crud.evaluations.cron import dispatch_fast_evaluation_barriers
from app.crud.evaluations.fast import (
    CHUNK_CONFIG_INDEX,
    CHUNK_CONFIG_RUN_ID,
    JOB_TYPE_EMBEDDING_FAST,
    JOB_TYPE_EVALUATION_FAST,
    JOB_TYPE_EVALUATION_FAST_CHUNK,
    _create_response,
    _get_chunk_job,
    _is_failure_threshold_breached,
    _merge_response_chunks,
    _stage2_embeddings,
    _stage3_score_and_trace,
    list_response_chunk_jobs,
    run_fast_evaluation,
    run_response_chunk,
)
from app.models import Config, EvaluationDataset, EvaluationRun
from app.models.batch_job import BatchJob
from app.models.evaluation import RunModeEnum
from app.models.llm.request import (
    ConfigBlob,
    TextLLMParams,
    build_kaapi_completion_config,
)
from app.services.evaluations.fast import (
    execute_fast_evaluation_chunk,
    validate_and_start_fast_evaluation,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)
from app.tests.utils.utils import random_lower_string


def _api_error(resp_body: dict) -> str:
    """Pull the human-readable error string out of an APIResponse failure body."""
    return str(resp_body.get("error") or resp_body.get("detail") or resp_body).lower()


@pytest.fixture(autouse=True)
def _seeded_random() -> Iterator[None]:
    """Make jitter / random.choice deterministic so tests are repeatable."""
    random.seed(0)
    yield


# Pure-function helpers (no FastAPI, no DB)


class TestFailureThreshold:
    """`_is_failure_threshold_breached` controls run-level fail-fast."""

    def test_returns_false_when_total_is_zero(self) -> None:
        assert _is_failure_threshold_breached(failed_rows=0, total_rows=0) is False

    def test_returns_true_above_threshold(self) -> None:
        # default EVAL_FAST_FAILURE_THRESHOLD = 0.5
        assert _is_failure_threshold_breached(failed_rows=6, total_rows=10) is True

    def test_returns_false_at_threshold(self) -> None:
        # 0.5 / 1.0 is NOT greater-than the threshold, so do not breach
        assert _is_failure_threshold_breached(failed_rows=5, total_rows=10) is False


class TestCallWithRetry:
    """FR-8: transient OpenAI errors retry; permanent ones do not."""

    def test_returns_immediately_on_success(self) -> None:
        client = MagicMock()
        client.responses.create.return_value = "ok"

        result = _create_response(client, {"model": "gpt-4o"})

        assert result == "ok"
        assert client.responses.create.call_count == 1

    def test_retries_on_transient_then_succeeds(self, monkeypatch) -> None:
        # tenacity sleeps via tenacity.nap.sleep — make backoff a no-op.
        monkeypatch.setattr("tenacity.nap.sleep", lambda *_: None)

        client = MagicMock()
        client.responses.create.side_effect = [
            # APIConnectionError needs a request; pass a minimal object.
            openai.APIConnectionError(request=MagicMock()),
            openai.APIConnectionError(request=MagicMock()),
            "ok",
        ]

        result = _create_response(client, {"model": "gpt-4o"})

        assert result == "ok"
        assert client.responses.create.call_count == 3

    def test_does_not_retry_on_permanent_error(self) -> None:
        client = MagicMock()
        # AuthenticationError is a non-retryable OpenAIError subclass.
        client.responses.create.side_effect = openai.AuthenticationError(
            message="bad key", response=MagicMock(), body=None
        )

        with pytest.raises(openai.AuthenticationError):
            _create_response(client, {"model": "gpt-4o"})

        assert client.responses.create.call_count == 1


# Shared factories + mock boundaries


def _make_fast_eligible_dataset(
    *,
    db: Session,
    user_api_key: TestAuthContext,
    original_items_count: int = 3,
) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        original_items_count=original_items_count,
        duplication_factor=1,
    )


def _make_text_openai_config(db: Session, project_id: int) -> Config:
    """Create a stored text-OpenAI Kaapi config (eligible for fast eval).

    The model name is intentionally absent from `model_config` so blob
    validation short-circuits before loading a ModelConfig row (whose
    ARRAY-of-enum column trips a known SQLAlchemy result-processor bug in some
    environments). Fast-eval eligibility only inspects provider + type, so the
    exact model name is immaterial to every assertion here.
    """
    blob = ConfigBlob(
        completion=build_kaapi_completion_config(
            provider="openai",
            type="text",
            params={"model": "gpt-4o-fast-eval-test", "temperature": 0.7},
        )
    )
    return create_test_config(
        db=db,
        project_id=project_id,
        use_kaapi_schema=True,
        config_blob=blob,
    )


def _make_fast_run(
    *,
    db: Session,
    user_api_key: TestAuthContext,
    status: str = "processing",
    total_items: int = 0,
    run_name: str | None = None,
    batch_job_id: int | None = None,
) -> EvaluationRun:
    """Persist a fast-mode EvaluationRun with a real dataset + config."""
    dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
    config = _make_text_openai_config(db, user_api_key.project_id)
    run = EvaluationRun(
        run_name=run_name or f"run-{random_lower_string()}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=1,
        status=status,
        run_mode=RunModeEnum.FAST.value,
        total_items=total_items,
        batch_job_id=batch_job_id,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _dataset_item(
    item_id: str, question: str = "Q", answer: str = "A"
) -> dict[str, Any]:
    return {
        "id": item_id,
        "input": {"question": question},
        "expected_output": {"answer": answer},
        "metadata": {"question_id": item_id},
    }


def _resp_result(
    item_id: str,
    question: str = "Q",
    ground_truth: str = "A",
    *,
    failed: bool = False,
    question_id: Any = None,
) -> dict[str, Any]:
    """A Stage-1 per-item result in the shape a chunk partial persists."""
    return {
        "item_id": item_id,
        "question": question,
        "generated_output": "" if failed else f"answer to {question}",
        "ground_truth": ground_truth,
        "response_id": f"resp_{item_id}",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "question_id": question_id,
        "failed": failed,
    }


@pytest.fixture
def _s3_store() -> Iterator[dict[str, list[dict[str, Any]]]]:
    """Back the S3 upload/load edge with an in-memory dict keyed by url.

    Both the chunk partials and the merged canonical unit round-trip through
    this store so `_merge_response_chunks` and the retry-reload path exercise
    real concatenation instead of a stubbed return value.
    """
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


def _seed_response_chunk(
    *,
    db: Session,
    eval_run: EvaluationRun,
    chunk_index: int,
    results: list[dict[str, Any]],
    store: dict[str, list[dict[str, Any]]],
    model: str = "gpt-4o",
    raw_output_url: str | None = "__auto__",
) -> BatchJob:
    """Create a chunk BatchJob (+ its S3 partial) as a completed chunk would."""
    if raw_output_url == "__auto__":
        raw_output_url = f"s3://bucket/responses_{eval_run.id}_{chunk_index}.json"
        store[raw_output_url] = results
    job = BatchJob(
        provider="openai",
        job_type=JOB_TYPE_EVALUATION_FAST_CHUNK,
        config={
            "model": model,
            CHUNK_CONFIG_RUN_ID: eval_run.id,
            CHUNK_CONFIG_INDEX: chunk_index,
        },
        raw_output_url=raw_output_url,
        total_items=len(results),
        organization_id=eval_run.organization_id,
        project_id=eval_run.project_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


class _FakeSessionCtx:
    """Context manager returning the test session; `__exit__` never closes it."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def __enter__(self) -> Session:
        return self._db

    def __exit__(self, *exc: object) -> bool:
        return False


# Route validation: POST /evaluations with run_mode=fast (FR-1..FR-5, FR-15)


@pytest.fixture
def _patch_dispatch() -> Iterator[MagicMock]:
    """Stub the synchronous request-path boundaries of `validate_and_start`.

    The route now fetches the dataset in-request to size the fan-out, so the
    Langfuse client + `fetch_dataset_items` must be stubbed alongside the chunk
    enqueue. Yields the chunk-enqueue mock (call count == number of chunks).
    """
    with (
        patch("app.services.evaluations.fast.get_langfuse_client"),
        patch(
            "app.services.evaluations.fast.fetch_dataset_items",
            return_value=[_dataset_item(f"item-{i}") for i in range(3)],
        ),
        patch(
            "app.services.evaluations.fast.start_fast_evaluation_chunk",
            return_value="fake-task-id",
        ) as mock_start_chunk,
    ):
        yield mock_start_chunk


class TestFastEvaluationRoute:
    """End-to-end validation on POST /evaluations with run_mode='fast'."""

    def test_fr4_accepts_eligible_request_and_dispatches(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-4: eligible request returns processing + dispatches chunk task."""
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)

        resp = client.post(
            "/api/v1/evaluations",
            json={
                "experiment_name": "fr4-fast-run",
                "dataset_id": dataset.id,
                "config_id": str(config.id),
                "config_version": 1,
                "run_mode": "fast",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["run_mode"] == "fast"
        assert body["status"] == "processing"
        assert body["run_name"] == "fr4-fast-run"
        # 3 items / chunk-size 50 = 1 chunk.
        _patch_dispatch.assert_called_once()

        run = db.get(EvaluationRun, body["id"])
        assert run is not None
        assert run.run_mode == RunModeEnum.FAST
        assert run.status == "processing"

    def test_fr2_rejects_dataset_with_too_many_unique_rows(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-2: >100 unique rows → 422 dataset_too_large_for_fast."""
        dataset = _make_fast_eligible_dataset(
            db=db, user_api_key=user_api_key, original_items_count=101
        )
        config = _make_text_openai_config(db, user_api_key.project_id)

        resp = client.post(
            "/api/v1/evaluations",
            json={
                "experiment_name": "fr2-fast-run",
                "dataset_id": dataset.id,
                "config_id": str(config.id),
                "config_version": 1,
                "run_mode": "fast",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 422
        error_str = _api_error(resp.json())
        assert "dataset_too_large_for_fast" in error_str
        assert "101" in error_str
        _patch_dispatch.assert_not_called()

    def test_fr1_rejects_non_text_config(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-1: non-text config for fast mode → 422 config_type_unsupported."""
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)

        fake_blob = ConfigBlob(
            completion=build_kaapi_completion_config(
                provider="openai",
                type="stt",
                params={"model": "whisper-1"},
            )
        )

        with patch(
            "app.services.evaluations.fast.resolve_evaluation_config",
            return_value=(fake_blob, None),
        ):
            resp = client.post(
                "/api/v1/evaluations",
                json={
                    "experiment_name": "fr1-fast-run",
                    "dataset_id": dataset.id,
                    "config_id": str(config.id),
                    "config_version": 1,
                    "run_mode": "fast",
                },
                headers=user_api_key_header,
            )

        assert resp.status_code == 422
        assert "config_type_unsupported" in _api_error(resp.json())
        _patch_dispatch.assert_not_called()

    def test_fr3_rejects_duplicate_run_name(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-3: duplicate (org, project, run_name) → 409, no second dispatch."""
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        payload = {
            "experiment_name": "fr3-dup-run",
            "dataset_id": dataset.id,
            "config_id": str(config.id),
            "config_version": 1,
            "run_mode": "fast",
        }

        first = client.post(
            "/api/v1/evaluations", json=payload, headers=user_api_key_header
        )
        assert first.status_code == 200, first.text

        second = client.post(
            "/api/v1/evaluations", json=payload, headers=user_api_key_header
        )
        assert second.status_code == 409
        assert "run_name_already_exists" in _api_error(second.json())
        # First call dispatched its one chunk; second must not have.
        assert _patch_dispatch.call_count == 1

    def test_fr15_get_evaluation_returns_run_mode(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """FR-15: GET /evaluations/{id} surfaces `run_mode` for both modes."""
        eval_run = _make_fast_run(
            db=db,
            user_api_key=user_api_key,
            status="completed",
            total_items=3,
            run_name="fr15-existing",
        )

        resp = client.get(
            f"/api/v1/evaluations/{eval_run.id}", headers=user_api_key_header
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["run_mode"] == "fast"


# Dataset listing eligibility filter (FR-5)


class TestDatasetListEligibleForFast:
    def test_fr5_filters_to_fast_eligible_only(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """FR-5: eligible_for_fast=true filters list to eligible datasets."""
        eligible = _make_fast_eligible_dataset(
            db=db, user_api_key=user_api_key, original_items_count=5
        )
        ineligible = _make_fast_eligible_dataset(
            db=db, user_api_key=user_api_key, original_items_count=200
        )

        resp = client.get(
            "/api/v1/evaluations/datasets",
            params={"eligible_for_fast": "true"},
            headers=user_api_key_header,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        ids = {d["dataset_id"] for d in data}
        assert eligible.id in ids
        assert ineligible.id not in ids
        assert all(d["eligible_for_fast"] is True for d in data)


# Response fixtures for the OpenAI SDK shapes


def _fake_openai_response(text: str = "answer", item_id: str = "item-1"):
    """Mimic the SDK's response.responses.create return shape."""
    return SimpleNamespace(
        id=f"resp_{item_id}",
        output_text=text,
        output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
    )


def _fake_embedding_response():
    """Mimic openai.embeddings.create return shape (2 identical vectors)."""
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            SimpleNamespace(index=1, embedding=[1.0, 0.0, 0.0]),
        ],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )


# run_response_chunk: one parallel responses chunk + idempotency


class TestRunResponseChunk:
    def test_writes_chunk_job_and_partial_unit(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
    ):
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key)
        items = [_dataset_item("item-1", "Q1"), _dataset_item("item-2", "Q2")]

        fake_openai = MagicMock()
        fake_openai.responses.create.side_effect = lambda **_: _fake_openai_response()

        with patch(
            "app.crud.evaluations.fast.map_kaapi_to_openai_params",
            return_value=({"model": "gpt-4o"}, []),
        ):
            run_response_chunk(
                session=db,
                openai_client=fake_openai,
                eval_run=eval_run,
                config=TextLLMParams(model="gpt-4o", instructions="x"),
                dataset_items_slice=items,
                chunk_index=0,
                log_prefix="[t]",
            )

        assert fake_openai.responses.create.call_count == 2

        job = _get_chunk_job(session=db, eval_run_id=eval_run.id, chunk_index=0)
        assert job is not None
        assert job.job_type == JOB_TYPE_EVALUATION_FAST_CHUNK
        assert job.config[CHUNK_CONFIG_RUN_ID] == eval_run.id
        assert job.config[CHUNK_CONFIG_INDEX] == 0
        assert job.raw_output_url == f"s3://bucket/responses_{eval_run.id}_0.json"
        assert len(_s3_store[job.raw_output_url]) == 2

    def test_idempotent_skips_openai_when_chunk_already_done(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
    ):
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key)
        items = [_dataset_item("item-1", "Q1"), _dataset_item("item-2", "Q2")]

        fake_openai = MagicMock()
        fake_openai.responses.create.side_effect = lambda **_: _fake_openai_response()

        kwargs = {
            "session": db,
            "openai_client": fake_openai,
            "eval_run": eval_run,
            "config": TextLLMParams(model="gpt-4o", instructions="x"),
            "dataset_items_slice": items,
            "chunk_index": 0,
            "log_prefix": "[t]",
        }
        with patch(
            "app.crud.evaluations.fast.map_kaapi_to_openai_params",
            return_value=({"model": "gpt-4o"}, []),
        ):
            run_response_chunk(**kwargs)
            assert fake_openai.responses.create.call_count == 2

            # Second run for the same (run, index) must not re-charge OpenAI.
            run_response_chunk(**kwargs)
            assert fake_openai.responses.create.call_count == 2

        jobs = list_response_chunk_jobs(session=db, eval_run_id=eval_run.id)
        assert len([j for j in jobs if j.config[CHUNK_CONFIG_INDEX] == 0]) == 1


# _merge_response_chunks: concat in index order, dedup, retry-reload skip


class TestMergeResponseChunks:
    def test_concatenates_in_index_order_dedups_and_sets_batch_job_id(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
    ):
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key)
        # Seed out of order: index 1 first, then 0, plus a duplicate index-1 row.
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=1,
            results=[_resp_result("item-b", "Qb")],
            store=_s3_store,
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[_resp_result("item-a", "Qa")],
            store=_s3_store,
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=1,
            results=[_resp_result("item-b", "Qb")],
            store=_s3_store,
            raw_output_url="s3://bucket/dup_chunk_1.json",
        )
        _s3_store["s3://bucket/dup_chunk_1.json"] = [_resp_result("item-b", "Qb")]

        merged_run, results = _merge_response_chunks(session=db, eval_run=eval_run)

        assert [r["item_id"] for r in results] == ["item-a", "item-b"]
        assert merged_run.batch_job_id is not None
        canonical = db.get(BatchJob, merged_run.batch_job_id)
        assert canonical.job_type == JOB_TYPE_EVALUATION_FAST
        assert canonical.raw_output_url == f"s3://bucket/responses_{eval_run.id}.json"
        assert _s3_store[canonical.raw_output_url] == results
        assert merged_run.total_items == 2

    def test_retry_reloads_canonical_when_batch_job_id_set(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
    ):
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key)
        canonical_unit = [_resp_result("item-1", "Q1"), _resp_result("item-2", "Q2")]
        canonical_url = f"s3://bucket/responses_{eval_run.id}.json"
        _s3_store[canonical_url] = canonical_unit
        canonical_job = BatchJob(
            provider="openai",
            job_type=JOB_TYPE_EVALUATION_FAST,
            config={"endpoint": "/v1/responses"},
            raw_output_url=canonical_url,
            total_items=2,
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
        )
        db.add(canonical_job)
        db.commit()
        db.refresh(canonical_job)
        eval_run.batch_job_id = canonical_job.id
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        merged_run, results = _merge_response_chunks(session=db, eval_run=eval_run)

        # Reloads the canonical unit; no re-merge, no new canonical job.
        assert results == canonical_unit
        assert merged_run.batch_job_id == canonical_job.id
        canonical_rows = db.exec(
            select(BatchJob).where(
                BatchJob.project_id == eval_run.project_id,
                BatchJob.job_type == JOB_TYPE_EVALUATION_FAST,
            )
        ).all()
        assert len(canonical_rows) == 1


# Stage skipping on retry (FR-7 — embeddings)


class TestStageSkipping:
    def test_fr7_stage2_skips_when_embedding_batch_job_id_already_set(
        self,
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """Pre-existing embedding batch_job → Stage 2 does not call Embeddings API."""
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key)

        marker = BatchJob(
            provider="openai",
            job_type=JOB_TYPE_EMBEDDING_FAST,
            config={},
            raw_output_url="s3://bucket/embeddings_x.json",
            total_items=1,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(marker)
        db.commit()
        db.refresh(marker)

        eval_run.embedding_batch_job_id = marker.id
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        cached = [
            {
                "item_id": "item-1",
                "output_embedding": [1.0, 0.0],
                "ground_truth_embedding": [1.0, 0.0],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "failed": False,
            }
        ]

        fake_openai = MagicMock()
        with patch("app.crud.evaluations.fast._load_unit_from_s3", return_value=cached):
            _, results = _stage2_embeddings(
                session=db,
                openai_client=fake_openai,
                eval_run=eval_run,
                response_results=[
                    {
                        "item_id": "item-1",
                        "question": "q",
                        "generated_output": "a",
                        "ground_truth": "a",
                        "failed": False,
                    }
                ],
                log_prefix="[t]",
            )

        assert results == cached
        fake_openai.embeddings.create.assert_not_called()


# End-to-end aggregate pipeline with mocked externals (FR-9..FR-14)


@pytest.fixture
def _fast_pipeline_mocks(_s3_store):
    """Patch the Langfuse / cost / model boundaries inside `run_fast_evaluation`.

    Response data now comes from merged chunk partials (`_s3_store`), so the
    pipeline no longer fetches the dataset; the test seeds the chunks instead.
    """
    with (
        patch(
            "app.crud.evaluations.fast.create_langfuse_dataset_run"
        ) as mock_create_traces,
        patch(
            "app.crud.evaluations.fast.update_traces_with_cosine_scores"
        ) as mock_update_traces,
        patch(
            "app.crud.evaluations.fast.resolve_model_from_config", return_value="gpt-4o"
        ),
        patch("app.crud.evaluations.fast.attach_cost") as mock_attach_cost,
    ):
        mock_create_traces.return_value = {
            "item-1": "trace-1",
            "item-2": "trace-2",
        }
        yield SimpleNamespace(
            store=_s3_store,
            create_traces=mock_create_traces,
            update_traces=mock_update_traces,
            attach_cost=mock_attach_cost,
        )


class TestFastPipelineEndToEnd:
    def test_pipeline_completes_with_scores_and_writes_batch_jobs(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _fast_pipeline_mocks,
    ):
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key, status="pending")
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[
                _resp_result("item-1", "Q1", "A1", question_id=1),
                _resp_result("item-2", "Q2", "A2", question_id=2),
            ],
            store=_fast_pipeline_mocks.store,
        )

        fake_openai = MagicMock()
        fake_openai.embeddings.create.return_value = _fake_embedding_response()
        fake_langfuse = MagicMock()

        # save_score opens its own Session(engine), invisible to this test's
        # rolled-back transaction. Mirror its S3-success path against the test
        # session so the persisted state is observable here.
        def _fake_save_score(*, eval_run_id, score, **_):
            run = db.get(EvaluationRun, eval_run_id)
            run.score = {"summary_scores": score["summary_scores"]}
            run.score_trace_url = f"s3://bucket/traces_{eval_run_id}.json"
            db.add(run)
            db.commit()
            db.refresh(run)
            return run

        with patch(
            "app.crud.evaluations.fast.save_score", side_effect=_fake_save_score
        ):
            result = run_fast_evaluation(
                session=db,
                openai_client=fake_openai,
                langfuse=fake_langfuse,
                eval_run=eval_run,
            )

        # FR-11/FR-14: completed, summary cosine ≈ 1.0 for identical vectors.
        assert result.status == "completed"
        assert result.score is not None
        cosine = result.score["summary_scores"][0]
        assert cosine["name"] == "Cosine Similarity"
        assert cosine["avg"] == pytest.approx(1.0, abs=0.01)
        assert cosine["total_pairs"] == 2

        # Canonical responses job (from the merge) + embeddings job exist.
        assert result.batch_job_id is not None
        assert result.embedding_batch_job_id is not None
        responses_job = db.get(BatchJob, result.batch_job_id)
        embeddings_job = db.get(BatchJob, result.embedding_batch_job_id)
        assert responses_job.job_type == JOB_TYPE_EVALUATION_FAST
        assert responses_job.raw_output_url is not None
        assert embeddings_job.job_type == JOB_TYPE_EMBEDDING_FAST
        assert embeddings_job.raw_output_url is not None

        assert _fast_pipeline_mocks.create_traces.called
        assert _fast_pipeline_mocks.update_traces.called
        # FR-13: attach_cost called twice (response + embedding stages).
        assert _fast_pipeline_mocks.attach_cost.call_count == 2

        run = db.get(EvaluationRun, result.id)
        assert run.score_trace_url is not None
        traces = result.score["traces"]
        assert len(traces) == 2
        by_trace = {t["trace_id"]: t for t in traces}
        assert {"trace-1", "trace-2"} == set(by_trace)
        sample = by_trace["trace-1"]
        assert sample["question"] == "Q1"
        assert sample["ground_truth_answer"] == "A1"
        assert sample["scores"][0]["name"] == "Cosine Similarity"
        assert sample["scores"][0]["value"] == pytest.approx(1.0, abs=0.01)


# Aggregate-level failure threshold (FR-14)


class TestAggregateFailureThreshold:
    def test_run_fast_evaluation_raises_when_merged_failure_ratio_breaches(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
    ):
        """Merged failure ratio > threshold → run_fast_evaluation fails fast."""
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key)
        # 3 of 4 failed → 0.75 > 0.5 threshold.
        results = [
            _resp_result("item-0", "Q0", "A0", failed=True),
            _resp_result("item-1", "Q1", "A1", failed=True),
            _resp_result("item-2", "Q2", "A2", failed=True),
            _resp_result("item-3", "Q3", "A3", failed=False),
        ]
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=results,
            store=_s3_store,
        )

        fake_openai = MagicMock()
        with pytest.raises(RuntimeError, match="failure threshold"):
            run_fast_evaluation(
                session=db,
                openai_client=fake_openai,
                langfuse=MagicMock(),
                eval_run=eval_run,
            )

        # Fails before the embeddings stage.
        fake_openai.embeddings.create.assert_not_called()


# Post-completion chunk-artifact cleanup (_cleanup_response_chunks)


def _persist_score_into(db: Session):
    """save_score side_effect that mirrors its S3-success path onto the test session.

    save_score opens its own Session(engine), invisible to this test's rolled-back
    transaction; this makes the completion visible on `db` so the run keeps going.
    """

    def _fake_save_score(*, eval_run_id, score, **_):
        run = db.get(EvaluationRun, eval_run_id)
        run.score = {"summary_scores": score["summary_scores"]}
        run.score_trace_url = f"s3://bucket/traces_{eval_run_id}.json"
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    return _fake_save_score


class TestCleanupResponseChunks:
    """A completed run collapses its per-chunk artifacts; a failed run keeps them."""

    def _run_to_completion(self, db, eval_run, fake_openai, storage):
        with (
            patch("app.crud.evaluations.fast.get_cloud_storage", return_value=storage),
            patch(
                "app.crud.evaluations.fast.save_score",
                side_effect=_persist_score_into(db),
            ),
        ):
            return run_fast_evaluation(
                session=db,
                openai_client=fake_openai,
                langfuse=MagicMock(),
                eval_run=eval_run,
            )

    def test_completed_run_deletes_chunk_rows_and_s3_files(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _fast_pipeline_mocks,
    ):
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key, status="pending")
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[_resp_result("item-1", "Q1", "A1", question_id=1)],
            store=_fast_pipeline_mocks.store,
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=1,
            results=[_resp_result("item-2", "Q2", "A2", question_id=2)],
            store=_fast_pipeline_mocks.store,
        )
        chunk_urls = {
            j.raw_output_url
            for j in list_response_chunk_jobs(session=db, eval_run_id=eval_run.id)
        }
        assert len(chunk_urls) == 2

        deleted: list[str] = []
        storage = MagicMock()
        storage.delete.side_effect = deleted.append

        fake_openai = MagicMock()
        fake_openai.embeddings.create.return_value = _fake_embedding_response()

        result = self._run_to_completion(db, eval_run, fake_openai, storage)

        assert result.status == "completed"
        # Chunk rows gone, and every chunk S3 file was deleted.
        assert list_response_chunk_jobs(session=db, eval_run_id=eval_run.id) == []
        assert set(deleted) == chunk_urls
        # Canonical responses job (JOB_TYPE_EVALUATION_FAST) is untouched.
        canonical = db.get(BatchJob, result.batch_job_id)
        assert canonical is not None
        assert canonical.job_type == JOB_TYPE_EVALUATION_FAST
        assert canonical.raw_output_url not in deleted

    def test_failed_run_retains_chunk_artifacts(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
    ):
        """A run that breaches the failure threshold never reaches Stage 4, so its
        chunk rows/files survive for the cron healer — nothing is deleted."""
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key)
        # 3 of 4 failed → 0.75 > 0.5 threshold.
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[
                _resp_result("item-0", "Q0", "A0", failed=True),
                _resp_result("item-1", "Q1", "A1", failed=True),
                _resp_result("item-2", "Q2", "A2", failed=True),
                _resp_result("item-3", "Q3", "A3", failed=False),
            ],
            store=_s3_store,
        )

        storage = MagicMock()
        fake_openai = MagicMock()
        with patch("app.crud.evaluations.fast.get_cloud_storage", return_value=storage):
            with pytest.raises(RuntimeError, match="failure threshold"):
                run_fast_evaluation(
                    session=db,
                    openai_client=fake_openai,
                    langfuse=MagicMock(),
                    eval_run=eval_run,
                )

        assert list_response_chunk_jobs(session=db, eval_run_id=eval_run.id) != []
        storage.delete.assert_not_called()

    def test_cleanup_failure_does_not_fail_the_run(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _fast_pipeline_mocks,
    ):
        """storage.delete raising during cleanup is swallowed; the run still completes."""
        eval_run = _make_fast_run(db=db, user_api_key=user_api_key, status="pending")
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[_resp_result("item-1", "Q1", "A1", question_id=1)],
            store=_fast_pipeline_mocks.store,
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=1,
            results=[_resp_result("item-2", "Q2", "A2", question_id=2)],
            store=_fast_pipeline_mocks.store,
        )

        storage = MagicMock()
        storage.delete.side_effect = RuntimeError("s3 delete down")

        fake_openai = MagicMock()
        fake_openai.embeddings.create.return_value = _fake_embedding_response()

        result = self._run_to_completion(db, eval_run, fake_openai, storage)

        assert result.status == "completed"
        # The swallow only counts if delete was actually attempted and raised.
        storage.delete.assert_called()


# Chunk failure isolation (execute_fast_evaluation_chunk)


class TestChunkFailureIsolation:
    def test_chunk_failure_reraises_and_leaves_run_processing(
        self,
        db: Session,
        user_api_key: TestAuthContext,
    ):
        # A failed chunk writes no marker: it simply leaves no raw_output_url, so
        # the run stays `processing` and the cron healer re-enqueues the index.
        eval_run = _make_fast_run(
            db=db, user_api_key=user_api_key, status="processing", total_items=4
        )

        with (
            patch(
                "app.services.evaluations.fast.Session",
                lambda *a, **k: _FakeSessionCtx(db),
            ),
            patch(
                "app.services.evaluations.fast._resolve_config_and_clients",
                return_value=(
                    TextLLMParams(model="gpt-4o", instructions="x"),
                    MagicMock(),
                    MagicMock(),
                ),
            ),
            patch(
                "app.services.evaluations.fast.fetch_dataset_items",
                return_value=[_dataset_item(f"item-{i}") for i in range(4)],
            ),
            patch(
                "app.services.evaluations.fast.run_response_chunk",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                execute_fast_evaluation_chunk(eval_run_id=eval_run.id, chunk_index=0)

        db.expire_all()
        # No chunk row for the failed index, run untouched.
        assert db.get(EvaluationRun, eval_run.id).status == "processing"
        chunk_rows = list_response_chunk_jobs(session=db, eval_run_id=eval_run.id)
        assert chunk_rows == []


# Fan-out partition (validate_and_start + execute_fast_evaluation_chunk)


class TestFanOutPartition:
    def test_validate_and_start_fans_out_exactly_ceil_chunks(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "EVAL_FAST_CHUNK_SIZE", 2)
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        items = [_dataset_item(f"item-{i}") for i in range(5)]

        with (
            patch("app.services.evaluations.fast.get_langfuse_client"),
            patch(
                "app.services.evaluations.fast.fetch_dataset_items",
                return_value=items,
            ),
            patch(
                "app.services.evaluations.fast.start_fast_evaluation_chunk"
            ) as mock_start,
        ):
            run = validate_and_start_fast_evaluation(
                session=db,
                dataset_id=dataset.id,
                run_name=f"fanout-{random_lower_string()}",
                config_id=config.id,
                config_version=1,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
            )

        # ceil(5 / 2) = 3 chunks, indices 0..2, no gaps or dupes.
        assert mock_start.call_count == 3
        dispatched = {c.kwargs["chunk_index"] for c in mock_start.call_args_list}
        assert dispatched == {0, 1, 2}
        assert all(c.kwargs["eval_run_id"] == run.id for c in mock_start.call_args_list)
        assert run.total_items == 5

    def test_chunk_slices_cover_the_sorted_dataset_without_overlap(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "EVAL_FAST_CHUNK_SIZE", 2)
        eval_run = _make_fast_run(
            db=db, user_api_key=user_api_key, status="processing", total_items=5
        )
        # Deliberately unsorted so the deterministic sort inside the worker matters.
        items = [
            _dataset_item(item_id)
            for item_id in ["item-3", "item-1", "item-5", "item-2", "item-4"]
        ]

        captured: list[list[str]] = []

        def _capture(*, dataset_items_slice, **_):
            captured.append([it["id"] for it in dataset_items_slice])

        with (
            patch(
                "app.services.evaluations.fast.Session",
                lambda *a, **k: _FakeSessionCtx(db),
            ),
            patch(
                "app.services.evaluations.fast._resolve_config_and_clients",
                return_value=(
                    TextLLMParams(model="gpt-4o", instructions="x"),
                    MagicMock(),
                    MagicMock(),
                ),
            ),
            patch(
                "app.services.evaluations.fast.fetch_dataset_items",
                return_value=items,
            ),
            patch(
                "app.services.evaluations.fast.run_response_chunk", side_effect=_capture
            ),
        ):
            for chunk_index in range(3):
                execute_fast_evaluation_chunk(
                    eval_run_id=eval_run.id, chunk_index=chunk_index
                )

        assert captured == [["item-1", "item-2"], ["item-3", "item-4"], ["item-5"]]
        union = [item_id for chunk_slice in captured for item_id in chunk_slice]
        assert union == sorted(union)
        assert len(union) == len(set(union)) == 5


# validate_and_start failure handling (dataset fetch error → 500 + failed run)


class TestValidateAndStartFailure:
    def test_dataset_fetch_error_marks_run_failed_and_raises_500(
        self,
        db: Session,
        user_api_key: TestAuthContext,
    ):
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        run_name = f"fetch-fail-{random_lower_string()}"

        with (
            patch("app.services.evaluations.fast.get_langfuse_client"),
            patch(
                "app.services.evaluations.fast.fetch_dataset_items",
                side_effect=RuntimeError("langfuse down"),
            ),
            patch(
                "app.services.evaluations.fast.start_fast_evaluation_chunk"
            ) as mock_start,
        ):
            with pytest.raises(HTTPException) as exc:
                validate_and_start_fast_evaluation(
                    session=db,
                    dataset_id=dataset.id,
                    run_name=run_name,
                    config_id=config.id,
                    config_version=1,
                    organization_id=user_api_key.organization_id,
                    project_id=user_api_key.project_id,
                )

        assert exc.value.status_code == 500
        mock_start.assert_not_called()
        failed = db.exec(
            select(EvaluationRun).where(EvaluationRun.run_name == run_name)
        ).first()
        assert failed is not None
        assert failed.status == "failed"


# Cron fan-in barrier + stall healer (dispatch_fast_evaluation_barriers)


class TestCronBarrier:
    def test_all_chunks_done_dispatches_one_aggregate(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "EVAL_FAST_CHUNK_SIZE", 2)
        eval_run = _make_fast_run(
            db=db, user_api_key=user_api_key, status="processing", total_items=4
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[_resp_result("a")],
            store=_s3_store,
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=1,
            results=[_resp_result("b")],
            store=_s3_store,
        )

        with (
            patch("app.celery.utils.start_fast_evaluation_aggregate") as mock_agg,
            patch("app.celery.utils.start_fast_evaluation_chunk") as mock_chunk,
        ):
            summary = dispatch_fast_evaluation_barriers(session=db)

        mock_agg.assert_called_once_with(eval_run_id=eval_run.id)
        mock_chunk.assert_not_called()
        assert summary["aggregates_dispatched"] >= 1
        assert summary["chunks_reenqueued"] == 0

    def test_stalled_run_reenqueues_only_missing_chunks(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "EVAL_FAST_CHUNK_SIZE", 2)
        eval_run = _make_fast_run(
            db=db, user_api_key=user_api_key, status="processing", total_items=4
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[_resp_result("a")],
            store=_s3_store,
        )
        # index 1 is missing; push updated_at past the stall threshold.
        eval_run.updated_at = now() - timedelta(
            minutes=settings.EVAL_FAST_STALL_THRESHOLD_MINUTES + 5
        )
        db.add(eval_run)
        db.commit()

        with (
            patch("app.celery.utils.start_fast_evaluation_aggregate") as mock_agg,
            patch("app.celery.utils.start_fast_evaluation_chunk") as mock_chunk,
        ):
            summary = dispatch_fast_evaluation_barriers(session=db)

        mock_agg.assert_not_called()
        mock_chunk.assert_called_once_with(eval_run_id=eval_run.id, chunk_index=1)
        assert summary["chunks_reenqueued"] == 1

    def test_incomplete_but_not_stalled_does_nothing(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _s3_store,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "EVAL_FAST_CHUNK_SIZE", 2)
        eval_run = _make_fast_run(
            db=db, user_api_key=user_api_key, status="processing", total_items=4
        )
        _seed_response_chunk(
            db=db,
            eval_run=eval_run,
            chunk_index=0,
            results=[_resp_result("a")],
            store=_s3_store,
        )
        # index 1 missing but updated_at is recent → within the stall window.
        eval_run.updated_at = now()
        db.add(eval_run)
        db.commit()

        with (
            patch("app.celery.utils.start_fast_evaluation_aggregate") as mock_agg,
            patch("app.celery.utils.start_fast_evaluation_chunk") as mock_chunk,
        ):
            summary = dispatch_fast_evaluation_barriers(session=db)

        mock_agg.assert_not_called()
        mock_chunk.assert_not_called()
        # The run is matched by the barrier query (guards against a vacuous pass).
        assert summary["total"] >= 1
        assert summary["aggregates_dispatched"] == 0
        assert summary["chunks_reenqueued"] == 0
