"""Tests for the fast (synchronous) evaluation path.

Covers FR-1 through FR-15 from the Fast Evaluation SRD. External boundaries
(OpenAI, Langfuse, S3, Celery dispatch) are mocked; the DB is real (`db`
fixture).
"""

import random
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.crud.evaluations.fast import (
    JOB_TYPE_EMBEDDING_FAST,
    JOB_TYPE_EVALUATION_FAST,
    _create_response,
    _is_failure_threshold_breached,
    _stage1_responses,
    _stage2_embeddings,
    run_fast_evaluation,
)
from app.models import Config, EvaluationDataset, EvaluationRun
from app.models.batch_job import BatchJob
from app.models.evaluation import RunModeEnum
from app.models.llm.request import (
    ConfigBlob,
    KaapiCompletionConfig,
    TextLLMParams,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)


def _api_error(resp_body: dict) -> str:
    """Pull the human-readable error string out of an APIResponse failure body."""
    return str(resp_body.get("error") or resp_body.get("detail") or resp_body).lower()


@pytest.fixture(autouse=True)
def _seeded_random() -> Iterator[None]:
    """Make jitter / random.choice deterministic so tests are repeatable."""
    random.seed(0)
    yield


# ---------------------------------------------------------------------------
# Pure-function helpers (no FastAPI, no DB)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Route validation: POST /evaluations with run_mode=fast (FR-1..FR-5, FR-15)
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_dispatch() -> Iterator[MagicMock]:
    """Stub the Celery dispatch so tests don't actually enqueue work."""
    with patch(
        "app.services.evaluations.fast.enqueue_fast_evaluation",
        return_value="fake-task-id",
    ) as m:
        yield m


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
    """Create a stored text-OpenAI Kaapi config (eligible for fast eval)."""
    return create_test_config(
        db=db,
        project_id=project_id,
        use_kaapi_schema=True,
    )


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
        """FR-4: eligible request returns processing + dispatches orchestrator."""
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
        _patch_dispatch.assert_called_once()

        # DB state matches the response.
        run = db.get(EvaluationRun, body["id"])
        assert run is not None
        assert run.run_mode == RunModeEnum.FAST.value
        assert run.status == "processing"

    def test_fr2_rejects_dataset_with_too_many_unique_rows(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-2: >10 unique rows → 422 dataset_too_large_for_fast."""
        # default EVAL_FAST_MAX_UNIQUE_ROWS = 10; create 11 unique rows
        dataset = _make_fast_eligible_dataset(
            db=db, user_api_key=user_api_key, original_items_count=11
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
        # The route wraps HTTPException.detail into APIResponse.error.
        error_str = _api_error(resp.json())
        assert "dataset_too_large_for_fast" in error_str
        assert "11" in error_str  # surfaces actual unique-row count
        _patch_dispatch.assert_not_called()

    def test_fr1_rejects_non_text_config(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-1: non-text config for fast mode → 422 config_type_unsupported.

        Build a stored config whose completion.type is not 'text'. The current
        config factories produce text configs by default, so we patch the
        resolved blob to look like an STT config.
        """
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)

        # Patch resolve_evaluation_config to return an STT-type blob.
        fake_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
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
        # First call dispatched; second must not have.
        assert _patch_dispatch.call_count == 1

    def test_fr15_get_evaluation_returns_run_mode(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """FR-15: GET /evaluations/{id} surfaces `run_mode` for both modes."""
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="fr15-existing",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="completed",
            run_mode=RunModeEnum.FAST.value,
            total_items=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        resp = client.get(
            f"/api/v1/evaluations/{eval_run.id}", headers=user_api_key_header
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["run_mode"] == "fast"


# ---------------------------------------------------------------------------
# Dataset listing eligibility filter (FR-5)
# ---------------------------------------------------------------------------


class TestDatasetListEligibleForFast:
    def test_fr5_filters_to_fast_eligible_only(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """FR-5: eligible_for_fast=true filters list to datasets with ≤10 unique rows."""
        eligible = _make_fast_eligible_dataset(
            db=db, user_api_key=user_api_key, original_items_count=5
        )
        ineligible = _make_fast_eligible_dataset(
            db=db, user_api_key=user_api_key, original_items_count=20
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


# ---------------------------------------------------------------------------
# Stage skipping on retry (FR-6, FR-7)
# ---------------------------------------------------------------------------


def _fake_openai_response(text: str = "answer", item_id: str = "item-1"):
    """Mimic the SDK's response.responses.create return shape."""
    return SimpleNamespace(
        id=f"resp_{item_id}",
        output_text=text,
        output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
    )


def _fake_embedding_response():
    """Mimic openai.embeddings.create return shape (2 vectors)."""
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            SimpleNamespace(index=1, embedding=[1.0, 0.0, 0.0]),
        ],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )


class TestStageSkipping:
    """FR-6 / FR-7: stages skip on retry when their batch_job marker is set."""

    def test_fr6_stage1_skips_when_batch_job_id_already_set(
        self,
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """Pre-existing batch_job → Stage 1 does not call the Responses API."""
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="fr6-stage1-skip",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="processing",
            run_mode=RunModeEnum.FAST.value,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        # Pre-create a batch_job marker as if Stage 1 had already completed.
        existing = BatchJob(
            provider="openai",
            job_type=JOB_TYPE_EVALUATION_FAST,
            config={},
            raw_output_url="s3://bucket/responses_x.json",
            total_items=1,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(existing)
        db.commit()
        db.refresh(existing)

        eval_run.batch_job_id = existing.id
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        cached = [
            {
                "item_id": "item-1",
                "question": "q",
                "generated_output": "a",
                "ground_truth": "a",
                "response_id": "resp_1",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                "question_id": 1,
                "failed": False,
            }
        ]

        fake_openai = MagicMock()
        with patch("app.crud.evaluations.fast._load_unit_from_s3", return_value=cached):
            _, results = _stage1_responses(
                session=db,
                openai_client=fake_openai,
                eval_run=eval_run,
                config=TextLLMParams(model="gpt-4o", instructions="x"),
                dataset_items=[
                    {
                        "id": "item-1",
                        "input": {"question": "q"},
                        "expected_output": {"answer": "a"},
                        "metadata": {},
                    }
                ],
                log_prefix="[t]",
            )

        # Skip path returns the cached unit and never calls the OpenAI client.
        assert results == cached
        fake_openai.responses.create.assert_not_called()

    def test_fr7_stage2_skips_when_embedding_batch_job_id_already_set(
        self,
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """Pre-existing embedding batch_job → Stage 2 does not call the Embeddings API."""
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="fr7-stage2-skip",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="processing",
            run_mode=RunModeEnum.FAST.value,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

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


# ---------------------------------------------------------------------------
# End-to-end orchestrator pipeline with mocked externals (FR-9..FR-14)
# ---------------------------------------------------------------------------


@pytest.fixture
def _fast_pipeline_mocks():
    """Patch external boundaries used inside `run_fast_evaluation`.

    OpenAI client returns fixed responses/embeddings, Langfuse returns trace
    ids, S3 upload returns a URL, and `attach_cost` no-ops so the test does
    not need a model_config row.
    """
    with (
        patch("app.crud.evaluations.fast.fetch_dataset_items") as mock_fetch_items,
        patch(
            "app.crud.evaluations.fast._upload_unit_to_s3",
            side_effect=lambda **kw: f"s3://bucket/{kw['filename']}",
        ),
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
        mock_fetch_items.return_value = [
            {
                "id": "item-1",
                "input": {"question": "Q1"},
                "expected_output": {"answer": "A1"},
                "metadata": {"question_id": 1},
            },
            {
                "id": "item-2",
                "input": {"question": "Q2"},
                "expected_output": {"answer": "A2"},
                "metadata": {"question_id": 2},
            },
        ]
        mock_create_traces.return_value = {
            "item-1": "trace-1",
            "item-2": "trace-2",
        }
        yield SimpleNamespace(
            fetch_items=mock_fetch_items,
            create_traces=mock_create_traces,
            update_traces=mock_update_traces,
            attach_cost=mock_attach_cost,
        )


class TestFastPipelineEndToEnd:
    """Run the orchestrator with mocked external boundaries (FR-9..FR-13)."""

    def test_pipeline_completes_with_scores_and_writes_batch_jobs(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _fast_pipeline_mocks,
    ):
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="pipeline-happy-path",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="pending",
            run_mode=RunModeEnum.FAST.value,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        fake_openai = MagicMock()
        fake_openai.responses.create.side_effect = lambda **_: _fake_openai_response(
            text="LLM answer", item_id="x"
        )
        fake_openai.embeddings.create.return_value = _fake_embedding_response()
        fake_langfuse = MagicMock()

        # save_score opens its own Session(engine), which can't see this test's
        # rolled-back transaction. Mirror its S3-success path against the test
        # session so the persisted state (summary in DB, traces in S3) is
        # observable here. (In production the worker's commits are visible across
        # connections, so the real save_score works unchanged.)
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
                config=TextLLMParams(model="gpt-4o", instructions="be helpful"),
            )

        # FR-11/FR-14: status completed, summary cosine ≈ 1.0 for identical vectors
        assert result.status == "completed"
        assert result.score is not None
        cosine = result.score["summary_scores"][0]
        assert cosine["name"] == "Cosine Similarity"
        assert cosine["avg"] == pytest.approx(1.0, abs=0.01)
        assert cosine["total_pairs"] == 2

        # Stage markers exist (FR-6/FR-7 invariant + FR-9 — no llm_call rows).
        assert result.batch_job_id is not None
        assert result.embedding_batch_job_id is not None
        responses_job = db.get(BatchJob, result.batch_job_id)
        embeddings_job = db.get(BatchJob, result.embedding_batch_job_id)
        assert responses_job.job_type == JOB_TYPE_EVALUATION_FAST
        assert responses_job.raw_output_url is not None
        assert embeddings_job.job_type == JOB_TYPE_EMBEDDING_FAST
        assert embeddings_job.raw_output_url is not None

        # FR-12: Langfuse traces created and per-trace scores attached.
        assert _fast_pipeline_mocks.create_traces.called
        assert _fast_pipeline_mocks.update_traces.called

        # FR-13: attach_cost called twice (response + embedding stages).
        assert _fast_pipeline_mocks.attach_cost.call_count == 2

        # Cached trace unit is persisted like the batch path so the read path
        # (trace view / resync / grouped export) never has to hit Langfuse.
        run = db.get(EvaluationRun, result.id)
        assert run.score_trace_url is not None
        # Full unit (summary + per-trace records) is surfaced on the return.
        traces = result.score["traces"]
        assert len(traces) == 2
        by_trace = {t["trace_id"]: t for t in traces}
        assert {"trace-1", "trace-2"} == set(by_trace)
        sample = by_trace["trace-1"]
        assert sample["question"] == "Q1"
        assert sample["ground_truth_answer"] == "A1"
        assert sample["scores"][0]["name"] == "Cosine Similarity"
        assert sample["scores"][0]["value"] == pytest.approx(1.0, abs=0.01)

    def test_completion_clears_stale_error_message(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        _fast_pipeline_mocks,
    ):
        """A successful fast run clears any error_message left by a transient
        failure (e.g. the batch poller racing this synchronous run), so it never
        displays as completed-with-error."""
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="pipeline-clears-error",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="processing",
            error_message="Checking failed: EvaluationRun 640 has no batch_job_id",
            run_mode=RunModeEnum.FAST.value,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        fake_openai = MagicMock()
        fake_openai.responses.create.side_effect = lambda **_: _fake_openai_response(
            text="LLM answer", item_id="x"
        )
        fake_openai.embeddings.create.return_value = _fake_embedding_response()

        with patch(
            "app.crud.evaluations.fast.save_score",
            side_effect=lambda *, eval_run_id, **_: db.get(EvaluationRun, eval_run_id),
        ):
            result = run_fast_evaluation(
                session=db,
                openai_client=fake_openai,
                langfuse=MagicMock(),
                eval_run=eval_run,
                config=TextLLMParams(model="gpt-4o", instructions="be helpful"),
            )

        assert result.status == "completed"
        assert result.error_message is None
        db.refresh(eval_run)
        assert eval_run.error_message is None


# ---------------------------------------------------------------------------
# Failure-threshold short-circuit (FR-14)
# ---------------------------------------------------------------------------


class TestFailureThresholdInPipeline:
    def test_fr14_stage1_raises_when_failure_ratio_exceeds_threshold(
        self,
        db: Session,
        user_api_key: TestAuthContext,
    ):
        """FR-14: Stage 1 raises RuntimeError when failure ratio > threshold.

        The outer orchestrator (run_fast_evaluation / execute_fast_evaluation)
        catches the RuntimeError and marks the run failed; the structural
        guarantee under test is that Stage 1 fails fast instead of proceeding
        to Stage 2 with a mostly-broken response set.
        """
        dataset = _make_fast_eligible_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_openai_config(db, user_api_key.project_id)
        eval_run = EvaluationRun(
            run_name="fr14-fail-threshold",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="processing",
            run_mode=RunModeEnum.FAST.value,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        # Make all OpenAI Responses calls fail with a permanent error so retries
        # short-circuit and every item gets failed=True. With every item
        # failing, the failure fraction (1.0) is well above the 0.5 threshold.
        fake_openai = MagicMock()
        fake_openai.responses.create.side_effect = openai.AuthenticationError(
            message="bad key", response=MagicMock(), body=None
        )

        dataset_items = [
            {
                "id": f"item-{i}",
                "input": {"question": f"Q{i}"},
                "expected_output": {"answer": f"A{i}"},
                "metadata": {},
            }
            for i in range(4)
        ]

        with pytest.raises(RuntimeError, match="failure threshold"):
            _stage1_responses(
                session=db,
                openai_client=fake_openai,
                eval_run=eval_run,
                config=TextLLMParams(model="gpt-4o", instructions="x"),
                dataset_items=dataset_items,
                log_prefix="[t]",
            )
