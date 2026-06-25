"""Tests for POST /evaluations/{evaluation_id}/improve-prompt.

Covers all functional requirements from docs/srd-ai-prompt-improvement.md.
The Anthropic LLM call is mocked at the Anthropic client level; the DB is real.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations.score import (
    COSINE_SCORE_NAME,
    SCORE_DATA_TYPE_CATEGORICAL,
    SCORE_DATA_TYPE_NUMERIC,
)
from app.models import ConfigVersion, EvaluationDataset, EvaluationRun
from app.services.evaluations.prompt_improvement import AI_GENERATED_MARKER
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_config, create_test_evaluation_dataset
from app.tests.utils.utils import random_lower_string

# ── constants ─────────────────────────────────────────────────────────────────

IMPROVE_URL = "/api/v1/evaluations/{evaluation_id}/improve-prompt"

_IMPROVED_INSTRUCTIONS = "You are an improved assistant. Answer precisely."
_RATIONALE = (
    "Tightened answer scoping to address weak questions in the Payments category."
)
_LLM_JSON_RESPONSE = json.dumps(
    {
        "improved_instructions": _IMPROVED_INSTRUCTIONS,
        "rationale": _RATIONALE,
    }
)


# ── shared fixture helpers ─────────────────────────────────────────────────────


def _make_anthropic_mock(text_content: str = _LLM_JSON_RESPONSE) -> MagicMock:
    """Return a mock that looks like anthropic.Anthropic().messages.create(...)."""
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text_content

    response = MagicMock()
    response.content = [content_block]
    response.id = "msg_test_id"

    client_instance = MagicMock()
    client_instance.messages.create.return_value = response

    return client_instance


def _make_config_with_instructions(
    db: Session,
    project_id: int,
    instructions: str = "You are a helpful assistant.",
) -> Any:
    """Create a config whose config_blob has completion.params.instructions set."""
    from app.crud.config import ConfigCrud
    from app.models.llm.request import ConfigBlob
    from app.models.llm import KaapiCompletionConfig

    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "temperature": 0.5,
                "instructions": instructions,
                "knowledge_base_ids": ["vs_abc123"],
            },
        )
    )
    from app.models.config.config import ConfigCreate, ConfigTag

    config_create = ConfigCreate(
        name=f"test-config-{random_lower_string()}",
        description="Test configuration for improve-prompt",
        config_blob=config_blob,
        commit_message="Initial version",
        tag=ConfigTag.DEFAULT,
    )
    config_crud = ConfigCrud(session=db, project_id=project_id)
    config, _ = config_crud.create_or_raise(config_create)
    return config


def _make_completed_run(
    db: Session,
    config_id: Any,
    config_version: int,
    organization_id: int,
    project_id: int,
    dataset_id: int,
    score: dict[str, Any] | None = None,
    status: str = "completed",
    run_name: str | None = None,
) -> EvaluationRun:
    """Create and persist an EvaluationRun with the given parameters."""
    if run_name is None:
        run_name = f"run-{random_lower_string()}"

    if score is None:
        # Minimal score structure: one weak trace, COSINE_SCORE_NAME summary with data_type NUMERIC
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.3,
                )
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.3,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )

    run = EvaluationRun(
        run_name=run_name,
        dataset_name=f"ds-{random_lower_string()}",
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        status=status,
        total_items=1,
        score=score,
        organization_id=organization_id,
        project_id=project_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _make_trace(
    *,
    question_id: int | None,
    trace_id: str,
    metric_name: str = COSINE_SCORE_NAME,
    metric_value: float | None = None,
    metric_data_type: str = SCORE_DATA_TYPE_NUMERIC,
    unscoreable: bool = False,
    extra_scores: list[dict[str, Any]] | None = None,
    question: str = "What is the capital?",
    llm_answer: str = "Paris",
    ground_truth_answer: str = "Paris, France",
    category: str = "Geography",
) -> dict[str, Any]:
    """Build a trace dict whose scores list uses the new inline-scores shape.

    Each entry in scores has: name, value, data_type, unscoreable.
    extra_scores lets a caller inject additional score entries (e.g. a categorical score).
    """
    scores: list[dict[str, Any]] = []
    if metric_value is not None or unscoreable:
        scores.append(
            {
                "name": metric_name,
                "value": metric_value,
                "data_type": metric_data_type,
                "comment": None,
                "unscoreable": unscoreable,
            }
        )
    if extra_scores:
        scores.extend(extra_scores)
    return {
        "trace_id": trace_id,
        "question_id": question_id,
        "question": question,
        "llm_answer": llm_answer,
        "ground_truth_answer": ground_truth_answer,
        "category": category,
        "scores": scores,
    }


def _score_payload(
    *,
    traces: list[dict[str, Any]],
    summary_scores: list[dict[str, Any]] | None = None,
    category_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "traces": traces,
        "summary_scores": summary_scores or [],
        "category_metrics": category_metrics or [],
    }


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def auth(user_api_key: TestAuthContext) -> TestAuthContext:
    return user_api_key


@pytest.fixture
def headers(user_api_key_header: dict[str, str]) -> dict[str, str]:
    return user_api_key_header


@pytest.fixture
def dataset(db: Session, auth: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )


@pytest.fixture
def config_with_instructions(db: Session, auth: TestAuthContext) -> Any:
    return _make_config_with_instructions(
        db=db,
        project_id=auth.project_id,
        instructions="You are a helpful assistant. Answer clearly.",
    )


@pytest.fixture
def anthropic_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the platform-owned Anthropic key the service reads from settings."""
    monkeypatch.setattr(
        settings,
        "ANTHROPIC_API_KEY",
        "sk-ant-test-" + random_lower_string(),
    )


@pytest.fixture
def completed_run(
    db: Session,
    auth: TestAuthContext,
    dataset: EvaluationDataset,
    config_with_instructions: Any,
    anthropic_creds: None,
) -> EvaluationRun:
    return _make_completed_run(
        db=db,
        config_id=config_with_instructions.id,
        config_version=1,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        dataset_id=dataset.id,
    )


# ── FR-1, FR-10, FR-11, FR-12, FR-13 — happy path ────────────────────────────


class TestHappyPath:
    """Completed run → 201 with correctly formed new version."""

    def test_returns_201_with_new_version(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
        config_with_instructions: Any,
    ) -> None:
        """FR-1: completed run returns 201 and new config_version at previous latest + 1."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert body["version"] == 2  # initial version was 1
        assert body["config_id"] == str(config_with_instructions.id)

    def test_commit_message_starts_with_ai_generated_marker(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """FR-11: new version's commit_message starts with the AI_GENERATED_MARKER."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert body["commit_message"].startswith(AI_GENERATED_MARKER)

    def test_commit_message_contains_source_evaluation_run_id(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """FR-12: commit_message embeds source_evaluation_run_id == evaluation_id."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert f"source_evaluation_run_id={completed_run.id}" in body["commit_message"]

    def test_commit_message_contains_rationale_metric_and_threshold(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """FR-13: commit_message contains the LLM rationale, metric=<display-name>, and threshold=."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        commit_message = resp.json()["data"]["commit_message"]
        assert _RATIONALE in commit_message
        # The metric display name (not an enum slug) is embedded verbatim.
        assert f"metric={COSINE_SCORE_NAME}" in commit_message
        assert "threshold=" in commit_message

    def test_commit_message_contains_rationale(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """FR-13: commit_message contains the LLM rationale."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert _RATIONALE in body["commit_message"]

    def test_prompt_only_change_other_fields_identical(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
        config_with_instructions: Any,
    ) -> None:
        """FR-10: new version's config_blob equals source blob except
        completion.params.instructions; model, knowledge_base_ids, temperature
        are byte-for-byte identical."""
        # Fetch source blob from DB before the call
        crud = ConfigVersionCrud(
            session=db,
            config_id=config_with_instructions.id,
            project_id=auth.project_id,
        )
        source_version = crud.read_one(version_number=1)
        assert source_version is not None
        source_blob = source_version.config_blob

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        new_version_id = resp.json()["data"]["id"]

        # Fetch the new version from DB
        stmt = select(ConfigVersion).where(ConfigVersion.id == new_version_id)
        new_version = db.exec(stmt).one()

        new_blob = new_version.config_blob
        src_completion = source_blob["completion"]
        new_completion = new_blob["completion"]

        # Only instructions changed
        assert new_completion["params"]["instructions"] == _IMPROVED_INSTRUCTIONS
        assert src_completion["params"]["instructions"] != _IMPROVED_INSTRUCTIONS

        # Everything else is identical
        assert new_completion["params"]["model"] == src_completion["params"]["model"]
        assert (
            new_completion["params"]["temperature"]
            == src_completion["params"]["temperature"]
        )
        assert new_completion["params"].get("knowledge_base_ids") == src_completion[
            "params"
        ].get("knowledge_base_ids")
        assert new_completion["provider"] == src_completion["provider"]
        assert new_completion["type"] == src_completion["type"]


# ── FR-2 — non-completed status → 409 ────────────────────────────────────────


class TestNonCompletedStatus:
    """FR-2: pending/processing/failed status → 409 evaluation_not_completed."""

    @pytest.mark.parametrize("status", ["pending", "processing", "failed"])
    def test_non_completed_returns_409(
        self,
        status: str,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            status=status,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
            headers=headers,
        )

        assert resp.status_code == 409, resp.text
        assert "evaluation_not_completed" in resp.json()["error"]


# ── FR-3 — source config unavailable → 409 ───────────────────────────────────


class TestSourceConfigUnavailable:
    """FR-3: soft-deleted config or missing config_version → 409."""

    def test_soft_deleted_config_returns_409(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        anthropic_creds: None,
    ) -> None:
        from app.core.util import now

        config = _make_config_with_instructions(db=db, project_id=auth.project_id)

        run = _make_completed_run(
            db=db,
            config_id=config.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        # Soft-delete the config
        from app.models.config.config import Config

        stmt = select(Config).where(Config.id == config.id)
        cfg = db.exec(stmt).one()
        cfg.deleted_at = now()
        db.add(cfg)
        db.commit()

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
            headers=headers,
        )

        assert resp.status_code == 409, resp.text
        assert "source_config_unavailable" in resp.json()["error"]

    def test_missing_config_version_returns_409(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        anthropic_creds: None,
    ) -> None:
        config = _make_config_with_instructions(db=db, project_id=auth.project_id)

        # Run points at config_version=99 which does not exist
        run = _make_completed_run(
            db=db,
            config_id=config.id,
            config_version=99,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
            headers=headers,
        )

        assert resp.status_code == 409, resp.text
        assert "source_config_unavailable" in resp.json()["error"]


# ── FR-4 — metric selection: free-form score name ────────────────────────────


class TestMetricSelection:
    """FR-4: metric is a free-form score name; matching is case-insensitive and trimmed."""

    def test_valid_score_name_accepted(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """Sending the exact display name present in summary_scores → accepted."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text

    def test_case_insensitive_metric_name_accepted(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """Lower-cased metric name still resolves correctly (case-insensitive match)."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME.lower(), "threshold": 0.7},
                headers=headers,
            )

        # The lower-cased name must resolve to the same score → 201
        assert resp.status_code == 201, resp.text


# ── FR-5 — metric not in run → 422 ───────────────────────────────────────────


class TestMetricNotAvailable:
    """FR-5: a metric name with no matching summary_scores entry → 422 metric_not_available."""

    def test_unknown_metric_name_returns_422(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """Score has only COSINE_SCORE_NAME; a completely unknown name → metric_not_available."""
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.3,
                )
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.3,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": "No Such Score", "threshold": 0.7},
            headers=headers,
        )

        assert resp.status_code == 422, resp.text
        assert "metric_not_available" in resp.json()["error"]


# ── FR-5b — metric is categorical → 422 metric_not_numeric ───────────────────


class TestMetricNotNumeric:
    """FR-5b: a metric whose summary_scores entry has data_type=CATEGORICAL → 422 metric_not_numeric."""

    def test_categorical_metric_returns_422(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """A categorical summary score must be rejected before any weak-signal computation."""
        categorical_score_name = "Sentiment"
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=categorical_score_name,
                    metric_value=None,
                    metric_data_type=SCORE_DATA_TYPE_CATEGORICAL,
                )
            ],
            summary_scores=[
                {
                    "name": categorical_score_name,
                    "distribution": {"positive": 5, "negative": 2},
                    "total_pairs": 7,
                    "data_type": SCORE_DATA_TYPE_CATEGORICAL,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": categorical_score_name, "threshold": 0.7},
            headers=headers,
        )

        assert resp.status_code == 422, resp.text
        assert "metric_not_numeric" in resp.json()["error"]
        # No new version should have been created
        crud = ConfigVersionCrud(
            session=db,
            config_id=config_with_instructions.id,
            project_id=auth.project_id,
        )
        versions = crud.read_all()
        assert len(versions) == 1


# ── FR-6 — threshold: unbounded, any numeric value accepted ──────────────────


class TestThreshold:
    """FR-6: threshold has no [0,1] constraint; out-of-range values are accepted."""

    @pytest.mark.parametrize("threshold", [2.5, 5.0, 100.0, -1.0])
    def test_out_of_range_threshold_is_accepted(
        self,
        threshold: float,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """threshold > 1 or < 0 no longer triggers validation error — the value is accepted."""
        # With threshold=2.5 all scores on a [0,1] scale are below → weak signals exist
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.3,
                    category="X",
                )
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.3,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": threshold},
                headers=headers,
            )

        # Must NOT be 422 due to threshold validation; either 201 (weak signals found) or
        # 422 no_weak_signals (threshold so low nothing is below it) is acceptable,
        # but never a validation error from the request schema.
        if resp.status_code == 422:
            assert "no_weak_signals" in resp.json()["error"], resp.text
        else:
            assert resp.status_code == 201, resp.text

    def test_non_unit_scale_score_selects_weak_questions_correctly(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """A custom numeric score on a 1–5 Likert scale with threshold=3
        correctly identifies questions scoring below 3."""
        likert_score_name = "Quality Rating"
        # Two questions: one below threshold (score=2.0), one above (score=4.0)
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=likert_score_name,
                    metric_value=2.0,
                    category="A",
                    question="Question weak?",
                ),
                _make_trace(
                    question_id=2,
                    trace_id="t2",
                    metric_name=likert_score_name,
                    metric_value=4.0,
                    category="A",
                    question="Question strong?",
                ),
            ],
            summary_scores=[
                {
                    "name": likert_score_name,
                    "avg": 3.0,
                    "std": 1.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        mock_client = _make_anthropic_mock()
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=mock_client,
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": likert_score_name, "threshold": 3.0},
                headers=headers,
            )

        # The weak question (score=2.0 < 3.0) should be found → 201
        assert resp.status_code == 201, resp.text

        # The LLM prompt should mention only the weak question
        call_args = mock_client.messages.create.call_args
        user_content = ""
        for msg in call_args.kwargs.get("messages") or []:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        assert "Question weak?" in user_content
        assert "Question strong?" not in user_content


# ── FR-7 — consistency ratio ──────────────────────────────────────────────────


class TestConsistencyRatio:
    """FR-7: question with 1-of-3 reps below threshold excluded;
    2-of-3 reps below threshold included.

    Trace-level scores are read from each trace's inline `scores` list.
    """

    def test_one_of_three_below_threshold_excluded(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """1/3 ≈ 0.33 < MIN_CONSISTENCY_RATIO(0.5): question excluded."""
        # question_id=10 has 3 reps: one below threshold, two above
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=10,
                    trace_id="t10a",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.3,
                    category="Info",
                ),  # below
                _make_trace(
                    question_id=10,
                    trace_id="t10b",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.9,
                    category="Info",
                ),  # above
                _make_trace(
                    question_id=10,
                    trace_id="t10c",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.9,
                    category="Info",
                ),  # above
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.7,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.5},
            headers=headers,
        )

        # Category avg (0.7) above threshold, question excluded → no weak signals → 422
        assert resp.status_code == 422, resp.text
        assert "no_weak_signals" in resp.json()["error"]

    def test_two_of_three_below_threshold_included(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """2/3 ≈ 0.67 ≥ MIN_CONSISTENCY_RATIO(0.5): question included — 201 returned."""
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=20,
                    trace_id="t20a",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.3,
                    category="Pay",
                ),  # below
                _make_trace(
                    question_id=20,
                    trace_id="t20b",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.2,
                    category="Pay",
                ),  # below
                _make_trace(
                    question_id=20,
                    trace_id="t20c",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.9,
                    category="Pay",
                ),  # above
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.47,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.5},
                headers=headers,
            )

        # Weak question was included so LLM was called → 201
        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert body["commit_message"].startswith(AI_GENERATED_MARKER)

    def test_unscoreable_repetition_ignored_in_consistency_fraction(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """An unscoreable repetition must be skipped; only scoreable reps count.

        Setup: question_id=30 has 3 reps — one unscoreable, one below threshold,
        one above. The unscoreable rep is ignored, leaving 1-of-2 below threshold
        (ratio 0.5 = MIN_CONSISTENCY_RATIO), so the question IS included.
        """
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=30,
                    trace_id="t30a",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=None,
                    unscoreable=True,
                    category="Z",
                ),  # unscoreable — must be skipped
                _make_trace(
                    question_id=30,
                    trace_id="t30b",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.2,
                    category="Z",
                ),  # below threshold
                _make_trace(
                    question_id=30,
                    trace_id="t30c",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.9,
                    category="Z",
                ),  # above threshold
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.55,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.5},
                headers=headers,
            )

        # 1/2 scoreable reps below threshold == MIN_CONSISTENCY_RATIO → included → 201
        assert resp.status_code == 201, resp.text


# ── FR-8 — underperforming categories ────────────────────────────────────────


class TestWeakCategories:
    """FR-8: underperforming categories identified from trace-level scores, not
    category_metrics.  A category whose mean trace score < threshold is weak."""

    def test_weak_categories_produce_ai_version(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """Categories A (avg=0.3) and B (avg=0.4) are below threshold=0.7;
        C (avg=0.8) is above.  Service calls LLM → 201."""
        score = _score_payload(
            traces=[
                # Category A: one trace, score 0.3
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.3,
                    category="A",
                ),
                # Category B: one trace, score 0.4
                _make_trace(
                    question_id=2,
                    trace_id="t2",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.4,
                    category="B",
                ),
                # Category C: one trace, score 0.8 — above threshold
                _make_trace(
                    question_id=3,
                    trace_id="t3",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.8,
                    category="C",
                ),
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.5,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
            # category_metrics is intentionally omitted / empty — service must NOT use it
            category_metrics=[],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        # Provenance is captured in commit_message
        assert body["commit_message"].startswith(AI_GENERATED_MARKER)
        assert f"metric={COSINE_SCORE_NAME}" in body["commit_message"]


# ── FR-9 — nothing to improve → 422 ──────────────────────────────────────────


class TestNoWeakSignals:
    """FR-9: no weak questions and no weak categories → 422 no_weak_signals,
    no new version created."""

    def test_no_weak_signals_returns_422(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        # All traces above threshold; category avgs above threshold (computed from traces)
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.9,
                    category="Good",
                ),
                _make_trace(
                    question_id=2,
                    trace_id="t2",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.95,
                    category="Good",
                ),
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.92,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.5},
            headers=headers,
        )

        assert resp.status_code == 422, resp.text
        assert "no_weak_signals" in resp.json()["error"]

    def test_no_weak_signals_version_count_unchanged(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """No new version row should be created when no_weak_signals fires."""
        config_id = config_with_instructions.id
        crud = ConfigVersionCrud(
            session=db, config_id=config_id, project_id=auth.project_id
        )
        versions_before = crud.read_all()
        count_before = len(versions_before)

        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t1",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.95,
                    category="Fine",
                )
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.95,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.5},
            headers=headers,
        )

        assert resp.status_code == 422, resp.text

        # The session may need a refresh because CRUD reads from the same session
        db.expire_all()
        versions_after = crud.read_all()
        assert len(versions_after) == count_before


# ── FR-14 — prior iterations preserved ───────────────────────────────────────


class TestPriorVersionsPreserved:
    """FR-14: pre-existing config_version rows unchanged and retrievable."""

    def test_prior_versions_still_retrievable(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        config_id = config_with_instructions.id
        crud = ConfigVersionCrud(
            session=db, config_id=config_id, project_id=auth.project_id
        )

        # Read the original version before improvement
        original_version = crud.read_one(version_number=1)
        assert original_version is not None
        original_blob_instructions = original_version.config_blob["completion"][
            "params"
        ]["instructions"]

        run = _make_completed_run(
            db=db,
            config_id=config_id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text

        # Version 1 still exists and is unchanged
        db.expire_all()
        still_v1 = crud.read_one(version_number=1)
        assert still_v1 is not None
        assert (
            still_v1.config_blob["completion"]["params"]["instructions"]
            == original_blob_instructions
        )

        # Version 2 now exists and is marked as AI-generated via commit_message
        v2 = crud.read_one(version_number=2)
        assert v2 is not None
        assert v2.commit_message is not None
        assert v2.commit_message.startswith(AI_GENERATED_MARKER)


# ── FR-15 — cap enforced and disclosed ───────────────────────────────────────


class TestWeakQuestionCap:
    """FR-15: > MAX_WEAK_QUESTIONS weak questions → only first MAX_WEAK_QUESTIONS
    sent to the LLM and the cap is applied before calling the model."""

    def test_cap_applied_when_over_limit(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """With 55 weak questions only MAX_WEAK_QUESTIONS are passed to the LLM
        and the response is still 201 (the new version is persisted)."""
        num_questions = 55
        traces = [
            _make_trace(
                question_id=q,
                trace_id=f"tq{q}",
                metric_name=COSINE_SCORE_NAME,
                metric_value=0.1,  # well below threshold
                category="Weak",
                question=f"Question number {q}?",
            )
            for q in range(1, num_questions + 1)
        ]
        score = _score_payload(
            traces=traces,
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.1,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        mock_client = _make_anthropic_mock()
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=mock_client,
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text

        # Inspect the user message passed to the LLM to confirm truncation was applied.
        # The prompt lists each weak question as a numbered item; there should be at most
        # MAX_WEAK_QUESTIONS items in the prompt content.
        call_args = mock_client.messages.create.call_args
        assert call_args is not None, "LLM mock was not called"
        messages_arg = (
            call_args.kwargs.get("messages") or call_args.args[0]
            if call_args.args
            else []
        )
        if not messages_arg and call_args.kwargs.get("messages"):
            messages_arg = call_args.kwargs["messages"]
        # Extract the user message content
        user_content = ""
        for msg in call_args.kwargs.get("messages") or []:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        cap = settings.PROMPT_IMPROVEMENT_MAX_WEAK_QUESTIONS
        # The prompt numbers each question; the last item should be at most cap
        assert f"{cap}." in user_content, (
            f"Expected at most {cap} questions in the LLM prompt but did not find "
            f"item #{cap}. Content snippet: {user_content[:500]}"
        )
        # Confirm there are no more than cap items (item cap+1 should not appear)
        assert (
            f"{cap + 1}." not in user_content
        ), f"Found item #{cap + 1} in the LLM prompt — truncation was not applied"

        # The new version commit_message still carries provenance
        body = resp.json()["data"]
        assert body["commit_message"].startswith(AI_GENERATED_MARKER)


# ── FR-16 — tenant isolation ──────────────────────────────────────────────────


class TestTenantIsolation:
    """FR-16: run not in caller's org+project → 404."""

    def test_run_from_different_project_returns_404(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        superuser_api_key: TestAuthContext,
    ) -> None:
        """A run belonging to the superuser's project is invisible to the normal user."""
        from app.tests.utils.test_data import create_test_evaluation_dataset

        su_dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=superuser_api_key.organization_id,
            project_id=superuser_api_key.project_id,
        )
        su_config = _make_config_with_instructions(
            db=db,
            project_id=superuser_api_key.project_id,
        )
        su_run = _make_completed_run(
            db=db,
            config_id=su_config.id,
            config_version=1,
            organization_id=superuser_api_key.organization_id,
            project_id=superuser_api_key.project_id,
            dataset_id=su_dataset.id,
        )

        # Normal user headers cannot see superuser's run
        resp = client.post(
            IMPROVE_URL.format(evaluation_id=su_run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
            headers=headers,
        )

        assert resp.status_code == 404, resp.text
        assert "evaluation_not_found" in resp.json()["error"]


# ── FR-17 — repeatable iteration ─────────────────────────────────────────────


class TestRepeatableIteration:
    """FR-17: running improvement twice creates a further version at next number."""

    def test_second_improvement_creates_next_version(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        config_id = config_with_instructions.id

        run1 = _make_completed_run(
            db=db,
            config_id=config_id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp1 = client.post(
                IMPROVE_URL.format(evaluation_id=run1.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )
        assert resp1.status_code == 201, resp1.text
        assert resp1.json()["data"]["version"] == 2

        # Create a second run pointing at version 2
        run2 = _make_completed_run(
            db=db,
            config_id=config_id,
            config_version=2,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp2 = client.post(
                IMPROVE_URL.format(evaluation_id=run2.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )
        assert resp2.status_code == 201, resp2.text
        assert resp2.json()["data"]["version"] == 3


# ── 502 — prompt generation failures ─────────────────────────────────────────


class TestPromptGenerationFailures:
    """502 prompt_generation_failed: LLM raises an exception or returns unusable output."""

    def test_llm_exception_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """If messages.create raises, the endpoint returns 502."""
        failing_client = MagicMock()
        failing_client.messages.create.side_effect = RuntimeError("API key invalid")

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=failing_client,
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"]

    def test_non_json_llm_response_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """If messages.create returns non-JSON text, the endpoint returns 502."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock("This is not valid JSON at all."),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"]

    def test_missing_instructions_key_in_llm_response_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        """If LLM JSON response lacks 'improved_instructions', returns 502."""
        bad_response = json.dumps({"rationale": "Only rationale, no instructions."})

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(bad_response),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"]

    def test_missing_anthropic_credentials_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        # Note: no anthropic_creds fixture here, so the platform key stays unset
    ) -> None:
        """If the platform Anthropic key is not configured, returns 502."""
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
            headers=headers,
        )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"]


# ── regression: improve derives from SOURCE version, not latest ──────────────


class TestImprovesSourceNotLatest:
    """Regression: the new version must derive its non-instruction fields from the
    SOURCE config version (the one the run evaluated), not the LATEST version.

    Before the fix, `create_or_raise` was called without a base blob, so the
    merge used the latest version; a non-instruction field that diverged between
    source and latest would leak into the improved version.  The fix calls
    `create_from_blob_or_raise(source_version.config_blob, ...)` so the base is
    always the source.
    """

    def test_new_version_derives_model_from_source_not_latest(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        anthropic_creds: None,
    ) -> None:
        """Version 1 (source) has no top_p; version 2 (latest) adds top_p=0.9.
        The run is evaluated against version 1.  The improved version (3) must
        NOT carry top_p (absent in source), proving it derived from source (v1),
        not from the latest (v2).

        Before the fix: create_or_raise merged onto the latest version, so
        top_p=0.9 from v2 would leak into v3 because new_blob (built from source)
        doesn't have top_p to override it.
        After the fix: create_from_blob_or_raise uses source_blob as the merge
        base, so top_p never enters the result.
        """
        from app.crud.config import ConfigCrud
        from app.models.config.version import ConfigVersionUpdate
        from app.models.llm.request import ConfigBlob
        from app.models.llm import KaapiCompletionConfig

        _SOURCE_INSTRUCTIONS = "You are a helpful assistant. Answer clearly."
        _LATEST_INSTRUCTIONS = "You are an expert. Be concise and technical."
        _SHARED_MODEL = "gpt-4o"
        # top_p is a field present in TextLLMParams (optional); only v2 sets it
        _LATEST_ONLY_TOP_P = 0.9

        # ── version 1 (source) — no top_p ────────────────────────────────────
        source_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                type="text",
                params={
                    "model": _SHARED_MODEL,
                    "temperature": 0.5,
                    "instructions": _SOURCE_INSTRUCTIONS,
                    "knowledge_base_ids": ["vs_source123"],
                },
            )
        )
        from app.models.config.config import ConfigCreate, ConfigTag

        config_create = ConfigCreate(
            name=f"test-config-source-not-latest-{random_lower_string()}",
            description="Regression test: improve-prompt must branch from source",
            config_blob=source_blob,
            commit_message="Source version (v1) — no top_p",
            tag=ConfigTag.DEFAULT,
        )
        config_crud = ConfigCrud(session=db, project_id=auth.project_id)
        config, _ = config_crud.create_or_raise(config_create)

        # ── version 2 (latest) — adds top_p that source does not have ────────
        latest_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                type="text",
                params={
                    "model": _SHARED_MODEL,
                    "temperature": 0.5,
                    "instructions": _LATEST_INSTRUCTIONS,
                    "knowledge_base_ids": ["vs_source123"],
                    "top_p": _LATEST_ONLY_TOP_P,
                },
            )
        )
        version_crud = ConfigVersionCrud(
            session=db,
            config_id=config.id,
            project_id=auth.project_id,
        )
        version_crud.create_or_raise(
            ConfigVersionUpdate(
                config_blob=latest_blob.model_dump(),
                commit_message="Latest version (v2) — adds top_p",
            )
        )

        # Sanity check: source (v1) has no top_p; latest (v2) has top_p
        v1 = version_crud.read_one(version_number=1)
        v2 = version_crud.read_one(version_number=2)
        assert v1 is not None
        assert v2 is not None
        assert v1.config_blob["completion"]["params"].get("top_p") is None
        assert v2.config_blob["completion"]["params"].get("top_p") == _LATEST_ONLY_TOP_P

        # ── run evaluated against version 1 (source) ─────────────────────────
        score = _score_payload(
            traces=[
                _make_trace(
                    question_id=1,
                    trace_id="t-regression-1",
                    metric_name=COSINE_SCORE_NAME,
                    metric_value=0.2,
                    category="Regression",
                    question="What is the capital of France?",
                    llm_answer="Lyon",
                    ground_truth_answer="Paris",
                )
            ],
            summary_scores=[
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": 0.2,
                    "std": 0.0,
                    "data_type": SCORE_DATA_TYPE_NUMERIC,
                }
            ],
        )
        run = _make_completed_run(
            db=db,
            config_id=config.id,
            config_version=1,  # explicitly points at the SOURCE version, not latest
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score=score,
        )

        # ── call the endpoint ─────────────────────────────────────────────────
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                json={"metric": COSINE_SCORE_NAME, "threshold": 0.7},
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        # New version is latest+1 = 3
        assert body["version"] == 3

        # ── fetch new version from DB and assert it derived from source (v1) ─
        from sqlmodel import select as sql_select

        stmt = sql_select(ConfigVersion).where(ConfigVersion.id == body["id"])
        new_version = db.exec(stmt).one()
        new_params = new_version.config_blob["completion"]["params"]

        # The LLM's improved instructions must have been applied
        assert new_params["instructions"] == _IMPROVED_INSTRUCTIONS

        # top_p must NOT be present in the new version: it was absent in source (v1)
        # and must not have leaked in from latest (v2).
        # On the old buggy code, create_or_raise used the latest blob as the merge
        # base, so top_p=0.9 from v2 would have leaked into v3.
        assert new_params.get("top_p") is None, (
            f"top_p={new_params.get('top_p')!r} leaked from latest (v2) into the "
            "improved version — new version derived from latest instead of source (v1)"
        )
