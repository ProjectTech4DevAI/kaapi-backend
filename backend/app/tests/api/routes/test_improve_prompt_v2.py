"""Tests for the v2 prompt-improvement feature (judged runs, three-metric judge).

The v2 endpoint is:
  - POST /api/v2/evaluations/{id}/improve-prompt → 202 + a job handle. Requires a
    JSON body {"callback_url": "https://..."} and a *judged* (is_judge_run) run;
    calls start_prompt_improvement_job(..., require_judge_run=True).
  - execute_prompt_improvement (shared with v1) → branches on run.is_judge_run:
    a judge run drafts via _draft_improved_prompt_v2 and delivers a
    PromptRecommendationJobPublic callback; a non-judge run keeps the v1 shape.

HTTP boundaries mocked (patched as bound in the modules under test):
- ClaudeProvider.create_client (fake Anthropic client)
- get_cloud_storage / load_json_from_object_store (v2 judge traces)
- start_prompt_improvement (the Celery enqueue helper) — never touch a broker
- send_callback + get_webhook_secret — never make real outbound HTTP
- validate_callback_url at the v2 route call site
- Session (the worker opens its own Session(engine); redirected at the db fixture)

DB is real (transactional db fixture; rolls back after each test).
"""

import contextlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations.score import (
    GROUND_TRUTH_SCORE_NAME,
    KNOWLEDGE_BASE_SCORE_NAME,
    PROMPT_SCORE_NAME,
)
from app.crud.jobs import JobCrud
from app.models import EvaluationDataset, EvaluationRun
from app.models.config.config import ConfigTag
from app.models.job import Job, JobStatus, JobType
from app.services.evaluations.prompt_improvement import (
    AI_GENERATED_MARKER,
    execute_prompt_improvement,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_evaluation_dataset
from app.tests.utils.utils import random_lower_string

_SERVICE = "app.services.evaluations.prompt_improvement"
_ROUTE_VALIDATE = (
    "app.api.routes.evaluations.prompt_improvement_v2.validate_callback_url"
)
POST_URL = f"{settings.API_V2_STR}/evaluations/{{evaluation_id}}/improve-prompt"

_CALLBACK_URL = "https://example.com/callback"

_IMPROVED_INSTRUCTIONS = "You are an improved assistant. Answer precisely."
_RATIONALE = "Tightened language adherence and grounding to fix weak rows."

# The judge's reasoning strings — asserted verbatim in the v2 LLM message, so kept
# as named constants and reused in the trace fixture (single independent source).
_GT_COMMENT = "missed key fact X"
_PROMPT_COMMENT = "answered in wrong language"
_KB_COMMENT = "Knowledge base not queried."

# v2 judge traces: three Adherence-to-* metrics, KB unscoreable (value "N/A").
_JUDGE_TRACES: list[dict] = [
    {
        "trace_id": "t1",
        "question": "Q",
        "llm_answer": "A",
        "ground_truth_answer": "G",
        "question_id": "q1",
        "scores": [
            {
                "name": GROUND_TRUTH_SCORE_NAME,
                "value": 0.4,
                "data_type": "NUMERIC",
                "comment": _GT_COMMENT,
            },
            {
                "name": PROMPT_SCORE_NAME,
                "value": 0.3,
                "data_type": "NUMERIC",
                "comment": _PROMPT_COMMENT,
            },
            {
                "name": KNOWLEDGE_BASE_SCORE_NAME,
                "value": "N/A",
                "data_type": "CATEGORICAL",
                "comment": _KB_COMMENT,
                "unscoreable": True,
            },
        ],
    }
]

_AUTO_URL = object()  # "generate a valid score_trace_url from the run id after commit"


def _llm_json(
    improved_instructions: str = _IMPROVED_INSTRUCTIONS,
    rationale: str = _RATIONALE,
) -> str:
    return json.dumps(
        {"improved_instructions": improved_instructions, "rationale": rationale}
    )


def _make_fake_claude_client(text_content: str | None = None) -> MagicMock:
    if text_content is None:
        text_content = _llm_json()

    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text_content

    response = MagicMock()
    response.content = [content_block]
    response.id = "msg_test_id"

    client = MagicMock()
    client.messages.create.return_value = response
    return client


@dataclass
class WorkerEnv:
    """Handles yielded by ``_worker_env``: the LLM boundary and the callback sink."""

    llm: MagicMock
    callback: MagicMock


@contextlib.contextmanager
def _worker_env(
    db: Session,
    *,
    draft_v2: MagicMock | None = None,
    claude_client: MagicMock | None = None,
    traces: Any = _JUDGE_TRACES,
) -> Iterator[WorkerEnv]:
    """Redirect the worker's Session(engine) at the test db and mock its boundaries.

    Default: the fake Claude client, so the real _draft_improved_prompt_v2 runs and
    its user message is inspectable. Pass ``draft_v2`` to stub the v2 draft wholesale
    (used to assert it is NOT re-called on redelivery). ``env.callback`` is the
    patched send_callback — no real HTTP ever.
    """
    session_cm = MagicMock()
    session_cm.__enter__.return_value = db
    session_cm.__exit__.return_value = None

    with ExitStack() as stack:
        stack.enter_context(patch(f"{_SERVICE}.Session", return_value=session_cm))
        stack.enter_context(
            patch(f"{_SERVICE}.get_cloud_storage", return_value=MagicMock())
        )
        stack.enter_context(
            patch(f"{_SERVICE}.load_json_from_object_store", return_value=traces)
        )
        stack.enter_context(patch(f"{_SERVICE}.get_webhook_secret", return_value=None))
        callback = stack.enter_context(patch(f"{_SERVICE}.send_callback"))

        if draft_v2 is not None:
            stack.enter_context(
                patch(f"{_SERVICE}._draft_improved_prompt_v2", draft_v2)
            )
            yield WorkerEnv(llm=draft_v2, callback=callback)
        else:
            if claude_client is None:
                claude_client = _make_fake_claude_client()
            stack.enter_context(
                patch(
                    f"{_SERVICE}.ClaudeProvider.create_client",
                    return_value=claude_client,
                )
            )
            yield WorkerEnv(llm=claude_client, callback=callback)


def _callback_payload(callback: MagicMock) -> dict:
    """Single send_callback invocation → the APIResponse envelope it POSTed."""
    assert callback.call_count == 1
    return callback.call_args.args[1]


def _llm_user_message(claude_client: MagicMock) -> str:
    """The single user-message text passed to the mocked Anthropic create()."""
    messages = claude_client.messages.create.call_args.kwargs["messages"]
    return messages[0]["content"]


def _make_config_with_instructions(
    db: Session,
    project_id: int,
    instructions: str = "You are a helpful assistant.",
) -> Any:
    from app.crud.config import ConfigCrud
    from app.models.config.config import ConfigCreate
    from app.models.llm import KaapiCompletionConfig
    from app.models.llm.request import ConfigBlob

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
    config_create = ConfigCreate(
        name=f"test-config-{random_lower_string()}",
        description="Test configuration for improve-prompt v2",
        config_blob=config_blob,
        commit_message="Initial version",
        tag=ConfigTag.DEFAULT,
    )
    config, _ = ConfigCrud(session=db, project_id=project_id).create_or_raise(
        config_create
    )
    return config


def _make_run(
    db: Session,
    config_id: Any,
    config_version: int | None,
    organization_id: int,
    project_id: int,
    dataset_id: int,
    *,
    is_judge_run: bool | None,
    status: str = "completed",
    run_name: str | None = None,
    score_trace_url: Any = _AUTO_URL,
) -> EvaluationRun:
    if run_name is None:
        run_name = f"run-{random_lower_string()}"

    initial_url: str | None = None if score_trace_url is _AUTO_URL else score_trace_url

    run = EvaluationRun(
        run_name=run_name,
        dataset_name=f"ds-{random_lower_string()}",
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        status=status,
        total_items=1,
        type="text",
        organization_id=organization_id,
        project_id=project_id,
        score_trace_url=initial_url,
        is_judge_run=is_judge_run,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if score_trace_url is _AUTO_URL:
        run.score_trace_url = (
            f"s3://test-bucket/evaluations/score/{run.id}/traces_{run.id}.json"
        )
        db.add(run)
        db.commit()
        db.refresh(run)

    return run


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
    monkeypatch.setattr(
        settings, "ANTHROPIC_API_KEY", "sk-ant-test-" + random_lower_string()
    )


@pytest.fixture
def judge_run(
    db: Session,
    auth: TestAuthContext,
    dataset: EvaluationDataset,
    config_with_instructions: Any,
    anthropic_creds: None,
) -> EvaluationRun:
    return _make_run(
        db=db,
        config_id=config_with_instructions.id,
        config_version=1,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        dataset_id=dataset.id,
        is_judge_run=True,
    )


@pytest.fixture
def non_judge_run(
    db: Session,
    auth: TestAuthContext,
    dataset: EvaluationDataset,
    config_with_instructions: Any,
    anthropic_creds: None,
) -> EvaluationRun:
    return _make_run(
        db=db,
        config_id=config_with_instructions.id,
        config_version=1,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        dataset_id=dataset.id,
        is_judge_run=False,
    )


class TestV2Route:
    """POST /api/v2/.../improve-prompt: judged-run gate + request-side validation."""

    def test_judge_run_returns_202_with_job_handle_and_enqueues(
        self,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        judge_run: EvaluationRun,
    ) -> None:
        with patch(_ROUTE_VALIDATE), patch(
            f"{_SERVICE}.start_prompt_improvement", return_value="task-v2-1"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=judge_run.id),
                headers=headers,
                json={"callback_url": _CALLBACK_URL},
            )

        assert resp.status_code == 202, resp.text
        body = resp.json()["data"]
        assert body["status"] == JobStatus.PENDING.value
        job_id = body["job_id"]

        job = JobCrud(session=db).get(job_id=job_id, project_id=auth.project_id)
        assert job is not None
        assert job.job_type == JobType.PROMPT_IMPROVEMENT

        enqueue.assert_called_once()
        kwargs = enqueue.call_args.kwargs
        assert kwargs["job_id"] == job_id
        assert kwargs["evaluation_id"] == judge_run.id
        assert kwargs["callback_url"] == _CALLBACK_URL

    def test_non_judge_run_returns_422_not_a_judge_run(
        self,
        client: Any,
        headers: dict[str, str],
        non_judge_run: EvaluationRun,
    ) -> None:
        with patch(_ROUTE_VALIDATE), patch(
            f"{_SERVICE}.start_prompt_improvement"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=non_judge_run.id),
                headers=headers,
                json={"callback_url": _CALLBACK_URL},
            )

        assert resp.status_code == 422, resp.text
        assert "not_a_judge_run" in resp.json()["error"], resp.text
        enqueue.assert_not_called()

    def test_missing_run_returns_404(
        self,
        client: Any,
        headers: dict[str, str],
    ) -> None:
        with patch(_ROUTE_VALIDATE), patch(
            f"{_SERVICE}.start_prompt_improvement"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=9999999),
                headers=headers,
                json={"callback_url": _CALLBACK_URL},
            )
        assert resp.status_code == 404, resp.text
        assert "evaluation_not_found" in resp.json()["error"], resp.text
        enqueue.assert_not_called()

    def test_non_completed_run_returns_409(
        self,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
    ) -> None:
        run = _make_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            is_judge_run=True,
            status="processing",
        )
        with patch(_ROUTE_VALIDATE), patch(
            f"{_SERVICE}.start_prompt_improvement"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=run.id),
                headers=headers,
                json={"callback_url": _CALLBACK_URL},
            )
        assert resp.status_code == 409, resp.text
        assert "evaluation_not_completed" in resp.json()["error"], resp.text
        enqueue.assert_not_called()

    def test_missing_trace_url_returns_422(
        self,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
    ) -> None:
        run = _make_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            is_judge_run=True,
            score_trace_url=None,
        )
        with patch(_ROUTE_VALIDATE), patch(
            f"{_SERVICE}.start_prompt_improvement"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=run.id),
                headers=headers,
                json={"callback_url": _CALLBACK_URL},
            )
        assert resp.status_code == 422, resp.text
        assert "traces_not_available" in resp.json()["error"], resp.text
        enqueue.assert_not_called()

    @pytest.mark.parametrize(
        "callback_url",
        [
            "http://example.com/callback",  # non-HTTPS scheme
            "https://127.0.0.1/callback",  # loopback (SSRF)
        ],
    )
    def test_invalid_callback_url_rejected_by_ssrf_guard(
        self,
        callback_url: str,
        client: Any,
        headers: dict[str, str],
        judge_run: EvaluationRun,
    ) -> None:
        with patch(f"{_SERVICE}.start_prompt_improvement") as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=judge_run.id),
                headers=headers,
                json={"callback_url": callback_url},
            )
        assert resp.status_code == 422, resp.text
        assert "invalid_callback_url" in resp.json()["error"], resp.text
        enqueue.assert_not_called()


class TestV2Worker:
    """execute_prompt_improvement branches on is_judge_run: v2 draft + v2 callback."""

    def _pending_job(self, db: Session, project_id: int) -> Job:
        return JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT,
            trace_id="N/A",
            project_id=project_id,
        )

    def test_judge_run_mints_version_and_sends_recommendation_callback(
        self,
        db: Session,
        auth: TestAuthContext,
        judge_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        with _worker_env(db) as env:
            result = execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=judge_run.id,
                callback_url=_CALLBACK_URL,
            )

        assert result["success"] is True

        v2 = ConfigVersionCrud(
            session=db, config_id=judge_run.config_id, project_id=auth.project_id
        ).read_one(version_number=2)
        assert v2 is not None
        assert (
            v2.config_blob["completion"]["params"]["instructions"]
            == _IMPROVED_INSTRUCTIONS
        )
        assert v2.commit_message.startswith(AI_GENERATED_MARKER)

        payload = _callback_payload(env.callback)
        assert payload["success"] is True
        assert payload["data"]["status"] == JobStatus.SUCCESS.value
        assert payload["data"]["recommendation_type"] == "prompt"
        assert payload["data"]["config_version"]["version"] == 2

    def test_v2_message_carries_metric_reasoning_and_ignores_na(
        self,
        db: Session,
        auth: TestAuthContext,
        judge_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        with _worker_env(db) as env:
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=judge_run.id,
                callback_url=_CALLBACK_URL,
            )

        message = _llm_user_message(env.llm)

        for metric_name in (
            GROUND_TRUTH_SCORE_NAME,
            PROMPT_SCORE_NAME,
            KNOWLEDGE_BASE_SCORE_NAME,
        ):
            assert metric_name in message

        assert _GT_COMMENT in message
        assert _PROMPT_COMMENT in message

        # The unscoreable KB metric must be presented as ignore-worthy, not a real
        # score: the prompt tells the model to skip value="N/A" / unscoreable metrics.
        assert '"N/A"' in message
        assert "unscoreable" in message
        assert "ignore those" in message

    def test_non_judge_run_uses_v1_shape_no_recommendation_type(
        self,
        db: Session,
        auth: TestAuthContext,
        non_judge_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        # v1 traces carry cosine scores; feed those so the non-judge path is realistic.
        v1_traces = [
            {
                "trace_id": "t1",
                "question": "Q",
                "llm_answer": "A",
                "ground_truth_answer": "G",
                "category": "Geo",
                "scores": [
                    {"name": "cosine_similarity", "value": 0.3, "unscoreable": False}
                ],
            }
        ]
        with _worker_env(db, traces=v1_traces) as env:
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=non_judge_run.id,
                callback_url=_CALLBACK_URL,
            )

        payload = _callback_payload(env.callback)
        assert payload["success"] is True
        assert "recommendation_type" not in payload["data"]

        # v1 draft message does not surface the three-metric judge names.
        message = _llm_user_message(env.llm)
        assert PROMPT_SCORE_NAME not in message
        assert GROUND_TRUTH_SCORE_NAME not in message

    def test_redelivery_of_success_judge_job_resends_v2_callback_without_new_version(
        self,
        db: Session,
        auth: TestAuthContext,
        judge_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        with _worker_env(db) as env:
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=judge_run.id,
                callback_url=_CALLBACK_URL,
            )
        assert env.llm.messages.create.call_count == 1

        crud = ConfigVersionCrud(
            session=db, config_id=judge_run.config_id, project_id=auth.project_id
        )
        first_version = db.get(Job, job.id).meta["version"]
        assert crud.read_one(version_number=first_version + 1) is None

        draft = MagicMock(side_effect=AssertionError("v2 draft must not be re-called"))
        with _worker_env(db, draft_v2=draft) as env:
            result = execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=judge_run.id,
                callback_url=_CALLBACK_URL,
            )

        draft.assert_not_called()
        assert result["success"] is True
        assert result["version"] == first_version
        assert crud.read_one(version_number=first_version + 1) is None

        payload = _callback_payload(env.callback)
        assert payload["success"] is True
        assert payload["data"]["recommendation_type"] == "prompt"
        assert payload["data"]["config_version"]["version"] == first_version
