"""Tests for the async prompt-improvement feature (callback-URL / webhook flow).

The endpoint is:
  - POST /evaluations/{id}/improve-prompt → 202 + a job handle
    (LLMJobImmediatePublic). Requires a JSON body {"callback_url": "https://..."}.
    The result is delivered to callback_url once the worker finishes — there is
    no GET poll route anymore.
  - execute_prompt_improvement → the Celery worker entrypoint that does the
    Anthropic round-trip, mints the new config_version, and fires send_callback
    on every exit (SUCCESS, timeout, generic failure, redelivery no-op).

HTTP boundaries mocked (patched as bound in the service module):
- ClaudeProvider.create_client (fake Anthropic client) OR _draft_improved_prompt
- get_cloud_storage / load_json_from_object_store (traces)
- start_prompt_improvement (the Celery enqueue helper) — never touch a broker
- send_callback + get_webhook_secret — never make real outbound HTTP
- validate_callback_url at the route call site — real DNS is avoided for the
  happy/domain paths; the SSRF-rejection tests use the real validator with
  literal hosts that resolve without network.
- Session (the worker opens its own Session(engine); we redirect it at the
  transactional db fixture, matching the doctransformer worker tests)

DB is real (transactional db fixture; rolls back after each test).
"""

import contextlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Iterator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from fastapi import HTTPException
from gevent import Timeout
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.config.version import ConfigVersionCrud
from app.crud.jobs import JobCrud
from app.models import EvaluationDataset, EvaluationRun
from app.models.config.config import ConfigTag
from app.models.job import Job, JobStatus, JobType
from app.services.evaluations.prompt_improvement import (
    AI_GENERATED_MARKER,
    COMMIT_MESSAGE_MAX_LENGTH,
    execute_prompt_improvement,
    start_prompt_improvement_job,
    validate_improve_prompt,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_evaluation_dataset
from app.tests.utils.utils import random_lower_string


_SERVICE = "app.services.evaluations.prompt_improvement"
_ROUTE_VALIDATE = "app.api.routes.evaluations.evaluation.validate_callback_url"
POST_URL = "/api/v1/evaluations/{evaluation_id}/improve-prompt"
# The deleted GET poll route — any method here must now miss all routes.
OLD_POLL_URL = "/api/v1/evaluations/{evaluation_id}/improve-prompt/{job_id}"

_CALLBACK_URL = "https://example.com/callback"

_IMPROVED_INSTRUCTIONS = "You are an improved assistant. Answer precisely."
_RATIONALE = "Tightened answer scoping to address weak categories."

# Already-parsed trace data — the service loads parsed JSON, not bytes.
_TRACES: list[dict] = [
    {
        "trace_id": "t1",
        "question": "What is the capital?",
        "llm_answer": "Lyon",
        "ground_truth_answer": "Paris",
        "category": "Geography",
        "scores": [{"name": "cosine_similarity", "value": 0.3, "unscoreable": False}],
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
    draft: MagicMock | None = None,
    claude_client: MagicMock | None = None,
    traces: Any = _TRACES,
) -> Iterator[WorkerEnv]:
    """Redirect the worker's Session(engine) at the test db and mock its boundaries.

    ``env.llm`` is the draft mock when ``draft`` is given (LLM step stubbed
    wholesale), otherwise the fake Claude client (real _draft_improved_prompt
    exercised). ``env.callback`` is the patched send_callback — no real HTTP ever.
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

        if draft is not None:
            stack.enter_context(patch(f"{_SERVICE}._draft_improved_prompt", draft))
            yield WorkerEnv(llm=draft, callback=callback)
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
        description="Test configuration for improve-prompt",
        config_blob=config_blob,
        commit_message="Initial version",
        tag=ConfigTag.DEFAULT,
    )
    config, _ = ConfigCrud(session=db, project_id=project_id).create_or_raise(
        config_create
    )
    return config


def _make_completed_run(
    db: Session,
    config_id: Any,
    config_version: int | None,
    organization_id: int,
    project_id: int,
    dataset_id: int,
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


class TestValidateImprovePrompt:
    """Fast DB precondition checks — no LLM, no trace download."""

    def test_missing_run_raises_404(self, db: Session, auth: TestAuthContext) -> None:
        with pytest.raises(HTTPException) as exc:
            validate_improve_prompt(
                session=db,
                evaluation_id=9999999,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 404
        assert "evaluation_not_found" in exc.value.detail

    @pytest.mark.parametrize("status", ["pending", "processing", "failed"])
    def test_non_completed_raises_409(
        self,
        status: str,
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
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
        with pytest.raises(HTTPException) as exc:
            validate_improve_prompt(
                session=db,
                evaluation_id=run.id,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 409
        assert "evaluation_not_completed" in exc.value.detail

    @pytest.mark.parametrize("trace_url", [None, ""])
    def test_missing_trace_url_raises_422(
        self,
        trace_url: str | None,
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
    ) -> None:
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score_trace_url=trace_url,
        )
        with pytest.raises(HTTPException) as exc:
            validate_improve_prompt(
                session=db,
                evaluation_id=run.id,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 422
        assert "traces_not_available" in exc.value.detail

    def test_missing_config_reference_raises_422(
        self,
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
    ) -> None:
        run = _make_completed_run(
            db=db,
            config_id=None,
            config_version=None,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )
        with pytest.raises(HTTPException) as exc:
            validate_improve_prompt(
                session=db,
                evaluation_id=run.id,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 422
        assert "source_config_unavailable" in exc.value.detail

    def test_unresolvable_config_version_raises_409(
        self,
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
    ) -> None:
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=99,  # version does not exist
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )
        with pytest.raises(HTTPException) as exc:
            validate_improve_prompt(
                session=db,
                evaluation_id=run.id,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 409
        assert "source_config_unavailable" in exc.value.detail


class TestStartJobRoute:
    """POST → 202 + job handle; enqueue behavior and fast-fail without enqueue."""

    def test_returns_202_with_job_handle_and_persists_job(
        self,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        with patch(_ROUTE_VALIDATE), patch(
            f"{_SERVICE}.start_prompt_improvement", return_value="task-1"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=completed_run.id),
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
        assert kwargs["project_id"] == auth.project_id
        assert kwargs["job_id"] == job_id
        assert kwargs["organization_id"] == auth.organization_id
        assert kwargs["evaluation_id"] == completed_run.id
        assert kwargs["callback_url"] == _CALLBACK_URL

    def test_missing_callback_url_returns_422(
        self,
        client: Any,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        with patch(f"{_SERVICE}.start_prompt_improvement") as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=completed_run.id),
                headers=headers,
                json={},
            )
        assert resp.status_code == 422, resp.text
        enqueue.assert_not_called()

    @pytest.mark.parametrize(
        "callback_url",
        [
            "http://example.com/callback",  # non-HTTPS scheme
            "https://10.0.0.1/callback",  # RFC 1918 private IP
            "https://127.0.0.1/callback",  # loopback
        ],
    )
    def test_invalid_callback_url_rejected_by_ssrf_guard(
        self,
        callback_url: str,
        client: Any,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        # Real validate_callback_url runs; literal hosts resolve without DNS.
        with patch(f"{_SERVICE}.start_prompt_improvement") as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=completed_run.id),
                headers=headers,
                json={"callback_url": callback_url},
            )
        assert resp.status_code == 422, resp.text
        assert "invalid_callback_url" in resp.json()["error"], resp.text
        enqueue.assert_not_called()

    def test_enqueue_failure_marks_job_failed_and_raises_500(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        with patch(
            f"{_SERVICE}.start_prompt_improvement",
            side_effect=RuntimeError("broker down"),
        ):
            with pytest.raises(HTTPException) as exc:
                start_prompt_improvement_job(
                    session=db,
                    evaluation_id=completed_run.id,
                    organization_id=auth.organization_id,
                    project_id=auth.project_id,
                    callback_url=_CALLBACK_URL,
                )

        assert exc.value.status_code == 500

        job = db.exec(
            select(Job)
            .where(Job.project_id == auth.project_id)
            .where(Job.job_type == JobType.PROMPT_IMPROVEMENT)
        ).one()
        assert job.status == JobStatus.FAILED
        assert job.error_message is not None

    @pytest.mark.parametrize(
        "make_run_kwargs,expected_status,error_token",
        [
            (None, 404, "evaluation_not_found"),
            ({"status": "pending"}, 409, "evaluation_not_completed"),
            ({"score_trace_url": None}, 422, "traces_not_available"),
        ],
    )
    def test_fast_validation_error_does_not_enqueue(
        self,
        make_run_kwargs: dict | None,
        expected_status: int,
        error_token: str,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        if make_run_kwargs is None:
            evaluation_id = 9999999
        else:
            run = _make_completed_run(
                db=db,
                config_id=config_with_instructions.id,
                config_version=1,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
                dataset_id=dataset.id,
                **make_run_kwargs,
            )
            evaluation_id = run.id

        with patch(_ROUTE_VALIDATE), patch(
            f"{_SERVICE}.start_prompt_improvement"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=evaluation_id),
                headers=headers,
                json={"callback_url": _CALLBACK_URL},
            )

        assert resp.status_code == expected_status, resp.text
        assert error_token in resp.json()["error"], resp.text
        enqueue.assert_not_called()

        no_job = db.exec(
            select(Job)
            .where(Job.project_id == auth.project_id)
            .where(Job.job_type == JobType.PROMPT_IMPROVEMENT)
        ).first()
        assert no_job is None


class TestExecuteWorker:
    """Worker entrypoint: PROCESSING → SUCCESS/FAILED, callback delivery, redelivery."""

    def _pending_job(self, db: Session, project_id: int) -> Job:
        return JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT,
            trace_id="N/A",
            project_id=project_id,
        )

    def test_happy_path_mints_new_version_and_records_meta(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        with _worker_env(db) as env:
            result = execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
                task_id="celery-task-1",
                callback_url=_CALLBACK_URL,
            )

        assert result["success"] is True

        refreshed = db.get(Job, job.id)
        assert refreshed.status == JobStatus.SUCCESS
        assert refreshed.task_id == "celery-task-1"
        assert refreshed.meta["version"] == 2
        assert refreshed.meta["rationale"] == _RATIONALE

        crud = ConfigVersionCrud(
            session=db, config_id=completed_run.config_id, project_id=auth.project_id
        )
        v2 = crud.read_one(version_number=2)
        assert v2 is not None
        assert (
            v2.config_blob["completion"]["params"]["instructions"]
            == _IMPROVED_INSTRUCTIONS
        )
        assert v2.commit_message.startswith(AI_GENERATED_MARKER)
        assert completed_run.run_name in v2.commit_message
        assert _RATIONALE in v2.commit_message

        # structured-output request shape is what makes the JSON parse reliable
        output_config = env.llm.messages.create.call_args.kwargs["output_config"]
        assert output_config["format"]["type"] == "json_schema"

    def test_success_fires_callback_with_config_version(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        with _worker_env(db) as env:
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
                callback_url=_CALLBACK_URL,
            )

        assert env.callback.call_args.args[0] == _CALLBACK_URL
        payload = _callback_payload(env.callback)
        assert payload["success"] is True
        assert payload["data"]["status"] == JobStatus.SUCCESS.value
        assert payload["data"]["error_message"] is None
        assert payload["data"]["config_version"] is not None
        assert payload["data"]["config_version"]["version"] == 2

    def test_long_rationale_truncated_in_commit_message(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)
        long_rationale = "x" * (COMMIT_MESSAGE_MAX_LENGTH + 200)
        draft = MagicMock(return_value=(_IMPROVED_INSTRUCTIONS, long_rationale))

        with _worker_env(db, draft=draft):
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
                callback_url=_CALLBACK_URL,
            )

        v2 = ConfigVersionCrud(
            session=db, config_id=completed_run.config_id, project_id=auth.project_id
        ).read_one(version_number=2)
        assert len(v2.commit_message) == COMMIT_MESSAGE_MAX_LENGTH
        assert v2.commit_message.startswith(AI_GENERATED_MARKER)

    def test_llm_runtime_error_marks_failed_and_fires_failure_callback(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)
        draft = MagicMock(side_effect=RuntimeError("prompt_generation_failed: boom"))

        with _worker_env(db, draft=draft) as env:
            with pytest.raises(RuntimeError):
                execute_prompt_improvement(
                    project_id=auth.project_id,
                    job_id=str(job.id),
                    organization_id=auth.organization_id,
                    evaluation_id=completed_run.id,
                    callback_url=_CALLBACK_URL,
                )

        refreshed = db.get(Job, job.id)
        assert refreshed.status == JobStatus.FAILED
        assert "prompt_generation_failed" in refreshed.error_message

        payload = _callback_payload(env.callback)
        assert payload["success"] is False
        assert payload["data"]["status"] == JobStatus.FAILED.value
        assert payload["data"]["config_version"] is None
        assert payload["data"]["error_message"]
        assert "prompt_generation_failed" in payload["data"]["error_message"]

        # no version minted on failure
        v2 = ConfigVersionCrud(
            session=db, config_id=completed_run.config_id, project_id=auth.project_id
        ).read_one(version_number=2)
        assert v2 is None

    def test_trace_download_failure_fires_failure_callback(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        with _worker_env(db, traces=None) as env:
            with pytest.raises(RuntimeError):
                execute_prompt_improvement(
                    project_id=auth.project_id,
                    job_id=str(job.id),
                    organization_id=auth.organization_id,
                    evaluation_id=completed_run.id,
                    callback_url=_CALLBACK_URL,
                )

        refreshed = db.get(Job, job.id)
        assert refreshed.status == JobStatus.FAILED

        payload = _callback_payload(env.callback)
        assert payload["success"] is False
        assert payload["data"]["status"] == JobStatus.FAILED.value
        assert "trace_download_failed" in payload["data"]["error_message"]

    @pytest.mark.parametrize(
        "exc", [Timeout(1), SoftTimeLimitExceeded()], ids=["gevent", "celery"]
    )
    def test_timeout_marks_failed_and_fires_failure_callback(
        self,
        exc: Exception,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)
        draft = MagicMock(side_effect=exc)

        with _worker_env(db, draft=draft) as env:
            with pytest.raises(type(exc)):
                execute_prompt_improvement(
                    project_id=auth.project_id,
                    job_id=str(job.id),
                    organization_id=auth.organization_id,
                    evaluation_id=completed_run.id,
                    callback_url=_CALLBACK_URL,
                )

        refreshed = db.get(Job, job.id)
        assert refreshed.status == JobStatus.FAILED
        assert refreshed.error_message == "Task exceeded soft time limit"

        payload = _callback_payload(env.callback)
        assert payload["success"] is False
        assert payload["data"]["status"] == JobStatus.FAILED.value
        assert payload["data"]["error_message"] == "Task exceeded soft time limit"

    def test_redelivery_of_success_job_is_noop_and_resends_callback(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        # First run mints v2 and lands SUCCESS.
        with _worker_env(db) as env:
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
                callback_url=_CALLBACK_URL,
            )
        assert env.llm.messages.create.call_count == 1

        crud = ConfigVersionCrud(
            session=db, config_id=completed_run.config_id, project_id=auth.project_id
        )
        assert crud.read_one(version_number=3) is None
        first_meta = db.get(Job, job.id).meta

        # Redelivery: SUCCESS job must not re-call the LLM or mint a duplicate,
        # but at-least-once delivery re-sends the success callback.
        draft = MagicMock(side_effect=AssertionError("LLM must not be re-called"))
        with _worker_env(db, draft=draft) as env:
            result = execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
                callback_url=_CALLBACK_URL,
            )

        draft.assert_not_called()
        assert result["success"] is True
        assert result["version"] == first_meta["version"]
        assert crud.read_one(version_number=3) is None

        payload = _callback_payload(env.callback)
        assert payload["success"] is True
        assert payload["data"]["status"] == JobStatus.SUCCESS.value
        assert payload["data"]["config_version"]["version"] == first_meta["version"]


class TestPollRouteRemoved:
    """The GET poll endpoint was removed with the switch to callbacks."""

    def test_old_poll_path_no_longer_routed(
        self,
        client: Any,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        resp = client.get(
            OLD_POLL_URL.format(evaluation_id=completed_run.id, job_id=uuid4()),
            headers=headers,
        )
        assert resp.status_code == 404, resp.text
