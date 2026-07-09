"""Tests for the async prompt-improvement feature.

The endpoint is now split into:
  - POST /evaluations/{id}/improve-prompt → 202 + a job handle
    (LLMJobImmediatePublic), backed by start_prompt_improvement_job.
  - GET  /evaluations/{id}/improve-prompt/{job_id} → poll for the result
    (PromptImprovementJobPublic).
  - execute_prompt_improvement → the Celery worker entrypoint that does the
    Anthropic round-trip and mints the new config_version.

HTTP boundaries mocked (patched as bound in the service module):
- ClaudeProvider.create_client (fake Anthropic client) OR _draft_improved_prompt
- get_cloud_storage / load_json_from_object_store (traces)
- start_prompt_improvement (the Celery enqueue helper) — never touch a broker
- Session (the worker opens its own Session(engine); we redirect it at the
  transactional db fixture, matching the doctransformer worker tests)

DB is real (transactional db fixture; rolls back after each test).
"""

import contextlib
import json
from contextlib import ExitStack
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
from app.models import ConfigVersion, EvaluationDataset, EvaluationRun
from app.models.config.config import ConfigTag
from app.models.job import Job, JobStatus, JobType, JobUpdate
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
POST_URL = "/api/v1/evaluations/{evaluation_id}/improve-prompt"
POLL_URL = "/api/v1/evaluations/{evaluation_id}/improve-prompt/{job_id}"

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


@contextlib.contextmanager
def _worker_env(
    db: Session,
    *,
    draft: MagicMock | None = None,
    claude_client: MagicMock | None = None,
    traces: Any = _TRACES,
) -> Iterator[MagicMock]:
    """Redirect the worker's Session(engine) at the test db and mock its boundaries.

    Yields the draft mock when ``draft`` is given (LLM step stubbed wholesale),
    otherwise the fake Claude client (real _draft_improved_prompt exercised).
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
        if draft is not None:
            stack.enter_context(patch(f"{_SERVICE}._draft_improved_prompt", draft))
            yield draft
        else:
            if claude_client is None:
                claude_client = _make_fake_claude_client()
            stack.enter_context(
                patch(
                    f"{_SERVICE}.ClaudeProvider.create_client",
                    return_value=claude_client,
                )
            )
            yield claude_client


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
        with patch(
            f"{_SERVICE}.start_prompt_improvement", return_value="task-1"
        ) as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=completed_run.id), headers=headers
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

        with patch(f"{_SERVICE}.start_prompt_improvement") as enqueue:
            resp = client.post(
                POST_URL.format(evaluation_id=evaluation_id), headers=headers
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
    """Worker entrypoint: PROCESSING → SUCCESS/FAILED, idempotent redelivery."""

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

        with _worker_env(db) as fake_client:
            result = execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
                task_id="celery-task-1",
            )

        assert result["success"] is True

        refreshed = db.get(Job, job.id)
        assert refreshed.status == JobStatus.SUCCESS
        assert refreshed.task_id == "celery-task-1"
        assert refreshed.meta["version"] == 2
        assert refreshed.meta["rationale"] == _RATIONALE
        assert refreshed.meta["config_version_id"]

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
        output_config = fake_client.messages.create.call_args.kwargs["output_config"]
        assert output_config["format"]["type"] == "json_schema"

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
            )

        v2 = ConfigVersionCrud(
            session=db, config_id=completed_run.config_id, project_id=auth.project_id
        ).read_one(version_number=2)
        assert len(v2.commit_message) == COMMIT_MESSAGE_MAX_LENGTH
        assert v2.commit_message.startswith(AI_GENERATED_MARKER)

    def test_llm_runtime_error_marks_failed_and_reraises(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)
        draft = MagicMock(side_effect=RuntimeError("prompt_generation_failed: boom"))

        with _worker_env(db, draft=draft):
            with pytest.raises(RuntimeError):
                execute_prompt_improvement(
                    project_id=auth.project_id,
                    job_id=str(job.id),
                    organization_id=auth.organization_id,
                    evaluation_id=completed_run.id,
                )

        refreshed = db.get(Job, job.id)
        assert refreshed.status == JobStatus.FAILED
        assert "prompt_generation_failed" in refreshed.error_message

        # no version minted on failure
        v2 = ConfigVersionCrud(
            session=db, config_id=completed_run.config_id, project_id=auth.project_id
        ).read_one(version_number=2)
        assert v2 is None

    @pytest.mark.parametrize(
        "exc", [Timeout(1), SoftTimeLimitExceeded()], ids=["gevent", "celery"]
    )
    def test_timeout_marks_failed_and_reraises(
        self,
        exc: Exception,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)
        draft = MagicMock(side_effect=exc)

        with _worker_env(db, draft=draft):
            with pytest.raises(type(exc)):
                execute_prompt_improvement(
                    project_id=auth.project_id,
                    job_id=str(job.id),
                    organization_id=auth.organization_id,
                    evaluation_id=completed_run.id,
                )

        refreshed = db.get(Job, job.id)
        assert refreshed.status == JobStatus.FAILED
        assert refreshed.error_message == "Task exceeded soft time limit"

    def test_redelivery_of_success_job_is_noop(
        self,
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = self._pending_job(db, auth.project_id)

        # First run mints v2 and lands SUCCESS.
        with _worker_env(db) as fake_client:
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
            )
        assert fake_client.messages.create.call_count == 1

        crud = ConfigVersionCrud(
            session=db, config_id=completed_run.config_id, project_id=auth.project_id
        )
        assert crud.read_one(version_number=3) is None
        first_meta = db.get(Job, job.id).meta

        # Redelivery: SUCCESS job must not re-call the LLM or mint a duplicate.
        draft = MagicMock(side_effect=AssertionError("LLM must not be re-called"))
        with _worker_env(db, draft=draft):
            result = execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
            )

        draft.assert_not_called()
        assert result["success"] is True
        assert result["version"] == first_meta["version"]
        assert crud.read_one(version_number=3) is None


class TestPollStatus:
    """GET poll endpoint across job states and tenant isolation."""

    def test_unknown_job_returns_404(
        self,
        client: Any,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        resp = client.get(
            POLL_URL.format(evaluation_id=completed_run.id, job_id=uuid4()),
            headers=headers,
        )
        assert resp.status_code == 404, resp.text

    def test_cross_project_job_returns_404(
        self,
        client: Any,
        headers: dict[str, str],
        db: Session,
        superuser_api_key: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        other_job = JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT,
            project_id=superuser_api_key.project_id,
        )
        resp = client.get(
            POLL_URL.format(evaluation_id=completed_run.id, job_id=other_job.id),
            headers=headers,  # normal-user headers
        )
        assert resp.status_code == 404, resp.text

    @pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.PROCESSING])
    def test_in_progress_job_has_null_config_version(
        self,
        status: JobStatus,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT, project_id=auth.project_id
        )
        if status != JobStatus.PENDING:
            JobCrud(session=db).update(job.id, JobUpdate(status=status))

        resp = client.get(
            POLL_URL.format(evaluation_id=completed_run.id, job_id=job.id),
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["status"] == status.value
        assert body["config_version"] is None
        assert body["error_message"] is None

    def test_failed_job_returns_error_message(
        self,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT, project_id=auth.project_id
        )
        JobCrud(session=db).update(
            job.id,
            JobUpdate(
                status=JobStatus.FAILED, error_message="prompt_generation_failed"
            ),
        )
        resp = client.get(
            POLL_URL.format(evaluation_id=completed_run.id, job_id=job.id),
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["status"] == JobStatus.FAILED.value
        assert body["config_version"] is None
        assert body["error_message"] == "prompt_generation_failed"

    def test_success_job_returns_nested_config_version(
        self,
        client: Any,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        completed_run: EvaluationRun,
    ) -> None:
        job = JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT, project_id=auth.project_id
        )
        with _worker_env(db):
            execute_prompt_improvement(
                project_id=auth.project_id,
                job_id=str(job.id),
                organization_id=auth.organization_id,
                evaluation_id=completed_run.id,
            )

        resp = client.get(
            POLL_URL.format(evaluation_id=completed_run.id, job_id=job.id),
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["status"] == JobStatus.SUCCESS.value
        assert body["error_message"] is None
        assert body["config_version"] is not None
        assert body["config_version"]["version"] == 2
        assert (
            body["config_version"]["config_blob"]["completion"]["params"][
                "instructions"
            ]
            == _IMPROVED_INSTRUCTIONS
        )
