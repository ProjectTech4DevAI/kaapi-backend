import io
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session, select

from app.crud.language import get_language_by_locale
from app.models import EvaluationDataset, EvaluationRun
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.batch_job import execute_batch_submission
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_run_with_dataset,
    fake_gemini_batch,
    use_test_session,
)
from app.tests.utils.utils import get_non_existent_id

MODULE = "app.services.tts_evaluations.batch_job"
CRUD_BATCH = "app.crud.tts_evaluations.batch"
GEMINI = "gemini-2.5-pro-preview-tts"
SARVAM = "bulbul:v3"
ELEVEN = "eleven_v3"


def _csv(count: int) -> bytes:
    return ("text\n" + "".join(f"sample {i}\n" for i in range(count))).encode()


@pytest.fixture
def boundaries(db: Session) -> Iterator[SimpleNamespace]:
    state = SimpleNamespace(csv=_csv(2), enqueued=[], enqueue_errors={})
    storage = MagicMock()
    storage.stream.side_effect = lambda _url: io.BytesIO(state.csv)

    def enqueue(**kwargs: Any) -> str:
        if kwargs["model"] in state.enqueue_errors:
            raise state.enqueue_errors[kwargs["model"]]
        state.enqueued.append(kwargs)
        return f"task-{len(state.enqueued)}"

    with (
        use_test_session(MODULE, db),
        patch(f"{MODULE}.get_cloud_storage", return_value=storage),
        patch(f"{MODULE}.start_tts_sync_generation", side_effect=enqueue),
    ):
        yield state


def _run(
    db: Session, auth: TestAuthContext, models: list[str], **kwargs: Any
) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        models=models,
        **kwargs,
    )


def _execute(run: EvaluationRun, dataset_id: int | None = None) -> dict[str, Any]:
    return execute_batch_submission(
        project_id=run.project_id,
        job_id=str(run.id),
        task_id="celery-task",
        task_instance=MagicMock(),
        organization_id=run.organization_id,
        dataset_id=dataset_id or run.dataset_id,
        models=run.providers,
    )


def _results(db: Session, run: EvaluationRun, model: str) -> list[TTSResult]:
    stmt = (
        select(TTSResult)
        .where(TTSResult.evaluation_run_id == run.id, TTSResult.provider == model)
        .order_by(TTSResult.id)
    )
    return list(db.exec(stmt).all())


def _row_states(
    db: Session, run: EvaluationRun, model: str
) -> set[tuple[str, str | None]]:
    return {(r.status, r.error_message) for r in _results(db, run, model)}


@pytest.mark.usefixtures("boundaries")
class TestExecuteBatchSubmission:
    def test_mixed_run_creates_rows_and_starts_every_model(
        self, db: Session, user_api_key: TestAuthContext, boundaries: SimpleNamespace
    ) -> None:
        run = _run(db, user_api_key, [GEMINI, SARVAM, ELEVEN])

        with fake_gemini_batch(CRUD_BATCH) as gemini:
            outcome = _execute(run)

        assert (outcome["success"], outcome["result_count"]) == (True, 6)
        for model in (GEMINI, SARVAM, ELEVEN):
            assert [r.sample_text for r in _results(db, run, model)] == [
                "sample 0",
                "sample 1",
            ]
            assert _row_states(db, run, model) == {(JobStatus.PENDING.value, None)}
        assert {r["key"] for r in gemini.submitted[f"models/{GEMINI}"]} == {
            str(r.id) for r in _results(db, run, GEMINI)
        }
        assert [
            (c["model"], c["job_id"], c["language_code"]) for c in boundaries.enqueued
        ] == [(SARVAM, str(run.id), "en-IN"), (ELEVEN, str(run.id), "en-IN")]
        db.refresh(run)
        assert run.status == "processing"

    @pytest.mark.parametrize(
        ("locale", "expected_language"),
        [(None, "en-IN"), ("hi", "hi-IN"), ("or", "od-IN")],
    )
    def test_sync_language_comes_from_dataset_locale(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        boundaries: SimpleNamespace,
        locale: str | None,
        expected_language: str,
    ) -> None:
        language_id = (
            get_language_by_locale(session=db, locale=locale).id if locale else None
        )
        run = _run(db, user_api_key, [SARVAM], language_id=language_id)

        _execute(run)

        assert [c["language_code"] for c in boundaries.enqueued] == [expected_language]

    @pytest.mark.parametrize("models", [[GEMINI], [GEMINI, SARVAM]])
    def test_gemini_submission_failure_fails_only_gemini_rows(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        boundaries: SimpleNamespace,
        models: list[str],
    ) -> None:
        run = _run(db, user_api_key, models)

        with patch(f"{CRUD_BATCH}.GeminiClient") as gemini_client:
            gemini_client.from_credentials.side_effect = ValueError("no gemini key")
            outcome = _execute(run)

        assert outcome["success"] is True
        assert _row_states(db, run, GEMINI) == {
            (JobStatus.FAILED.value, "Batch submission failed: no gemini key")
        }
        if SARVAM in models:
            assert _row_states(db, run, SARVAM) == {(JobStatus.PENDING.value, None)}
            assert [c["model"] for c in boundaries.enqueued] == [SARVAM]
        # Left for the cron to finalize.
        db.refresh(run)
        assert run.status == "processing"

    def test_enqueue_failure_fails_only_that_models_rows(
        self, db: Session, user_api_key: TestAuthContext, boundaries: SimpleNamespace
    ) -> None:
        boundaries.enqueue_errors[SARVAM] = ConnectionError("broker unreachable")
        run = _run(db, user_api_key, [SARVAM, ELEVEN])

        outcome = _execute(run)

        assert outcome["success"] is True
        assert _row_states(db, run, SARVAM) == {
            (JobStatus.FAILED.value, "Failed to queue synthesis: broker unreachable")
        }
        assert _row_states(db, run, ELEVEN) == {(JobStatus.PENDING.value, None)}
        assert [c["model"] for c in boundaries.enqueued] == [ELEVEN]
        db.refresh(run)
        assert run.status == "processing"

    @pytest.mark.parametrize(
        ("case", "expected_error"),
        [
            ("missing_dataset", "Dataset not found"),
            ("empty_csv", "No samples found for dataset"),
            ("no_object_store_url", "No samples found for dataset"),
        ],
    )
    def test_unstartable_run_is_failed_without_rows(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        boundaries: SimpleNamespace,
        case: str,
        expected_error: str,
    ) -> None:
        if case == "empty_csv":
            boundaries.csv = _csv(0)
        run = _run(
            db,
            user_api_key,
            [SARVAM],
            object_store_url=None if case == "no_object_store_url" else "s3://b/x.csv",
        )
        dataset_id = (
            get_non_existent_id(db, EvaluationDataset)
            if case == "missing_dataset"
            else None
        )

        outcome = _execute(run, dataset_id=dataset_id)

        assert outcome["success"] is False
        db.refresh(run)
        assert (run.status, run.error_message) == ("failed", expected_error)
        assert _results(db, run, SARVAM) == []
        assert boundaries.enqueued == []

    @pytest.mark.parametrize(
        ("error", "expected_message"),
        [
            (Timeout(), "Task exceeded soft time limit"),
            (SoftTimeLimitExceeded(), "Task exceeded soft time limit"),
            (RuntimeError("no bucket"), "no bucket"),
        ],
        ids=["gevent_timeout", "soft_timeout", "unexpected"],
    )
    def test_worker_error_fails_run(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        error: BaseException,
        expected_message: str,
    ) -> None:
        run = _run(db, user_api_key, [SARVAM])

        with patch(f"{MODULE}.get_cloud_storage", side_effect=error):
            if isinstance(error, RuntimeError):
                assert _execute(run)["success"] is False
            else:
                # Timeouts re-raise so Celery records the task as timed out.
                with pytest.raises(type(error)):
                    _execute(run)

        db.refresh(run)
        assert (run.status, run.error_message) == ("failed", expected_message)
