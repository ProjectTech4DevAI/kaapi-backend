import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session, select

from app.crud.language import get_language_by_locale
from app.models import EvaluationRun
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.batch_job import execute_batch_submission
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    TEST_DATASET_URL,
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
    fake_gemini_batch,
    use_test_session,
)
from app.tests.utils.utils import get_non_existent_id

MODULE = "app.services.tts_evaluations.batch_job"
GEMINI = "gemini-2.5-pro-preview-tts"
SARVAM = "bulbul:v3"
ELEVEN = "eleven_v3"


def _csv(texts: list[str]) -> bytes:
    return ("text\n" + "\n".join(texts) + "\n").encode()


@dataclass
class Harness:
    csv: bytes = field(default_factory=lambda: _csv(["one", "two"]))
    enqueued: list[dict[str, Any]] = field(default_factory=list)
    enqueue_failures: set[str] = field(default_factory=set)
    on_stream: Any = None

    def stream(self, url: str) -> io.BytesIO:
        assert url == TEST_DATASET_URL
        if self.on_stream is not None:
            self.on_stream()
        return io.BytesIO(self.csv)

    def enqueue(self, **kwargs: Any) -> str:
        if kwargs["model"] in self.enqueue_failures:
            raise ConnectionError("broker unreachable")
        self.enqueued.append(kwargs)
        return f"task-{len(self.enqueued)}"


@pytest.fixture
def harness(db: Session) -> Iterator[Harness]:
    h = Harness()
    storage = MagicMock()
    storage.stream.side_effect = h.stream
    with (
        use_test_session(MODULE, db),
        patch(f"{MODULE}.get_cloud_storage", return_value=storage),
        patch(f"{MODULE}.start_tts_sync_chunk", side_effect=h.enqueue),
    ):
        yield h


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


def _execute(
    run: EvaluationRun, models: list[str], dataset_id: int | None = None
) -> dict[str, Any]:
    return execute_batch_submission(
        project_id=run.project_id,
        job_id=str(run.id),
        task_id="celery-task",
        task_instance=MagicMock(),
        organization_id=run.organization_id,
        dataset_id=dataset_id if dataset_id is not None else run.dataset_id,
        models=models,
    )


def _results(
    db: Session, run: EvaluationRun, model: str | None = None
) -> list[TTSResult]:
    stmt = select(TTSResult).where(TTSResult.evaluation_run_id == run.id)
    if model:
        stmt = stmt.where(TTSResult.provider == model)
    return list(db.exec(stmt.order_by(TTSResult.id)).all())


@pytest.mark.usefixtures("harness")
class TestExecuteBatchSubmission:
    def test_batch_only_submits_gemini_and_links_run(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        run = _run(db, user_api_key, [GEMINI])

        with fake_gemini_batch("app.crud.tts_evaluations.batch") as gemini:
            outcome = _execute(run, [GEMINI])

        assert outcome["success"] is True
        assert outcome["result_count"] == 2
        assert outcome["sync_chunks"] == {}
        assert list(outcome["batch_jobs"]) == [GEMINI]
        assert len(gemini.submitted[f"models/{GEMINI}"]) == 2
        assert harness.enqueued == []
        db.refresh(run)
        assert run.status == "processing"
        assert run.batch_job_id == outcome["batch_jobs"][GEMINI]["batch_job_id"]

    def test_sync_only_chunks_by_25_with_default_language(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        harness.csv = _csv([f"sample {i}" for i in range(30)])
        run = _run(db, user_api_key, [SARVAM])

        with fake_gemini_batch("app.crud.tts_evaluations.batch") as gemini:
            outcome = _execute(run, [SARVAM])

        assert outcome["success"] is True
        assert outcome["batch_jobs"] == {}
        assert outcome["sync_chunks"] == {SARVAM: ["task-1", "task-2"]}
        assert gemini.submitted == {}
        result_ids = [r.id for r in _results(db, run)]
        assert [c["result_ids"] for c in harness.enqueued] == [
            result_ids[:25],
            result_ids[25:],
        ]
        for chunk in harness.enqueued:
            assert chunk["model"] == SARVAM
            assert chunk["language_code"] == "en-IN"
            assert chunk["job_id"] == str(run.id)
            assert chunk["organization_id"] == run.organization_id
            assert chunk["trace_id"] == "N/A"
        db.refresh(run)
        assert run.status == "processing"
        assert run.batch_job_id is None

    def test_dataset_language_is_passed_to_chunks(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        hindi = get_language_by_locale(session=db, locale="hi")
        run = _run(db, user_api_key, [ELEVEN], language_id=hindi.id)

        _execute(run, [ELEVEN])

        assert [c["language_code"] for c in harness.enqueued] == ["hi-IN"]

    def test_mixed_models_route_rows_by_execution_mode(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        models = [GEMINI, SARVAM, ELEVEN]
        run = _run(db, user_api_key, models)

        with fake_gemini_batch("app.crud.tts_evaluations.batch") as gemini:
            outcome = _execute(run, models)

        assert outcome["result_count"] == 6
        assert list(outcome["batch_jobs"]) == [GEMINI]
        assert set(outcome["sync_chunks"]) == {SARVAM, ELEVEN}
        gemini_keys = {r["key"] for r in gemini.submitted[f"models/{GEMINI}"]}
        assert gemini_keys == {str(r.id) for r in _results(db, run, GEMINI)}
        enqueued_by_model = {c["model"]: c["result_ids"] for c in harness.enqueued}
        assert enqueued_by_model == {
            SARVAM: [r.id for r in _results(db, run, SARVAM)],
            ELEVEN: [r.id for r in _results(db, run, ELEVEN)],
        }
        db.refresh(run)
        assert run.status == "processing"
        assert run.batch_job_id is not None

    def test_batch_failure_with_sync_models_continues(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        run = _run(db, user_api_key, [GEMINI, SARVAM])

        with fake_gemini_batch(
            "app.crud.tts_evaluations.batch",
            {f"models/{GEMINI}": RuntimeError("quota")},
        ):
            outcome = _execute(run, [GEMINI, SARVAM])

        assert outcome["success"] is True
        assert outcome["batch_jobs"] == {}
        assert outcome["sync_chunks"] == {SARVAM: ["task-1"]}
        assert {r.status for r in _results(db, run, GEMINI)} == {JobStatus.FAILED.value}
        assert {r.status for r in _results(db, run, SARVAM)} == {
            JobStatus.PENDING.value
        }
        db.refresh(run)
        assert run.status == "processing"

    def test_batch_failure_without_sync_models_fails_run(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        run = _run(db, user_api_key, [GEMINI])

        with fake_gemini_batch(
            "app.crud.tts_evaluations.batch",
            {f"models/{GEMINI}": RuntimeError("quota")},
        ):
            outcome = _execute(run, [GEMINI])

        assert outcome == {
            "success": False,
            "error": "Batch submission failed for all models",
        }
        db.refresh(run)
        assert run.status == "failed"
        assert run.error_message == "Batch submission failed for all models"

    def test_enqueue_failure_fails_that_models_results(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        harness.enqueue_failures = {ELEVEN}
        run = _run(db, user_api_key, [SARVAM, ELEVEN])

        outcome = _execute(run, [SARVAM, ELEVEN])

        assert outcome["sync_chunks"] == {SARVAM: ["task-1"], ELEVEN: []}
        eleven_rows = _results(db, run, ELEVEN)
        assert {r.status for r in eleven_rows} == {JobStatus.FAILED.value}
        assert eleven_rows[0].error_message.startswith(
            f"[KAAPI] Failed to queue synthesis for {ELEVEN}: broker unreachable."
        )
        assert {r.status for r in _results(db, run, SARVAM)} == {
            JobStatus.PENDING.value
        }
        db.refresh(run)
        assert run.status == "processing"
        assert run.error_message == "2 synthesis(es) failed"

    def test_pending_rows_of_unrequested_models_are_not_enqueued(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        run = _run(db, user_api_key, [SARVAM])
        stray = create_test_tts_result_row(db, run=run, provider="eleven_v4")

        outcome = _execute(run, [SARVAM])

        assert list(outcome["sync_chunks"]) == [SARVAM]
        assert all(stray.id not in c["result_ids"] for c in harness.enqueued)

    def test_run_not_found(self, db: Session, user_api_key: TestAuthContext) -> None:
        run = _run(db, user_api_key, [SARVAM])
        missing = EvaluationRun(
            id=get_non_existent_id(db, EvaluationRun),
            project_id=run.project_id,
            organization_id=run.organization_id,
            dataset_id=run.dataset_id,
        )

        assert _execute(missing, [SARVAM]) == {
            "success": False,
            "error": "Run not found",
        }

    def test_run_deleted_mid_submission(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        run = _run(db, user_api_key, [SARVAM])
        run_id = run.id

        def delete_run() -> None:
            db.delete(db.get(EvaluationRun, run_id))
            db.commit()

        harness.on_stream = delete_run

        assert _execute(run, [SARVAM]) == {"success": False, "error": "Run not found"}
        assert harness.enqueued == []

    def test_dataset_not_found_fails_run(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _run(db, user_api_key, [SARVAM])

        outcome = _execute(run, [SARVAM], dataset_id=run.dataset_id + 10_000_000)

        assert outcome == {"success": False, "error": "Dataset not found"}
        db.refresh(run)
        assert (run.status, run.error_message) == ("failed", "Dataset not found")

    def test_dataset_without_csv_fails_run(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        run = _run(db, user_api_key, [SARVAM], object_store_url=None)

        outcome = _execute(run, [SARVAM])

        assert outcome == {"success": False, "error": "No samples found"}
        db.refresh(run)
        assert (run.status, run.error_message) == (
            "failed",
            "No samples found for dataset",
        )
        assert _results(db, run) == []

    def test_empty_csv_fails_run(
        self, db: Session, user_api_key: TestAuthContext, harness: Harness
    ) -> None:
        harness.csv = b"text\n"
        run = _run(db, user_api_key, [SARVAM])

        assert _execute(run, [SARVAM]) == {
            "success": False,
            "error": "No samples found",
        }

    @pytest.mark.parametrize(
        "timeout", [Timeout(), SoftTimeLimitExceeded()], ids=["gevent", "soft"]
    )
    def test_timeout_fails_run_and_reraises(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        harness: Harness,
        timeout: BaseException,
    ) -> None:
        run = _run(db, user_api_key, [SARVAM])
        with patch(f"{MODULE}.load_sample_texts", side_effect=timeout):
            with pytest.raises(type(timeout)):
                _execute(run, [SARVAM])

        db.refresh(run)
        assert (run.status, run.error_message) == (
            "failed",
            "Task exceeded soft time limit",
        )

    def test_unexpected_error_fails_run(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _run(db, user_api_key, [SARVAM])

        with patch(
            f"{MODULE}.get_cloud_storage", side_effect=RuntimeError("no bucket")
        ):
            outcome = _execute(run, [SARVAM])

        assert outcome == {"success": False, "error": "no bucket"}
        db.refresh(run)
        assert (run.status, run.error_message) == ("failed", "no bucket")
