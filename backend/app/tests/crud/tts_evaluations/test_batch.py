import pytest
from sqlmodel import Session, select

from app.crud.tts_evaluations.batch import (
    TTSBatchSubmissionError,
    start_tts_evaluation_batch,
)
from app.models import EvaluationRun
from app.models.batch_job import BatchJob, BatchJobType
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.constants import (
    TTS_MODEL_REGISTRY,
    TTSExecutionModeEnum,
    TTSModelSpec,
)
from app.core.providers import Provider
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
    fake_gemini_batch,
)

MODULE = "app.crud.tts_evaluations.batch"
GEMINI = "gemini-2.5-pro-preview-tts"
# A second batch-mode model so partial (per-model) failure is observable.
SECOND_BATCH_MODEL = "gemini-test-batch-tts"


@pytest.fixture
def run(db: Session, user_api_key: TestAuthContext) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[GEMINI, "bulbul:v3"],
    )


@pytest.fixture
def second_batch_model(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(
        TTS_MODEL_REGISTRY,
        SECOND_BATCH_MODEL,
        TTSModelSpec(
            provider=Provider.GOOGLE_AISTUDIO,
            execution_mode=TTSExecutionModeEnum.BATCH,
            default_voice="Puck",
        ),
    )
    return SECOND_BATCH_MODEL


def _submit(
    db: Session, run: EvaluationRun, results: list[TTSResult], models: list[str]
) -> dict:
    return start_tts_evaluation_batch(
        session=db,
        run=run,
        results=results,
        models=models,
        org_id=run.organization_id,
        project_id=run.project_id,
    )


def _batch_jobs_for(db: Session, run_id: int) -> list[BatchJob]:
    jobs = db.exec(
        select(BatchJob).where(BatchJob.job_type == BatchJobType.TTS_EVALUATION.value)
    ).all()
    return [j for j in jobs if j.config.get("evaluation_run_id") == run_id]


class TestStartTTSEvaluationBatch:
    def test_rejects_sync_model(self, db: Session, run: EvaluationRun) -> None:
        with fake_gemini_batch(MODULE) as fake:
            with pytest.raises(ValueError, match="'bulbul:v3' is not a batch-mode"):
                _submit(db, run, [], [GEMINI, "bulbul:v3"])

        assert fake.submitted == {}

    def test_submits_batch_rows_and_links_run(
        self, db: Session, run: EvaluationRun
    ) -> None:
        gemini_rows = [create_test_tts_result_row(db, run=run) for _ in range(2)]
        sync_row = create_test_tts_result_row(db, run=run, provider="bulbul:v3")

        with fake_gemini_batch(MODULE) as fake:
            outcome = _submit(db, run, [*gemini_rows, sync_row], [GEMINI])

        [job] = _batch_jobs_for(db, run.id)
        assert outcome == {
            "success": True,
            "run_id": run.id,
            "batch_jobs": {
                GEMINI: {
                    "batch_job_id": job.id,
                    "provider_batch_id": f"batches/{GEMINI}",
                }
            },
            "result_count": 2,
        }
        assert [r["key"] for r in fake.submitted[f"models/{GEMINI}"]] == [
            str(r.id) for r in gemini_rows
        ]
        assert job.config["tts_provider"] == GEMINI
        assert job.config["voice_name"] == "Kore"
        assert job.provider == "google-aistudio"
        db.refresh(run)
        assert run.status == "processing"
        assert run.batch_job_id == job.id

    def test_all_models_failing_raises_and_fails_rows(
        self, db: Session, run: EvaluationRun
    ) -> None:
        row = create_test_tts_result_row(db, run=run)
        sync_row = create_test_tts_result_row(db, run=run, provider="bulbul:v3")

        with fake_gemini_batch(MODULE, {f"models/{GEMINI}": RuntimeError("quota")}):
            with pytest.raises(TTSBatchSubmissionError):
                _submit(db, run, [row, sync_row], [GEMINI])

        db.refresh(row)
        db.refresh(sync_row)
        db.refresh(run)
        assert row.status == JobStatus.FAILED.value
        assert row.error_message == f"Batch submission failed for {GEMINI}: quota"
        assert sync_row.status == JobStatus.PENDING.value
        assert run.batch_job_id is None
        [job] = _batch_jobs_for(db, run.id)
        assert job.provider_status == "failed"

    def test_model_without_rows_is_skipped(
        self, db: Session, run: EvaluationRun
    ) -> None:
        with fake_gemini_batch(MODULE) as fake:
            with pytest.raises(TTSBatchSubmissionError):
                _submit(db, run, [], [GEMINI])

        assert fake.submitted == {}
        assert _batch_jobs_for(db, run.id) == []

    def test_one_model_failing_only_fails_its_rows(
        self, db: Session, run: EvaluationRun, second_batch_model: str
    ) -> None:
        failing_row = create_test_tts_result_row(db, run=run)
        ok_row = create_test_tts_result_row(db, run=run, provider=second_batch_model)

        with fake_gemini_batch(MODULE, {f"models/{GEMINI}": RuntimeError("bad key")}):
            outcome = _submit(
                db, run, [failing_row, ok_row], [GEMINI, second_batch_model]
            )

        assert list(outcome["batch_jobs"]) == [second_batch_model]
        assert outcome["result_count"] == 1
        db.refresh(failing_row)
        db.refresh(ok_row)
        db.refresh(run)
        assert failing_row.status == JobStatus.FAILED.value
        assert ok_row.status == JobStatus.PENDING.value
        assert (
            run.batch_job_id
            == outcome["batch_jobs"][second_batch_model]["batch_job_id"]
        )

    def test_run_links_first_submitted_job(
        self, db: Session, run: EvaluationRun, second_batch_model: str
    ) -> None:
        rows = [
            create_test_tts_result_row(db, run=run),
            create_test_tts_result_row(db, run=run, provider=second_batch_model),
        ]

        with fake_gemini_batch(MODULE):
            outcome = _submit(db, run, rows, [GEMINI, second_batch_model])

        assert len(_batch_jobs_for(db, run.id)) == 2
        db.refresh(run)
        assert run.batch_job_id == outcome["batch_jobs"][GEMINI]["batch_job_id"]
