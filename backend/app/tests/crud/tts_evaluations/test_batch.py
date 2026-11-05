from sqlmodel import Session, select

from app.crud.tts_evaluations.batch import start_tts_evaluation_batch
from app.models.batch_job import BatchJob
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
    fake_gemini_batch,
)

MODULE = "app.crud.tts_evaluations.batch"
GEMINI = "gemini-3.8-flash-tts"
GEMINI_LITE = "gemini-3.8-flash-lite-tts"


def _voice_names(jsonl: list[dict]) -> set[str]:
    return {
        r["request"]["generationConfig"]["speechConfig"]["voiceConfig"][
            "prebuiltVoiceConfig"
        ]["voiceName"]
        for r in jsonl
    }


def test_each_model_batch_uses_its_rows_voice(
    db: Session, user_api_key: TestAuthContext
) -> None:
    run = create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[GEMINI, GEMINI_LITE],
    )
    results = [
        create_test_tts_result_row(db, run=run, provider=GEMINI, voice="Puck"),
        create_test_tts_result_row(db, run=run, provider=GEMINI, voice="Puck"),
        # NULL: row created before per-model voices; it was synthesized with Kore.
        create_test_tts_result_row(db, run=run, provider=GEMINI_LITE),
    ]

    with fake_gemini_batch(MODULE) as gemini:
        outcome = start_tts_evaluation_batch(
            session=db,
            run=run,
            results=results,
            org_id=run.organization_id,
            project_id=run.project_id,
        )

    assert _voice_names(gemini.submitted[f"models/{GEMINI}"]) == {"Puck"}
    assert _voice_names(gemini.submitted[f"models/{GEMINI_LITE}"]) == {"Kore"}
    job_voices = {
        job.config["tts_provider"]: job.config["voice_name"]
        for job in db.exec(
            select(BatchJob).where(
                BatchJob.id.in_(
                    [j["batch_job_id"] for j in outcome["batch_jobs"].values()]
                )
            )
        ).all()
    }
    assert job_voices == {GEMINI: "Puck", GEMINI_LITE: "Kore"}
