from sqlmodel import Session

from app.crud.tts_evaluations.result import fail_pending_tts_results
from app.models.job import JobStatus
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
)

GEMINI = "gemini-2.5-pro-preview-tts"
SARVAM = "bulbul:v3"


def test_fail_pending_tts_results_only_touches_pending_rows_of_run_and_provider(
    db: Session, user_api_key: TestAuthContext
) -> None:
    run, other_run = (
        create_test_tts_run_with_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=[GEMINI, SARVAM],
        )
        for _ in range(2)
    )
    pending = [
        create_test_tts_result_row(db, run=run, provider=GEMINI) for _ in range(2)
    ]
    # A worker finishing a row just before a stale/cleanup pass must keep its outcome.
    succeeded = create_test_tts_result_row(
        db, run=run, provider=GEMINI, status=JobStatus.SUCCESS
    )
    already_failed = create_test_tts_result_row(
        db, run=run, provider=GEMINI, status=JobStatus.FAILED
    )
    already_failed.error_message = "original error"
    db.add(already_failed)
    db.commit()
    other_provider = create_test_tts_result_row(db, run=run, provider=SARVAM)
    other_run_row = create_test_tts_result_row(db, run=other_run, provider=GEMINI)

    failed_count = fail_pending_tts_results(
        session=db, run_id=run.id, provider=GEMINI, error_message="boom"
    )

    assert failed_count == 2
    for row in (*pending, succeeded, already_failed, other_provider, other_run_row):
        db.refresh(row)
    for row in pending:
        assert (row.status, row.error_message) == (JobStatus.FAILED.value, "boom")
    assert (succeeded.status, succeeded.error_message) == (
        JobStatus.SUCCESS.value,
        None,
    )
    assert already_failed.error_message == "original error"
    assert other_provider.status == JobStatus.PENDING.value
    assert other_run_row.status == JobStatus.PENDING.value
