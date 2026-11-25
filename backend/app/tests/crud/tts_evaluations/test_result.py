import pytest
from sqlmodel import Session

from app.crud.tts_evaluations.result import (
    create_tts_results,
    fail_pending_tts_results,
    get_tts_usage_by_model,
    update_tts_result,
)
from app.models.job import JobStatus
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
)

GEMINI = "gemini-3.8-flash-tts"
SARVAM = "bulbul:v3"
ELEVEN = "eleven_v3"


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


@pytest.mark.parametrize(
    ("voices", "expected"),
    [
        ({GEMINI: "Puck", SARVAM: "priya"}, {GEMINI: "Puck", SARVAM: "priya"}),
        ({GEMINI: "Puck"}, {GEMINI: "Puck", SARVAM: None}),
        (None, {GEMINI: None, SARVAM: None}),
    ],
    ids=["all_models", "model_missing_from_voices", "voices_omitted"],
)
def test_create_tts_results_stamps_voice_per_model(
    db: Session,
    user_api_key: TestAuthContext,
    voices: dict[str, str] | None,
    expected: dict[str, str | None],
) -> None:
    run = create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[GEMINI, SARVAM],
    )

    results = create_tts_results(
        session=db,
        sample_texts=["one", "two"],
        evaluation_run_id=run.id,
        org_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[GEMINI, SARVAM],
        voices=voices,
    )

    assert len(results) == 4
    for row in results:
        db.refresh(row)
        assert row.voice == expected[row.provider]


def test_get_tts_usage_by_model_sums_each_model_in_its_own_unit(
    db: Session, user_api_key: TestAuthContext
) -> None:
    run, other_run = (
        create_test_tts_run_with_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            models=[GEMINI, SARVAM, ELEVEN],
        )
        for _ in range(2)
    )
    usages = {
        GEMINI: [
            {
                "unit": "tokens",
                "input_tokens": 10,
                "output_tokens": 300,
                "total_tokens": 310,
            },
            {
                "unit": "tokens",
                "input_tokens": 0,
                "output_tokens": 200,
                "total_tokens": 200,
            },
        ],
        SARVAM: [
            {"unit": "characters", "characters": 12},
            {"unit": "characters", "characters": 30},
        ],
        ELEVEN: [
            {
                "unit": "credits",
                "characters": 10,
                "credits": 5.0,
                "credit_multiplier": 0.5,
            },
            {
                "unit": "credits",
                "characters": 20,
                "credits": 10.5,
                "credit_multiplier": 0.5,
            },
            # Multiplier fetch failed for this row: its characters still count.
            {
                "unit": "credits",
                "characters": 4,
                "credits": None,
                "credit_multiplier": None,
            },
        ],
    }
    for model, model_usages in usages.items():
        for usage in model_usages:
            create_test_tts_result_row(
                db, run=run, provider=model, status=JobStatus.SUCCESS, usage=usage
            )
        create_test_tts_result_row(
            db, run=run, provider=model, status=JobStatus.FAILED, usage=None
        )
        create_test_tts_result_row(
            db, run=other_run, provider=model, usage=model_usages[0]
        )

    usage_by_model = get_tts_usage_by_model(
        session=db,
        run_id=run.id,
        org_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )

    assert usage_by_model == {
        GEMINI: {
            "unit": "tokens",
            "input_tokens": 10,
            "output_tokens": 500,
            "total_tokens": 510,
            "results_with_usage": 2,
        },
        SARVAM: {"unit": "characters", "characters": 42, "results_with_usage": 2},
        ELEVEN: {
            "unit": "credits",
            "characters": 34,
            "credits": 15.5,
            "results_with_usage": 3,
        },
    }
    # SUM over bigint comes back as Decimal, which serializes as a JSON string.
    assert type(usage_by_model[GEMINI]["total_tokens"]) is int
    assert type(usage_by_model[SARVAM]["characters"]) is int
    assert type(usage_by_model[ELEVEN]["credits"]) is float


def test_get_tts_usage_by_model_is_empty_without_usage(
    db: Session, user_api_key: TestAuthContext
) -> None:
    run = create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    create_test_tts_result_row(db, run=run)

    assert (
        get_tts_usage_by_model(
            session=db,
            run_id=run.id,
            org_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        == {}
    )


def test_update_tts_result_skips_rows_no_longer_pending(
    db: Session, user_api_key: TestAuthContext
) -> None:
    # Regression: a sync worker finishing after the cron failed the row and
    # finalized the run used to overwrite FAILED with SUCCESS.
    run = create_test_tts_run_with_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        models=[SARVAM],
    )
    stale = create_test_tts_result_row(
        db, run=run, provider=SARVAM, status=JobStatus.FAILED
    )
    pending = create_test_tts_result_row(db, run=run, provider=SARVAM)

    late = update_tts_result(
        session=db,
        result_id=stale.id,
        status=JobStatus.SUCCESS.value,
        object_store_url="s3://bucket/late.wav",
    )
    written = update_tts_result(
        session=db, result_id=pending.id, status=JobStatus.SUCCESS.value
    )

    db.refresh(stale)
    assert late is None
    assert (stale.status, stale.object_store_url) == (JobStatus.FAILED.value, None)
    assert written is not None and written.status == JobStatus.SUCCESS.value
