from datetime import datetime, timedelta

import pytest
from sqlalchemy import text
from sqlmodel import Session

from app.models import Job
from app.crud.llm import REDACTED_SENTINEL, redact_llm_calls
from app.models.llm import LlmCall
from app.tests.utils.llm import create_aged_llm_call, create_llm_job

CUTOFF = datetime(2026, 1, 8, 0, 0, 0)
AGED = CUTOFF - timedelta(days=1)
RECENT = CUTOFF + timedelta(minutes=1)

TEXT_CONTENT = {"type": "text", "content": {"format": "text", "value": "hello"}}


@pytest.fixture
def job(db: Session) -> Job:
    # The statement targets llm_call_2, a copy table that only exists in the
    # deployed databases; an auto-updatable view over llm_call gives the tests
    # that name while keeping the real column types and the ORM factories.
    db.exec(text("DROP VIEW IF EXISTS llm_call_2"))
    # Other llm_call rows (seed data, sibling fixtures) would also match the
    # cutoff and skew the rowcount assertions, so start from an empty table.
    db.exec(text("DELETE FROM llm_call"))
    db.exec(text("CREATE VIEW llm_call_2 AS SELECT * FROM llm_call"))
    db.commit()
    yield create_llm_job(db)
    db.exec(text("DROP VIEW IF EXISTS llm_call_2"))
    db.commit()


def _reload(db: Session, llm_call: LlmCall) -> LlmCall:
    db.expire_all()
    return db.get(LlmCall, llm_call.id)


def test_redacts_input_and_content_value_only(db: Session, job: Job) -> None:
    llm_call = create_aged_llm_call(
        db,
        updated_at=AGED,
        job_id=job.id,
        input="my social security number is 123-45-6789",
        content=TEXT_CONTENT,
    )

    redacted = redact_llm_calls(session=db, cutoff=CUTOFF)

    assert redacted == 1
    row = _reload(db, llm_call)
    assert row.input == REDACTED_SENTINEL
    assert row.content["content"]["value"] == REDACTED_SENTINEL
    assert row.content["content"]["format"] == "text"
    assert row.content["type"] == "text"


def test_leaves_rows_newer_than_cutoff_untouched(db: Session, job: Job) -> None:
    llm_call = create_aged_llm_call(
        db,
        updated_at=RECENT,
        job_id=job.id,
        input="still within the retention window",
        content=TEXT_CONTENT,
    )

    redacted = redact_llm_calls(session=db, cutoff=CUTOFF)

    assert redacted == 0
    row = _reload(db, llm_call)
    assert row.input == "still within the retention window"
    assert row.content["content"]["value"] == "hello"


def test_row_exactly_at_cutoff_is_redacted(db: Session, job: Job) -> None:
    llm_call = create_aged_llm_call(
        db,
        updated_at=CUTOFF,
        job_id=job.id,
        content=TEXT_CONTENT,
    )

    assert redact_llm_calls(session=db, cutoff=CUTOFF) == 1
    assert _reload(db, llm_call).input == REDACTED_SENTINEL


def test_second_run_skips_already_redacted_rows(db: Session, job: Job) -> None:
    create_aged_llm_call(db, updated_at=AGED, job_id=job.id, content=TEXT_CONTENT)

    assert redact_llm_calls(session=db, cutoff=CUTOFF) == 1
    assert redact_llm_calls(session=db, cutoff=CUTOFF) == 0


def test_row_with_null_content_is_redacted_without_error(db: Session, job: Job) -> None:
    llm_call = create_aged_llm_call(
        db,
        updated_at=AGED,
        job_id=job.id,
        input="no response was ever recorded",
        content=None,
    )
    # A call that never got a response stores the JSONB scalar 'null', not SQL
    # NULL; jsonb_set errors on scalars, so this must take a guarded path.
    assert (
        db.exec(
            text(
                "SELECT jsonb_typeof(content) FROM llm_call WHERE id = :id"
            ).bindparams(id=llm_call.id)
        ).scalar_one()
        == "null"
    )

    assert redact_llm_calls(session=db, cutoff=CUTOFF) == 1
    row = _reload(db, llm_call)
    assert row.input == REDACTED_SENTINEL
    assert row.content is None


def test_returns_zero_when_no_rows_match(db: Session, job: Job) -> None:
    assert redact_llm_calls(session=db, cutoff=CUTOFF) == 0


def test_redacts_only_aged_rows_in_a_mixed_table(db: Session, job: Job) -> None:
    aged = create_aged_llm_call(
        db, updated_at=AGED, job_id=job.id, input="aged", content=TEXT_CONTENT
    )
    recent = create_aged_llm_call(
        db, updated_at=RECENT, job_id=job.id, input="recent", content=TEXT_CONTENT
    )

    assert redact_llm_calls(session=db, cutoff=CUTOFF) == 1
    assert _reload(db, aged).input == REDACTED_SENTINEL
    assert _reload(db, recent).input == "recent"


def test_redacts_content_value_of_row_whose_input_is_already_sentinel(
    db: Session, job: Job
) -> None:
    llm_call = create_aged_llm_call(
        db,
        updated_at=AGED,
        job_id=job.id,
        input=REDACTED_SENTINEL,
        content={"type": "audio", "content": {"format": "uri", "value": "s3://a.wav"}},
    )

    assert redact_llm_calls(session=db, cutoff=CUTOFF) == 1
    assert _reload(db, llm_call).content["content"]["value"] == REDACTED_SENTINEL
