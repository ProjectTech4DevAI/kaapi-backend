from collections.abc import Iterator
from typing import Any

import pytest
from sqlmodel import Session, delete, select

from app.core.batch.operations import start_batch_job
from app.core.db import engine
from app.models.batch_job import BatchJob, BatchJobType
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.utils import random_lower_string


class _RecordingProvider:
    """Fake batch provider that snapshots DB state while the 'network' call runs."""

    def __init__(
        self, session: Session, marker: str, error: Exception | None = None
    ) -> None:
        self._session = session
        self._marker = marker
        self._error = error
        self.in_transaction_during_call: bool | None = None
        self.job_visible_to_other_connections: bool | None = None

    def create_batch(
        self, jsonl_data: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.in_transaction_during_call = self._session.in_transaction()
        with Session(engine) as other:
            jobs = other.exec(
                select(BatchJob).where(BatchJob.provider == self._marker)
            ).all()
            self.job_visible_to_other_connections = len(jobs) == 1
        if self._error is not None:
            raise self._error
        return {
            "provider_batch_id": "batches/abc",
            "provider_file_id": "files/abc",
            "provider_status": "JOB_STATE_PENDING",
        }


@pytest.fixture
def committed_session(user_api_key: TestAuthContext) -> Iterator[tuple[Session, str]]:
    # The conftest `db` session always sits inside a savepoint, so it can never
    # report "no open transaction"; this needs a real session, cleaned up by hand.
    marker = f"test-provider-{random_lower_string()}"
    session = Session(engine)
    try:
        yield session, marker
    finally:
        session.rollback()
        session.exec(delete(BatchJob).where(BatchJob.provider == marker))
        session.commit()
        session.close()


def _start(
    session: Session, provider: _RecordingProvider, marker: str, auth: TestAuthContext
) -> BatchJob:
    return start_batch_job(
        session=session,
        provider=provider,
        provider_name=marker,
        job_type=BatchJobType.TTS_EVALUATION.value,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        jsonl_data=[{"key": "1"}, {"key": "2"}],
        config={"model": "m"},
    )


class TestStartBatchJob:
    def test_no_transaction_is_held_during_provider_call(
        self, committed_session: tuple[Session, str], user_api_key: TestAuthContext
    ) -> None:
        session, marker = committed_session
        provider = _RecordingProvider(session, marker)

        job = _start(session, provider, marker, user_api_key)

        assert provider.in_transaction_during_call is False
        assert provider.job_visible_to_other_connections is True
        assert job.provider_batch_id == "batches/abc"
        assert job.provider_status == "JOB_STATE_PENDING"
        assert job.total_items == 2

    def test_provider_failure_marks_job_failed_and_reraises(
        self, committed_session: tuple[Session, str], user_api_key: TestAuthContext
    ) -> None:
        session, marker = committed_session
        provider = _RecordingProvider(
            session, marker, error=RuntimeError("upload refused")
        )

        with pytest.raises(RuntimeError, match="upload refused"):
            _start(session, provider, marker, user_api_key)

        [job] = session.exec(select(BatchJob).where(BatchJob.provider == marker)).all()
        assert job.provider_status == "failed"
        assert job.error_message == "Batch creation failed: upload refused"
