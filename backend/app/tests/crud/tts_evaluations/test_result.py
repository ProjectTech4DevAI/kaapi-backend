from sqlmodel import Session

from app.crud.tts_evaluations.result import (
    bulk_update_pending_tts_results,
    list_pending_tts_results_by_ids,
)
from app.models import EvaluationRun
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResultUpdate
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import (
    create_test_tts_result_row,
    create_test_tts_run_with_dataset,
)


def _make_run(db: Session, auth: TestAuthContext) -> EvaluationRun:
    return create_test_tts_run_with_dataset(
        db, organization_id=auth.organization_id, project_id=auth.project_id
    )


class TestListPendingTTSResultsByIds:
    def test_returns_only_pending_rows_of_run_in_id_order(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        other_run = _make_run(db, user_api_key)
        pending_a = create_test_tts_result_row(db, run=run)
        done = create_test_tts_result_row(db, run=run, status=JobStatus.SUCCESS)
        pending_b = create_test_tts_result_row(db, run=run)
        not_requested = create_test_tts_result_row(db, run=run)
        foreign = create_test_tts_result_row(db, run=other_run)

        rows = list_pending_tts_results_by_ids(
            session=db,
            run_id=run.id,
            result_ids=[pending_b.id, done.id, pending_a.id, foreign.id],
        )

        assert [r.id for r in rows] == [pending_a.id, pending_b.id]
        assert not_requested.id not in {r.id for r in rows}

    def test_empty_ids_returns_empty_list(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        create_test_tts_result_row(db, run=run)

        assert (
            list_pending_tts_results_by_ids(session=db, run_id=run.id, result_ids=[])
            == []
        )


class TestBulkUpdatePendingTTSResults:
    def test_applies_success_and_failure_fields(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        ok = create_test_tts_result_row(db, run=run)
        bad = create_test_tts_result_row(db, run=run)

        updated = bulk_update_pending_tts_results(
            session=db,
            updates=[
                TTSResultUpdate(
                    result_id=ok.id,
                    status=JobStatus.SUCCESS,
                    object_store_url="s3://bucket/a.wav",
                    metadata={"duration_seconds": 1.5, "size_bytes": 72044},
                ),
                TTSResultUpdate(
                    result_id=bad.id,
                    status=JobStatus.FAILED,
                    error_message="[SARVAM] boom",
                ),
            ],
        )

        assert updated == 2
        db.refresh(ok)
        db.refresh(bad)
        assert ok.status == JobStatus.SUCCESS.value
        assert ok.object_store_url == "s3://bucket/a.wav"
        assert ok.metadata_ == {"duration_seconds": 1.5, "size_bytes": 72044}
        assert ok.error_message is None
        assert bad.status == JobStatus.FAILED.value
        assert bad.error_message == "[SARVAM] boom"
        assert bad.object_store_url is None

    def test_rows_no_longer_pending_are_not_clobbered(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key)
        finished = create_test_tts_result_row(
            db, run=run, status=JobStatus.SUCCESS, object_store_url="s3://b/keep.wav"
        )
        pending = create_test_tts_result_row(db, run=run)

        updated = bulk_update_pending_tts_results(
            session=db,
            updates=[
                TTSResultUpdate(
                    result_id=finished.id, status=JobStatus.FAILED, error_message="late"
                ),
                TTSResultUpdate(
                    result_id=pending.id, status=JobStatus.FAILED, error_message="late"
                ),
            ],
        )

        assert updated == 1
        db.refresh(finished)
        assert finished.status == JobStatus.SUCCESS.value
        assert finished.object_store_url == "s3://b/keep.wav"
        assert finished.error_message is None

    def test_empty_updates_returns_zero(self, db: Session) -> None:
        assert bulk_update_pending_tts_results(session=db, updates=[]) == 0
