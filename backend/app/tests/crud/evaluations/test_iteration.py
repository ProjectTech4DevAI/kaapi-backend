"""CRUD tests for the thin `evaluation_iteration_run` tracking row.

The round-by-round trajectory itself lives in the LangGraph checkpoint, not
here — see app/tests/services/evaluations/test_iteration_graph.py.
"""

from sqlmodel import Session

from app.crud.evaluations.iteration import (
    create_evaluation_iteration_run,
    get_evaluation_iteration_run_by_id,
    list_processing_evaluation_iteration_runs,
    update_evaluation_iteration_run,
)
from app.models.evaluation_iteration import (
    EvaluationIterationRunUpdate,
    EvaluationIterationStatusEnum,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)

_CALLBACK_URL = "https://example.com/callback"


def _make_dataset_and_config(db: Session, user_api_key: TestAuthContext):
    dataset = create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    config = create_test_config(
        db=db, project_id=user_api_key.project_id, use_kaapi_schema=True
    )
    return dataset, config


def _make_run(db: Session, user_api_key: TestAuthContext, experiment_name: str):
    dataset, config = _make_dataset_and_config(db, user_api_key)
    return create_evaluation_iteration_run(
        session=db,
        dataset_id=dataset.id,
        experiment_name=experiment_name,
        config_id=config.id,
        initial_config_version=1,
        callback_url=_CALLBACK_URL,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


class TestCreateEvaluationIterationRun:
    def test_creates_row_with_processing_status(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key, "crud-create")

        assert run.id is not None
        assert run.status == EvaluationIterationStatusEnum.PROCESSING
        assert run.stop_reason is None
        assert run.error_message is None


class TestGetEvaluationIterationRunById:
    def test_returns_none_when_missing(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        found = get_evaluation_iteration_run_by_id(
            session=db,
            iteration_run_id=999_999,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        assert found is None

    def test_returns_none_for_a_different_project_scope(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        superuser_api_key: TestAuthContext,
    ) -> None:
        run = _make_run(db, user_api_key, "crud-scope")

        found = get_evaluation_iteration_run_by_id(
            session=db,
            iteration_run_id=run.id,
            organization_id=superuser_api_key.organization_id,
            project_id=superuser_api_key.project_id,
        )
        assert found is None

    def test_returns_row_for_matching_scope(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key, "crud-match")

        found = get_evaluation_iteration_run_by_id(
            session=db,
            iteration_run_id=run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        assert found is not None
        assert found.id == run.id


class TestListProcessingEvaluationIterationRuns:
    def test_only_processing_rows_are_returned(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        processing = _make_run(db, user_api_key, "crud-list-processing")
        completed = _make_run(db, user_api_key, "crud-list-completed")
        failed = _make_run(db, user_api_key, "crud-list-failed")
        update_evaluation_iteration_run(
            session=db,
            iteration_run=completed,
            update=EvaluationIterationRunUpdate(
                status=EvaluationIterationStatusEnum.COMPLETED
            ),
        )
        update_evaluation_iteration_run(
            session=db,
            iteration_run=failed,
            update=EvaluationIterationRunUpdate(
                status=EvaluationIterationStatusEnum.FAILED
            ),
        )

        ids = {r.id for r in list_processing_evaluation_iteration_runs(session=db)}

        assert processing.id in ids
        assert completed.id not in ids
        assert failed.id not in ids


class TestUpdateEvaluationIterationRun:
    def test_partial_update_only_touches_set_fields(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        run = _make_run(db, user_api_key, "crud-update")

        updated = update_evaluation_iteration_run(
            session=db,
            iteration_run=run,
            update=EvaluationIterationRunUpdate(
                status=EvaluationIterationStatusEnum.FAILED,
                error_message="boom",
            ),
        )

        assert updated.status == EvaluationIterationStatusEnum.FAILED
        assert updated.error_message == "boom"
        # stop_reason was never set on the update payload, so it stays untouched.
        assert updated.stop_reason is None
