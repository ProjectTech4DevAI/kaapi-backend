from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from app.celery.tasks import job_execution
from app.crud.evaluations.core import create_evaluation_run
from app.models import EvaluationDataset, EvaluationRun, Organization, Project
from app.services.evaluations.batch_result_processing import (
    execute_evaluation_batch_result_processing,
)
from app.tests.utils.test_data import create_test_config, create_test_evaluation_dataset
from app.tests.utils.utils import random_lower_string

SERVICE = "app.services.evaluations.batch_result_processing"


class _BridgedSession:
    """Point the service's own ``with Session(engine)`` scope at the test session.

    Commits stay inside the conftest transaction (rolled back at teardown) and
    the service sees rows seeded by the test. ``rollback`` is neutered: in
    production the dispatcher commits the run in a *separate* transaction before
    the worker starts, so the worker's defensive rollback discards only its own
    in-flight work; under the shared test transaction a real rollback would also
    wipe the seeded run. The mocked failure leaves the session clean, so skipping
    that recovery rollback changes nothing observable.
    """

    def __init__(self, session: Session):
        object.__setattr__(self, "_session", session)

    def __enter__(self) -> "_BridgedSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def rollback(self) -> None:
        pass

    def __getattr__(self, name: str):
        return getattr(self._session, name)


@pytest.fixture
def patched_session(db: Session):
    with patch(
        f"{SERVICE}.Session",
        side_effect=lambda _engine: _BridgedSession(db),
    ):
        yield db


@pytest.fixture
def eval_run(db: Session) -> EvaluationRun:
    org = db.exec(select(Organization)).first()
    project = db.exec(select(Project).where(Project.organization_id == org.id)).first()
    dataset: EvaluationDataset = create_test_evaluation_dataset(
        db=db,
        organization_id=org.id,
        project_id=project.id,
        name="test_dataset_batch_result",
        description="Test dataset",
        original_items_count=1,
        duplication_factor=1,
    )
    config = create_test_config(db, project_id=project.id, use_kaapi_schema=True)
    run = create_evaluation_run(
        session=db,
        run_name=random_lower_string(),
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=1,
        organization_id=org.id,
        project_id=project.id,
    )
    run.status = "processing"
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


class TestExecuteEvaluationBatchResultProcessing:
    def test_happy_path_returns_result_and_flushes(
        self, patched_session: Session, eval_run: EvaluationRun
    ):
        langfuse = MagicMock()
        with (
            patch(f"{SERVICE}.get_openai_client", return_value=MagicMock()),
            patch(f"{SERVICE}.get_tracing_client", return_value=langfuse),
            patch(
                f"{SERVICE}.check_and_process_evaluation",
                new=AsyncMock(return_value={"action": "processed"}),
            ),
        ):
            result = execute_evaluation_batch_result_processing(
                project_id=eval_run.project_id,
                eval_run_id=eval_run.id,
                trace_id="N/A",
            )

        assert result == {"action": "processed"}
        langfuse.flush.assert_called_once()

    def test_missing_run_returns_none_and_builds_no_client(
        self, patched_session: Session
    ):
        with (
            patch(f"{SERVICE}.get_openai_client") as mock_openai,
            patch(f"{SERVICE}.get_tracing_client") as mock_langfuse,
        ):
            result = execute_evaluation_batch_result_processing(
                project_id=1,
                eval_run_id=99_999_999,
                trace_id="N/A",
            )

        assert result is None
        mock_openai.assert_not_called()
        mock_langfuse.assert_not_called()

    def test_failure_marks_run_failed_reraises_and_flushes(
        self, patched_session: Session, eval_run: EvaluationRun
    ):
        langfuse = MagicMock()
        boom = RuntimeError("provider exploded")
        with (
            patch(f"{SERVICE}.get_openai_client", return_value=MagicMock()),
            patch(f"{SERVICE}.get_tracing_client", return_value=langfuse),
            patch(
                f"{SERVICE}.check_and_process_evaluation",
                new=AsyncMock(side_effect=boom),
            ),
            pytest.raises(RuntimeError, match="provider exploded"),
        ):
            execute_evaluation_batch_result_processing(
                project_id=eval_run.project_id,
                eval_run_id=eval_run.id,
                trace_id="N/A",
            )

        patched_session.refresh(eval_run)
        assert eval_run.status == "failed"
        assert "provider exploded" in eval_run.error_message
        langfuse.flush.assert_called_once()


class TestRunEvaluationBatchResultProcessingShim:
    def test_forwards_eval_run_id_to_service(self):
        with patch(
            f"{SERVICE}.execute_evaluation_batch_result_processing",
            return_value={"action": "processed"},
        ) as mock_execute:
            result = job_execution.run_evaluation_batch_result_processing.apply(
                kwargs={
                    "project_id": 7,
                    "job_id": "42",
                    "trace_id": "N/A",
                    "eval_run_id": 42,
                }
            ).result

        assert result == {"action": "processed"}
        mock_execute.assert_called_once_with(
            project_id=7, eval_run_id=42, trace_id="N/A"
        )
