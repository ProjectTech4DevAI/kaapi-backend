"""Tests for assessment/crud.py."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from app.assessment.crud import (
    _determine_assessment_status,
    create_assessment,
    create_assessment_run,
    get_assessment_by_id,
    get_assessment_run_by_id,
    get_assessment_runs_for_manager,
    list_assessment_runs,
    list_assessments,
    recompute_assessment_status,
    update_assessment_run_status,
)


class TestDetermineAssessmentStatus:
    def test_status_variants(self) -> None:
        assert _determine_assessment_status(0, 0, 0, 0, 0) == "pending"
        assert _determine_assessment_status(2, 0, 0, 2, 0) == "completed"
        assert _determine_assessment_status(2, 0, 0, 0, 2) == "failed"
        assert _determine_assessment_status(2, 0, 0, 1, 1) == "completed_with_errors"
        assert _determine_assessment_status(2, 2, 0, 0, 0) == "pending"
        assert _determine_assessment_status(2, 1, 1, 0, 0) == "processing"


class TestCrudBasicQueries:
    def test_get_and_list_helpers(self) -> None:
        session = MagicMock()
        session.exec.return_value.first.return_value = "assessment"
        session.exec.return_value.all.return_value = ["a1", "a2"]

        assert get_assessment_by_id(session, 1, 1, 1) == "assessment"
        assert list_assessments(session, 1, 1, 10, 0) == ["a1", "a2"]
        assert get_assessment_run_by_id(session, 1, 1, 1) == "assessment"
        assert list_assessment_runs(session, 1, 1, None, 10, 0) == ["a1", "a2"]

    def test_get_assessment_runs_for_manager(self) -> None:
        session = MagicMock()
        session.exec.return_value.all.return_value = ["r1", "r2"]
        assessment = SimpleNamespace(id=10)
        assert get_assessment_runs_for_manager(session, assessment) == ["r1", "r2"]


class TestCrudWrites:
    def test_create_assessment_success(self) -> None:
        session = MagicMock()
        result = create_assessment(
            session=session,
            experiment_name="exp",
            dataset_id=1,
            dataset_name="ds",
            organization_id=1,
            project_id=1,
            total_runs=2,
        )
        assert result.experiment_name == "exp"
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

    def test_create_assessment_commit_failure_rolls_back(self) -> None:
        session = MagicMock()
        session.commit.side_effect = RuntimeError("db error")
        with pytest.raises(RuntimeError):
            create_assessment(session, "exp", 1, "ds", 1, 1, 1)
        session.rollback.assert_called_once()

    def test_create_assessment_run_success_and_failure(self) -> None:
        session = MagicMock()
        run = create_assessment_run(
            session=session,
            run_name="exp",
            dataset_name="ds",
            dataset_id=1,
            assessment_id=10,
            config_id=UUID("00000000-0000-0000-0000-000000000001"),
            config_version=1,
            organization_id=1,
            project_id=1,
            assessment_input={"k": "v"},
        )
        assert run.run_name == "exp"

        session2 = MagicMock()
        session2.commit.side_effect = RuntimeError("db error")
        with pytest.raises(RuntimeError):
            create_assessment_run(
                session=session2,
                run_name="exp",
                dataset_name="ds",
                dataset_id=1,
                assessment_id=10,
                config_id=UUID("00000000-0000-0000-0000-000000000001"),
                config_version=1,
                organization_id=1,
                project_id=1,
            )
        session2.rollback.assert_called_once()

    def test_update_assessment_run_status(self) -> None:
        session = MagicMock()
        run = SimpleNamespace(
            status="pending",
            updated_at=None,
            error_message=None,
            batch_job_id=None,
            total_items=0,
            object_store_url=None,
        )
        updated = update_assessment_run_status(
            session=session,
            run=run,
            status="processing",
            error_message="e",
            batch_job_id=11,
            total_items=9,
            object_store_url="s3://x",
        )
        assert updated.status == "processing"
        assert updated.error_message == "e"
        assert updated.batch_job_id == 11
        assert updated.total_items == 9
        assert updated.object_store_url == "s3://x"

    def test_update_assessment_run_status_failure_rolls_back(self) -> None:
        session = MagicMock()
        session.commit.side_effect = RuntimeError("db error")
        run = SimpleNamespace(
            status="pending",
            updated_at=None,
            error_message=None,
            batch_job_id=None,
            total_items=0,
            object_store_url=None,
        )
        with pytest.raises(RuntimeError):
            update_assessment_run_status(session, run, "failed")
        session.rollback.assert_called_once()


class TestRecomputeAssessmentStatus:
    def test_recompute_not_found(self) -> None:
        session = MagicMock()
        session.get.return_value = None
        with pytest.raises(ValueError, match="not found"):
            recompute_assessment_status(session=session, assessment_id=1)

    def test_recompute_success(self) -> None:
        session = MagicMock()
        assessment = SimpleNamespace(
            id=1,
            status="pending",
            total_runs=0,
            pending_runs=0,
            processing_runs=0,
            completed_runs=0,
            failed_runs=0,
            error_message=None,
            run_stats=[],
            updated_at=datetime(2024, 1, 1),
        )
        runs = [
            SimpleNamespace(
                id=1,
                config_id=UUID("00000000-0000-0000-0000-000000000001"),
                config_version=1,
                status="completed",
                total_items=2,
                error_message=None,
                updated_at=datetime(2024, 1, 1),
            ),
            SimpleNamespace(
                id=2,
                config_id=None,
                config_version=2,
                status="failed",
                total_items=2,
                error_message="bad",
                updated_at=datetime(2024, 1, 2),
            ),
        ]
        session.get.return_value = assessment
        session.exec.return_value.all.return_value = runs

        result = recompute_assessment_status(session=session, assessment_id=1)
        assert result.status == "completed_with_errors"
        assert result.error_message == "1 of 2 run(s) failed"
        assert len(result.run_stats) == 2
        session.commit.assert_called_once()

    def test_recompute_commit_failure_rolls_back(self) -> None:
        session = MagicMock()
        assessment = SimpleNamespace(
            id=1,
            status="pending",
            total_runs=0,
            pending_runs=0,
            processing_runs=0,
            completed_runs=0,
            failed_runs=0,
            error_message=None,
            run_stats=[],
            updated_at=datetime(2024, 1, 1),
        )
        session.get.return_value = assessment
        session.exec.return_value.all.return_value = []
        session.commit.side_effect = RuntimeError("db error")
        with pytest.raises(RuntimeError):
            recompute_assessment_status(session=session, assessment_id=1)
        session.rollback.assert_called_once()
