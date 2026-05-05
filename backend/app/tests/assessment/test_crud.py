"""Tests for assessment/crud.py."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.crud.assessment import (
    AssessmentRunCounts,
    build_run_stats,
    compute_run_counts,
    create_assessment,
    create_assessment_dataset,
    create_assessment_run,
    derive_aggregate_error,
    derive_assessment_status,
    get_assessment_by_id,
    get_assessment_run_by_id,
    get_assessment_runs_for_assessment,
    list_assessment_runs,
    list_assessments,
    recompute_assessment_status,
    update_assessment_run_status,
)
from app.models.stt_evaluation import EvaluationType


def _counts(total=0, pending=0, processing=0, completed=0, failed=0):
    return AssessmentRunCounts(
        total=total,
        pending=pending,
        processing=processing,
        completed=completed,
        failed=failed,
    )


class TestDeriveAssessmentStatus:
    def test_status_variants(self) -> None:
        assert derive_assessment_status(_counts()) == "pending"
        assert derive_assessment_status(_counts(total=2, completed=2)) == "completed"
        assert derive_assessment_status(_counts(total=2, failed=2)) == "failed"
        assert (
            derive_assessment_status(_counts(total=2, completed=1, failed=1))
            == "completed_with_errors"
        )
        assert derive_assessment_status(_counts(total=2, pending=2)) == "pending"
        assert (
            derive_assessment_status(_counts(total=2, pending=1, processing=1))
            == "processing"
        )


class TestCrudBasicQueries:
    def test_get_and_list_helpers(self) -> None:
        session = MagicMock()
        session.exec.return_value.first.return_value = "assessment"
        session.exec.return_value.all.return_value = ["a1", "a2"]

        assert get_assessment_by_id(session, 1, 1, 1) == "assessment"
        assert list_assessments(session, 1, 1, 10, 0) == ["a1", "a2"]
        assert get_assessment_run_by_id(session, 1, 1, 1) == "assessment"
        assert list_assessment_runs(session, 1, 1, None, 10, 0) == ["a1", "a2"]

    def test_get_assessment_by_id_not_found(self) -> None:
        session = MagicMock()
        session.exec.return_value.first.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            get_assessment_by_id(session, 99, 1, 1)
        assert exc_info.value.status_code == 404
        assert "99" in exc_info.value.detail

    def test_get_assessment_run_by_id_not_found(self) -> None:
        session = MagicMock()
        session.exec.return_value.first.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            get_assessment_run_by_id(session, 99, 1, 1)
        assert exc_info.value.status_code == 404
        assert "99" in exc_info.value.detail

    def test_get_assessment_runs_for_assessment(self) -> None:
        session = MagicMock()
        session.exec.return_value.all.return_value = ["r1", "r2"]
        assert get_assessment_runs_for_assessment(session, 10) == ["r1", "r2"]


class TestCrudWrites:
    def test_create_assessment_dataset_uses_assessment_type(self) -> None:
        session = MagicMock()
        result = create_assessment_dataset(
            session=session,
            name="dataset",
            description="desc",
            dataset_metadata={"total_items_count": 2},
            object_store_url="s3://datasets/file.csv",
            organization_id=1,
            project_id=1,
        )

        assert result.type == EvaluationType.ASSESSMENT.value
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

    def test_create_assessment_success(self) -> None:
        session = MagicMock()
        result = create_assessment(
            session=session,
            experiment_name="exp",
            dataset_id=1,
            organization_id=1,
            project_id=1,
        )
        assert result.experiment_name == "exp"
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

    def test_create_assessment_commit_failure_rolls_back(self) -> None:
        session = MagicMock()
        session.commit.side_effect = RuntimeError("db error")
        with pytest.raises(RuntimeError):
            create_assessment(session, "exp", 1, 1, 1)
        session.rollback.assert_called_once()

    def test_create_assessment_run_success_and_failure(self) -> None:
        session = MagicMock()
        run = create_assessment_run(
            session=session,
            assessment_id=10,
            config_id=UUID("00000000-0000-0000-0000-000000000001"),
            config_version=1,
            assessment_input={"k": "v"},
        )
        assert run.assessment_id == 10
        assert run.input == {"k": "v"}

        session2 = MagicMock()
        session2.commit.side_effect = RuntimeError("db error")
        with pytest.raises(RuntimeError):
            create_assessment_run(
                session=session2,
                assessment_id=10,
                config_id=UUID("00000000-0000-0000-0000-000000000001"),
                config_version=1,
                assessment_input={},
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


class TestDerivedAggregates:
    def test_compute_run_counts(self) -> None:
        runs = [
            SimpleNamespace(status="completed"),
            SimpleNamespace(status="failed"),
            SimpleNamespace(status="processing"),
            SimpleNamespace(status="pending"),
        ]
        counts = compute_run_counts(runs)
        assert counts.total == 4
        assert counts.completed == 1
        assert counts.failed == 1
        assert counts.processing == 1
        assert counts.pending == 1

    def test_build_run_stats(self) -> None:
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
        ]
        stats = build_run_stats(runs)
        assert len(stats) == 1
        assert stats[0].run_id == 1
        assert stats[0].status == "completed"

    def test_derive_aggregate_error(self) -> None:
        assert derive_aggregate_error(_counts(total=2, completed=2)) is None
        assert (
            derive_aggregate_error(_counts(total=3, completed=1, failed=2))
            == "2 of 3 run(s) failed"
        )


class TestRecomputeAssessmentStatus:
    def test_recompute_not_found(self) -> None:
        session = MagicMock()
        session.get.return_value = None
        with pytest.raises(ValueError, match="not found"):
            recompute_assessment_status(session=session, assessment_id=1)

    def test_recompute_success_persists_status_only(self) -> None:
        session = MagicMock()
        assessment = SimpleNamespace(
            id=1, status="pending", updated_at=datetime(2024, 1, 1)
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
        session.commit.assert_called_once()

    def test_recompute_commit_failure_rolls_back(self) -> None:
        session = MagicMock()
        assessment = SimpleNamespace(
            id=1, status="pending", updated_at=datetime(2024, 1, 1)
        )
        session.get.return_value = assessment
        session.exec.return_value.all.return_value = []
        session.commit.side_effect = RuntimeError("db error")
        with pytest.raises(RuntimeError):
            recompute_assessment_status(session=session, assessment_id=1)
        session.rollback.assert_called_once()
