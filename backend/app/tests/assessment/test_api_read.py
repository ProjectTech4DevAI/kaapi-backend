"""Tests for the API-client read endpoints: the assessment list and the poll detail."""

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.crud.assessment import api
from app.models.assessment import (
    AssessmentMethod,
    AssessmentStatus,
    BatchRunState,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.services.assessment.api.results import build_detail, build_summary
from app.tests.utils.auth import get_user_test_auth_context
from app.tests.utils.test_data import create_test_config
from app.tests.utils.utils import random_lower_string

_STORE = "app.services.assessment.api.results.get_cloud_storage"
_ROWS = "app.services.assessment.api.submission_store.load_submission_rows"


def _config(db, project_id):
    blob = AssessmentConfigBlob.model_validate(
        {
            "input_schema": {"a": {"type": "text"}},
            "assessment": {
                "provider": "openai",
                "type": "text",
                "params": {"model": "gpt-4o", "submission": "Assess {a}"},
            },
        }
    )
    return create_test_config(
        db,
        project_id=project_id,
        name=f"assess-{random_lower_string()}",
        config_blob=blob,
        tag=ConfigTag.ASSESSMENT,
    )


def _bag(total: int, **overrides) -> BatchRunState:
    bag: BatchRunState = {
        "pipeline": [{"stage": "assessment", "kind": "ASSESSMENT"}],
        "stage": "assessment",
        "stage_status": "PROCESSING",
        "stage_batches": {},
        "stage_output_urls": {},
        "verdicts": {},
        "counters": {},
        "gate_passed": [True] * total,
        "provider": "openai",
        "model": "gpt-4o",
        "input_schema": {"a": {"type": "text"}},
        "callback_url": None,
        "request_metadata": None,
    }
    bag.update(overrides)  # type: ignore[typeddict-item]
    return bag


def _seed(db, auth, *, total=2, config=None, name="run", bag=None):
    assessment = api.create_assessment(
        session=db,
        method=AssessmentMethod.BATCH,
        input=None,
        experiment_name=name,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )
    execution = api.create_execution(
        session=db,
        assessment_id=assessment.id,
        config_id=(config or _config(db, auth.project_id)).id,
        config_version=1,
        total_items=total,
    )
    api.save_execution_state(session=db, execution=execution, state=bag or _bag(total))
    return assessment, execution


class TestListFilters:
    def test_lists_newest_first(self, db) -> None:
        auth = get_user_test_auth_context(db)
        _seed(db, auth, name="older")
        _seed(db, auth, name="newer")

        rows = api.list_assessments_with_execution(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        names = [a.experiment_name for a, _, _ in rows]
        assert names.index("newer") < names.index("older")

    def test_config_and_version_narrow_the_list(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _config(db, auth.project_id)
        other = _config(db, auth.project_id)
        _seed(db, auth, config=config, name="wanted")
        _seed(db, auth, config=other, name="other-config")

        rows = api.list_assessments_with_execution(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            config_version=1,
        )
        assert [a.experiment_name for a, _, _ in rows] == ["wanted"]

        missed = api.list_assessments_with_execution(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            config_id=config.id,
            config_version=2,
        )
        assert missed == []

    def test_method_filter(self, db) -> None:
        auth = get_user_test_auth_context(db)
        _seed(db, auth, name="batch-run")

        assert (
            api.list_assessments_with_execution(
                session=db,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
                method=AssessmentMethod.RUN,
            )
            == []
        )

    def test_scoped_to_the_project(self, db) -> None:
        auth = get_user_test_auth_context(db)
        _seed(db, auth)

        assert (
            api.list_assessments_with_execution(
                session=db,
                organization_id=auth.organization_id,
                project_id=auth.project_id + 1,
            )
            == []
        )


class TestSummary:
    def test_reads_stage_off_the_bag_without_touching_storage(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth, total=3)

        with patch(_STORE) as storage:
            summary = build_summary(assessment, execution)

        storage.assert_not_called()
        assert summary.total_items == 3
        assert summary.stage == "assessment"
        assert summary.stage_status == "PROCESSING"
        assert summary.config is not None

    def test_survives_a_missing_execution(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment = api.create_assessment(
            session=db,
            method=AssessmentMethod.BATCH,
            input=None,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        summary = build_summary(assessment, None)
        assert summary.total_items == 0
        assert summary.config is None


class TestDetail:
    def test_mid_run_returns_placeholders_with_input_attached(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth, total=2)

        from app.models.assessment import BatchInput

        with patch(_ROWS, return_value=BatchInput(data=[{"a": "one"}, {"a": "two"}])):
            detail = build_detail(session=db, assessment=assessment)

        assert detail.status == AssessmentStatus.PENDING
        assert detail.total_items == 2
        assert [row.row_index for row in detail.items] == [0, 1]
        assert detail.items[0].input == {"a": "one"}
        assert detail.items[0].output.assessment is None

    def test_unreadable_submission_degrades_to_null_input(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth, total=1)

        from app.services.assessment.api.submission_store import (
            SubmissionUnavailableError,
        )

        with patch(_ROWS, side_effect=SubmissionUnavailableError("s3 down")):
            detail = build_detail(session=db, assessment=assessment)

        assert detail.items[0].input is None
        assert detail.total_items == 1

    def test_gated_rows_carry_their_verdict(self, db) -> None:
        auth = get_user_test_auth_context(db)
        bag = _bag(
            2,
            gate_passed=[False, True],
            verdicts={
                "topic_relevance": {
                    "0": {"verdict": False, "reasoning": "off topic"},
                    "1": {"verdict": True, "reasoning": "on topic"},
                }
            },
        )
        assessment, _ = _seed(db, auth, total=2, bag=bag)

        from app.models.assessment import BatchInput

        with patch(_ROWS, return_value=BatchInput(data=[{"a": "1"}, {"a": "2"}])):
            detail = build_detail(session=db, assessment=assessment)

        gated = detail.items[0]
        assert gated.output.assessment is None
        assert gated.output.pre_filter is not None
        assert gated.output.pre_filter.topic_relevance.verdict is False
        assert detail.counts.filtered == 1

    def test_input_shorter_than_total_items(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth, total=3)

        from app.models.assessment import BatchInput

        with patch(_ROWS, return_value=BatchInput(data=[{"a": "only one"}])):
            detail = build_detail(session=db, assessment=assessment)

        assert detail.items[0].input == {"a": "only one"}
        assert detail.items[1].input is None
        assert detail.items[2].input is None


class TestDetailRouteGuards:
    """The route wraps `auth_context.organization_` / `.project_`, unlike the crud."""

    @staticmethod
    def _route_context(auth):
        return SimpleNamespace(
            organization_=SimpleNamespace(id=auth.organization_id),
            project_=SimpleNamespace(id=auth.project_id),
        )

    def test_run_assessment_is_rejected(self, db) -> None:
        from app.api.routes.assessment.api import get_assessment_detail

        auth = get_user_test_auth_context(db)
        assessment = api.create_assessment(
            session=db,
            method=AssessmentMethod.RUN,
            input={"prompt": "p"},
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )

        with pytest.raises(HTTPException) as exc:
            get_assessment_detail(
                assessment_id=assessment.id,
                session=db,
                auth_context=self._route_context(auth),
            )
        assert exc.value.status_code == 422

    def test_unknown_assessment_is_404(self, db) -> None:
        from app.api.routes.assessment.api import get_assessment_detail

        auth = get_user_test_auth_context(db)
        with pytest.raises(HTTPException) as exc:
            get_assessment_detail(
                assessment_id=uuid4(),
                session=db,
                auth_context=self._route_context(auth),
            )
        assert exc.value.status_code == 404
