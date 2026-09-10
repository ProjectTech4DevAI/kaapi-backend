"""Tests for assessment submission CRUD (app/crud/assessment/submission.py)."""

from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.crud.assessment import api
from app.crud.assessment.submission import (
    create_submission,
    delete_submission,
    get_submission_by_id,
    get_submission_by_name,
    list_submissions,
)
from app.models.assessment import AssessmentMethod
from app.tests.utils.auth import get_user_test_auth_context
from app.tests.utils.utils import random_lower_string


def _create(db, auth, **kwargs):
    return create_submission(
        session=db,
        name=kwargs.pop("name", random_lower_string()),
        object_store_url="s3://bucket/sub.csv",
        total_items=kwargs.pop("total_items", 3),
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        **kwargs,
    )


class TestCreate:
    def test_persists_the_row(self, db) -> None:
        auth = get_user_test_auth_context(db)
        submission = _create(db, auth, description="desc")

        assert submission.id is not None
        assert submission.total_items == 3
        assert submission.description == "desc"

    def test_duplicate_name_is_409(self, db) -> None:
        auth = get_user_test_auth_context(db)
        name = random_lower_string()
        _create(db, auth, name=name)

        with pytest.raises(HTTPException) as exc:
            _create(db, auth, name=name)
        assert exc.value.status_code == 409


class TestReads:
    def test_get_by_id_scoped_to_project(self, db) -> None:
        auth = get_user_test_auth_context(db)
        submission = _create(db, auth)

        found = get_submission_by_id(
            session=db,
            submission_id=submission.id,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        assert found.id == submission.id

        with pytest.raises(HTTPException) as exc:
            get_submission_by_id(
                session=db,
                submission_id=submission.id,
                organization_id=auth.organization_id,
                project_id=auth.project_id + 1,
            )
        assert exc.value.status_code == 404

    def test_get_by_id_unknown_is_404(self, db) -> None:
        auth = get_user_test_auth_context(db)
        with pytest.raises(HTTPException) as exc:
            get_submission_by_id(
                session=db,
                submission_id=uuid4(),
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 404

    def test_get_by_name_returns_none_when_absent(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assert (
            get_submission_by_name(
                session=db,
                name=random_lower_string(),
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
            is None
        )

    def test_list_returns_the_project_rows(self, db) -> None:
        auth = get_user_test_auth_context(db)
        created = _create(db, auth)

        rows = list_submissions(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )
        assert created.id in {row.id for row in rows}


class TestDelete:
    def test_deletes_an_unreferenced_submission(self, db) -> None:
        auth = get_user_test_auth_context(db)
        submission = _create(db, auth)

        assert delete_submission(session=db, submission=submission) is None
        with pytest.raises(HTTPException):
            get_submission_by_id(
                session=db,
                submission_id=submission.id,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )

    def test_refuses_while_an_assessment_references_it(self, db) -> None:
        auth = get_user_test_auth_context(db)
        submission = _create(db, auth)
        api.create_assessment(
            session=db,
            method=AssessmentMethod.BATCH,
            input=None,
            submission_id=submission.id,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )

        error = delete_submission(session=db, submission=submission)
        assert error is not None
        assert "being used by" in error
