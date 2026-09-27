"""Uploaded submission files for the assessment domain.

Its own table rather than a row in ``evaluation_dataset``: that table multiplexes four
surfaces behind a ``type`` column, carries eval-only fields, and its name uniqueness is
type-agnostic, so an eval dataset name blocks an assessment one.
"""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel, UniqueConstraint

from app.core.util import now


class AssessmentSubmission(SQLModel, table=True):
    """One uploaded submission file (CSV/XLSX) an assessment can be run against."""

    __tablename__ = "assessment_submission"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "organization_id",
            "project_id",
            name="uq_assessment_submission_name_org_project",
        ),
    )

    id: UUID = SQLField(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the submission"},
    )
    name: str = SQLField(
        index=True,
        sa_column_kwargs={
            "comment": "Sanitized name; the object key is derived from it"
        },
    )
    description: str | None = SQLField(
        default=None, sa_column_kwargs={"comment": "Optional description"}
    )
    object_store_url: str = SQLField(
        sa_column_kwargs={
            "comment": "Object-store url of the uploaded file; its suffix gives the format"
        }
    )
    total_items: int = SQLField(
        default=0, sa_column_kwargs={"comment": "Row count, excluding the header"}
    )

    organization_id: int = SQLField(
        foreign_key="organization.id", nullable=False, ondelete="CASCADE"
    )
    project_id: int = SQLField(
        foreign_key="project.id", nullable=False, ondelete="CASCADE"
    )
    inserted_at: datetime = SQLField(default_factory=now, nullable=False)
    updated_at: datetime = SQLField(default_factory=now, nullable=False)


class AssessmentSubmissionPreview(BaseModel):
    """First N rows of a submission file, for the upload confirmation screen."""

    headers: list[str]
    rows: list[list[str]]
    returned_rows: int = 0
    truncated: bool = False


class AssessmentSubmissionResponse(BaseModel):
    """API shape for a stored submission; ``signed_url`` is minted per request."""

    submission_id: UUID
    name: str
    description: str | None = None
    total_items: int = 0
    object_store_url: str | None = None
    signed_url: str | None = None
    preview: AssessmentSubmissionPreview | None = None
