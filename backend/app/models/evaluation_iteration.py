from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import HttpUrl
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import ENUM
from sqlmodel import Field, SQLModel

from app.core.config import settings
from app.core.util import now


class EvaluationIterationStatusEnum(StrEnum):
    """Top-level bookkeeping only — the round-by-round state lives in the
    LangGraph checkpoint, not on this row. This just answers "is this loop
    still in flight?" for the cron resume tick and the API."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Without values_callable, SQLAlchemy's auto-derived Enum stores the member
# *name* (e.g. "PROCESSING"), but the migration created the Postgres enum type
# with the lowercase member *values* — mirrors the RunModeEnum/ConfigTag pattern.
_ITERATION_STATUS_PG_ENUM = ENUM(
    EvaluationIterationStatusEnum,
    name="evaluationiterationstatusenum",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
    create_type=False,
)


class EvaluationIterationRun(SQLModel, table=True):
    """Thin per-loop tracking row for the eval-iterate-improve LangGraph loop.

    `id` doubles as the LangGraph `thread_id` (`str(id)`) — no separate
    thread_id column needed, since a loop and its checkpoint thread are 1:1.
    """

    __tablename__ = "evaluation_iteration_run"

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={
            "comment": "Unique identifier; str(id) is the LangGraph thread_id"
        },
    )
    dataset_id: int = Field(
        foreign_key="evaluation_dataset.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Reference to the evaluation dataset iterated against"
        },
    )
    experiment_name: str = Field(
        max_length=255,
        description="Base name for derived per-round eval run names",
        sa_column_kwargs={
            "comment": "Base name; each round's eval run is named f'{experiment_name}-iter{id}-r{round_number}'"
        },
    )
    config_id: UUID = Field(
        foreign_key="config.id",
        nullable=False,
        index=True,
        ondelete="RESTRICT",
        sa_column_kwargs={
            "comment": "Reference to the stored config being iterated on"
        },
    )
    initial_config_version: int = Field(
        ge=1,
        description="Config version supplied at kickoff",
        sa_column_kwargs={
            "comment": "Config version supplied at kickoff, recorded for the report"
        },
    )
    status: EvaluationIterationStatusEnum = Field(
        default=EvaluationIterationStatusEnum.PROCESSING,
        sa_column=Column(
            _ITERATION_STATUS_PG_ENUM,
            nullable=False,
            index=True,
            comment="Loop bookkeeping status: processing, completed, or failed",
        ),
    )
    stop_reason: str | None = Field(
        default=None,
        sa_column_kwargs={
            "comment": "Copied from the final graph state once terminal: ceiling_reached, max_rounds_reached, or round_failed"
        },
    )
    callback_url: str = Field(
        sa_column_kwargs={
            "comment": "HTTPS webhook validated via validate_callback_url before create"
        },
    )
    error_message: str | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Error detail if the loop failed"},
    )
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the iteration loop was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the iteration loop was last updated"
        },
    )


class EvaluationIterationRunUpdate(SQLModel):
    """Partial update payload, `exclude_unset` semantics."""

    status: EvaluationIterationStatusEnum | None = None
    stop_reason: str | None = None
    error_message: str | None = None


class EvaluationIterationCreateRequest(SQLModel):
    """Body for POST /api/v2/evaluations/iterations."""

    dataset_id: int = Field(description="ID of the evaluation dataset")
    experiment_name: str = Field(
        min_length=3, description="Base name for this iteration loop's per-round runs"
    )
    config_id: UUID = Field(description="Stored config ID to start iterating from")
    config_version: int = Field(ge=1, description="Starting stored config version")
    max_rounds: int | None = Field(
        default=None,
        ge=1,
        le=settings.EVAL_ITERATION_MAX_ROUNDS_HARD_CAP,
        description=(
            "Safety cap on rounds; defaults to EVAL_ITERATION_MAX_ROUNDS_DEFAULT, "
            "rejected above EVAL_ITERATION_MAX_ROUNDS_HARD_CAP"
        ),
    )
    callback_url: HttpUrl = Field(
        description="HTTPS webhook that receives the round-by-round report once the loop stops."
    )


class EvaluationIterationRunImmediatePublic(SQLModel):
    """202 response body: the loop was created and the first round dispatched."""

    iteration_run_id: int
    status: EvaluationIterationStatusEnum
    message: str
    inserted_at: datetime
    updated_at: datetime


class EvaluationIterationRoundPublic(SQLModel):
    """One round's outcome — mirrors a `history` entry in the graph checkpoint state."""

    round_number: int
    eval_run_id: int
    config_version: int
    stop_score: float
    kb_score: float | None = Field(
        default=None,
        description="Adherence to Knowledge Base, recorded for visibility only — never gates the stop condition",
    )


class EvaluationIterationReportPublic(SQLModel):
    """Callback payload POSTed to `callback_url` once the loop reaches a terminal state."""

    iteration_run_id: int
    status: EvaluationIterationStatusEnum
    stop_reason: str | None
    best_round: EvaluationIterationRoundPublic | None
    history: list[EvaluationIterationRoundPublic]
    error_message: str | None = None
