from datetime import datetime

from sqlmodel import Field, SQLModel

from app.core.util import now


class OpenAIThreadBase(SQLModel):
    thread_id: str = Field(index=True, unique=True)
    prompt: str
    response: str | None = None
    status: str | None = None
    error: str | None = None


class OpenAIThreadCreate(OpenAIThreadBase):
    pass  # Used for requests, no `id` or timestamps


class OpenAIThread(OpenAIThreadBase, table=True):
    """Stores OpenAI thread interactions and their responses."""

    __tablename__ = "openai_thread"

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the thread record"},
    )
    thread_id: str = Field(
        index=True,
        unique=True,
        sa_column_kwargs={"comment": "OpenAI thread identifier"},
    )
    prompt: str = Field(
        sa_column_kwargs={"comment": "User prompt sent to the thread"},
    )
    response: str | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Response received from OpenAI"},
    )
    status: str | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Current status of the thread interaction"},
    )
    error: str | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Error message if the interaction failed"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={"comment": "Timestamp when the record was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={
            "comment": "Timestamp when the record was last updated",
            "onupdate": now,
        },
    )
