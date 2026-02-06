from datetime import datetime

from sqlmodel import Field, SQLModel

from app.core.util import now


# Shared properties for a Language
class LanguageBase(SQLModel):
    """Base model for languages with common data fields."""

    label: str = Field(
        max_length=255,
        sa_column_kwargs={"comment": "Language name in English (e.g., 'Hindi')"},
    )
    label_locale: str = Field(
        max_length=255,
        sa_column_kwargs={
            "comment": "Language name in its native script (e.g., 'हिंदी')"
        },
    )
    description: str | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Optional description of the language"},
    )
    locale: str = Field(
        max_length=255,
        unique=True,
        sa_column_kwargs={"comment": "ISO 639-1 language code (e.g., 'en', 'hi')"},
    )
    is_active: bool = Field(
        default=True,
        sa_column_kwargs={"comment": "Flag indicating if the language is available"},
    )


# Database model for Language in the global schema
class Language(LanguageBase, table=True):
    """Database model for languages. Stored in the 'global' schema."""

    __tablename__ = "languages"
    __table_args__ = {"schema": "global"}

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the language"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the language was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the language was last updated"},
    )


# Properties to return via API
class LanguagePublic(LanguageBase):
    id: int
    inserted_at: datetime
    updated_at: datetime


class LanguagesPublic(SQLModel):
    data: list[LanguagePublic]
    count: int
