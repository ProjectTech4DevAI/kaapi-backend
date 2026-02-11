"""File model for storing uploaded files metadata."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

from app.core.util import now


class FileType(str, Enum):
    """Type of file stored."""

    AUDIO = "audio"
    DOCUMENT = "document"
    IMAGE = "image"
    OTHER = "other"


class File(SQLModel, table=True):
    """Database table for storing uploaded file metadata."""

    __tablename__ = "file"

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the file"},
    )

    object_store_url: str = SQLField(
        description="S3 URL of the file",
        sa_column_kwargs={"comment": "S3 URL where the file is stored"},
    )

    filename: str = SQLField(
        max_length=255,
        description="Original filename",
        sa_column_kwargs={"comment": "Original filename as uploaded"},
    )

    size_bytes: int = SQLField(
        description="File size in bytes",
        sa_column_kwargs={"comment": "File size in bytes"},
    )

    content_type: str = SQLField(
        max_length=100,
        description="MIME type of the file",
        sa_column_kwargs={"comment": "MIME type of the file (e.g., audio/mp3)"},
    )

    file_type: str = SQLField(
        default=FileType.OTHER.value,
        max_length=20,
        description="Type of file: audio, document, image, other",
        sa_column_kwargs={"comment": "Type of file: audio, document, image, other"},
    )

    organization_id: int = SQLField(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = SQLField(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )

    inserted_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the file was created"},
    )
    updated_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the file was last updated"},
    )


class FilePublic(BaseModel):
    """Public model for file responses."""

    id: int
    object_store_url: str
    filename: str
    size_bytes: int
    content_type: str
    file_type: str
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime
