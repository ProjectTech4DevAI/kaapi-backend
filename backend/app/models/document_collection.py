from uuid import UUID

from sqlmodel import Field, SQLModel, UniqueConstraint


class DocumentCollection(SQLModel, table=True):
    """Junction table linking documents to collections."""

    __table_args__ = (
        UniqueConstraint("document_id", "collection_id", name="uq_document_collection"),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={
            "comment": "Unique identifier for the document-collection link"
        },
    )
    document_id: UUID = Field(
        foreign_key="document.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the document"},
    )
    collection_id: UUID = Field(
        foreign_key="collection.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the collection"},
    )
