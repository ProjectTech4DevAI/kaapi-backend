from sqlmodel import Field, SQLModel


# Generic message
class Message(SQLModel):
    message: str


# Optional request body for delete endpoints that support hard deletion.
class DeleteRequest(SQLModel):
    hard_delete: bool = Field(
        default=False,
        description=(
            "When true, permanently delete the record and all of its related "
            "data. When false (the default), perform a soft delete by marking "
            "the record inactive."
        ),
    )
