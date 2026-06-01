from sqlmodel import SQLModel


class ResponsesAPIRequest(SQLModel):
    assistant_id: str
    question: str
    callback_url: str | None = None
    response_id: str | None = None

    class Config:
        extra = "allow"


class ResponsesSyncAPIRequest(SQLModel):
    model: str
    instructions: str
    vector_store_ids: list[str]
    max_num_results: int = 20
    temperature: float = 0.1
    response_id: str | None = None
    question: str

    class Config:
        extra = "allow"


class ResponseJobStatus(SQLModel):
    status: str
    message: str | None = None

    class Config:
        extra = "allow"


class Diagnostics(SQLModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str


class FileResultChunk(SQLModel):
    score: float
    text: str


class CallbackResponse(SQLModel):
    status: str
    response_id: str
    message: str
    diagnostics: Diagnostics | None = None

    class Config:
        extra = "allow"
