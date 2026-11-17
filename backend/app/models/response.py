from sqlmodel import SQLModel


class ResponsesAPIRequest(SQLModel):
    assistant_id: str
    question: str
    callback_url: str | None = None
    response_id: str | None = None

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "assistant_id": "asst_abc123def456",
                    "question": "What documents do I need to apply for the scholarship?",
                    "callback_url": "https://example.com/webhooks/responses",
                }
            ]
        },
    }


class ResponsesSyncAPIRequest(SQLModel):
    model: str
    instructions: str
    vector_store_ids: list[str]
    max_num_results: int = 20
    temperature: float = 0.1
    response_id: str | None = None
    question: str

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "model": "gpt-4o-mini",
                    "instructions": "Answer using only the provided documents.",
                    "vector_store_ids": ["vs_abc123def456"],
                    "max_num_results": 20,
                    "temperature": 0.1,
                    "question": "What documents do I need to apply for the scholarship?",
                }
            ]
        },
    }


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
    filename: str | None = None


class CallbackResponse(SQLModel):
    status: str
    response_id: str
    message: str
    diagnostics: Diagnostics | None = None

    class Config:
        extra = "allow"
