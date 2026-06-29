from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from sqlmodel import Field, SQLModel


class GuardrailsJobImmediatePublic(SQLModel):
    """Immediate 200 response from POST /guardrails before the job runs."""

    job_id: UUID
    status: str
    message: str
    job_inserted_at: datetime
    job_updated_at: datetime


class GuardrailsOutputContent(SQLModel):
    format: Literal["text"] = "text"
    value: str = Field(
        ..., description="Sanitised text returned by the guardrails service."
    )


class GuardrailsOutput(SQLModel):
    type: Literal["text"] = "text"
    content: GuardrailsOutputContent


class GuardrailsCallbackUsage(SQLModel):
    """Token usage reported by the upstream guardrails service, if any.

    All fields default to 0 so the callback shape stays stable when the
    upstream payload omits usage data.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


class GuardrailsCallbackResponse(SQLModel):
    response_id: str | None = Field(
        default=None,
        description="Response ID assigned by the guardrails service, if returned.",
    )
    output: GuardrailsOutput


class GuardrailsCallbackData(SQLModel):
    """Payload delivered to the webhook on completion.

    Wrapped by the standard APIResponse envelope: ``data`` carries this model,
    ``metadata`` carries the request's ``metadata`` plus any server warnings.
    """

    response: GuardrailsCallbackResponse
    usage: GuardrailsCallbackUsage = Field(default_factory=GuardrailsCallbackUsage)
    provider_raw_response: dict[str, Any] | None = None


class GuardrailsJobPublic(SQLModel):
    """Full job response for GET /guardrails/{job_id}."""

    job_id: UUID
    status: str
    guardrails_response: GuardrailsCallbackData | None = None
    error_message: str | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Server-emitted warnings for this job (e.g. "
            "'guardrails_service_unavailable_text_returned_unchanged'). "
            "Mirrors the `metadata.warnings` field of the callback payload "
            "so polling callers do not miss the bypass signal."
        ),
    )
