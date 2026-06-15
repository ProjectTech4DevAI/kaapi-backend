from typing import Any
from uuid import UUID

from pydantic import HttpUrl
from sqlmodel import Field, SQLModel


class GuardrailValidator(SQLModel):
    """Single validator reference inside a /guardrails request.

    Only ``validator_config_id`` is meaningful to the server. ``type`` and
    ``tag`` are caller-side bookkeeping fields: they are accepted and
    persisted on ``job.meta.request`` for traceability, but they are not
    echoed back on the webhook payload. To correlate webhooks with
    requests, put your correlation keys in ``request_metadata`` (which is
    echoed under the callback's ``metadata`` field).
    """

    validator_config_id: UUID = Field(
        ...,
        description="ID of a validator configuration registered with the guardrails service.",
    )
    type: str | None = Field(
        default=None,
        description=(
            "Optional bookkeeping label for the caller (e.g. 'input_guardrail' "
            "or 'output_guardrail'). Server does not interpret this field."
        ),
    )
    tag: str | None = Field(
        default=None,
        description="Optional caller-side tag. Server does not interpret this field.",
    )


class GuardrailsRequest(SQLModel):
    """Request body for ``POST /api/v1/guardrails``.

    The endpoint is symmetric for input and output guardrails: send the text
    that needs sanitisation in ``text`` along with the validator IDs to apply.
    The sanitised text is delivered via the configured ``callback_url``.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="Text to validate/sanitise. May be a user prompt or an LLM response.",
    )
    guardrail_config: list[GuardrailValidator] = Field(
        ...,
        min_length=1,
        description="Validators to apply, identified by validator_config_id.",
    )
    callback_url: HttpUrl | None = Field(
        default=None,
        description=(
            "Webhook URL that will receive the sanitised result. When omitted "
            "the caller must poll GET /guardrails/{job_id}."
        ),
    )
    request_metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Client-provided metadata passed through unchanged in the callback's "
            "`metadata` field. Use this to correlate callbacks with requests. "
            "Note: the server appends a `warnings: [...]` key to this object "
            "on the wire; any caller-supplied `warnings` key will be overwritten."
        ),
    )
