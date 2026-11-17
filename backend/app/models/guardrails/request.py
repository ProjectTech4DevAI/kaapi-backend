from typing import Any

from pydantic import HttpUrl
from sqlmodel import Field, SQLModel

from app.models.llm.request import GuardrailEntry


class GuardrailsRequest(SQLModel):
    """Request body for ``POST /api/v1/guardrails``.

    The endpoint is symmetric for input and output guardrails: send the text
    that needs sanitisation in ``text`` along with the validators to apply.
    The sanitised text is delivered via the configured ``callback_url``.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hi, I am Priya, my phone number is 98765 43210. Is fever normal after vaccination?",
                    "config": [
                        {"type": "pii_remover", "on_fail": "fix"},
                        {"validator_config_id": "8c1d2e3f-4a5b-4c6d-9e7f-0a1b2c3d4e5f"},
                    ],
                    "callback_url": "https://example.com/webhooks/guardrails",
                    "request_metadata": {"message_id": "msg-1042"},
                }
            ]
        }
    }

    text: str = Field(
        ...,
        min_length=1,
        description="Text to validate/sanitise. May be a user prompt or an LLM response.",
    )
    config: list[GuardrailEntry] = Field(
        ...,
        min_length=1,
        description=(
            "Validators to apply: `{validator_config_id}` for a stored config, "
            "or an inline spec `{type, ...params}`."
        ),
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
