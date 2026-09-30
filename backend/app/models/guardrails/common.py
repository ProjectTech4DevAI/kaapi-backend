from sqlmodel import Field, SQLModel


class GuardrailsDeletePublic(SQLModel):
    """Body returned by the guardrails delete routes.

    The guardrails service answers deletes with 200 and a confirmation
    message rather than a 204, so the proxy surfaces a body here.
    """

    message: str = Field(..., description="Human-readable confirmation.")
