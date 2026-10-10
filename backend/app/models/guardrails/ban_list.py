"""Ban list payloads proxied to the kaapi-guardrails service.

Mirrors `kaapi-guardrails/backend/app/schemas/ban_list.py` at the level of
field names and types only. Length and format rules are deliberately left to
the upstream service: duplicating them here would reject payloads upstream
would accept the moment the two drift apart.

Timestamps keep the upstream ``created_at``/``updated_at`` spelling rather
than the Kaapi-wide ``inserted_at`` because these objects are echoed verbatim
from the other service.
"""

from datetime import datetime
from uuid import UUID

from sqlmodel import Field, SQLModel


class BanListBase(SQLModel):
    name: str = Field(
        ..., description="Human-readable name for the ban list. Must be unique."
    )
    description: str = Field(
        ..., description="What this ban list covers and when it should be applied."
    )
    banned_words: list[str] = Field(
        ...,
        description=(
            "Words to redact. Consumed by the `ban_list` validator when it is "
            "given this list's `ban_list_id`."
        ),
    )
    domain: str = Field(
        ...,
        description=(
            "Caller-defined grouping label (e.g. 'abuse'). Used to filter lists "
            "on GET /guardrails/ban_lists."
        ),
    )
    is_public: bool = Field(
        default=False,
        description=(
            "When true the list is readable by other tenants. Updates and "
            "deletes remain restricted to the owning project."
        ),
    )


class BanListCreate(BanListBase):
    """Request body for ``POST /api/v1/guardrails/ban_lists``."""


class BanListUpdate(SQLModel):
    """Request body for ``PATCH /api/v1/guardrails/ban_lists/{ban_list_id}``.

    Omitted fields are left unchanged.
    """

    name: str | None = None
    description: str | None = None
    banned_words: list[str] | None = None
    domain: str | None = None
    is_public: bool | None = None


class BanListPublic(BanListBase):
    """A ban list as returned by the guardrails service."""

    id: UUID
    created_at: datetime
    updated_at: datetime
