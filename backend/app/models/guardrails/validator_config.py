"""Validator config payloads proxied to the kaapi-guardrails service.

Mirrors `kaapi-guardrails/backend/app/schemas/validator_config.py`. Timestamps
keep the upstream ``created_at``/``updated_at`` spelling because these objects
are owned by the guardrails service and echoed verbatim.

A validator config is stored as a fixed set of base columns plus a JSONB
``config`` blob holding the validator-specific tuning. The service flattens
those back into one object on the way out, which is why the create body and
the response are both open shapes.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ConfigDict
from sqlmodel import Field, SQLModel

from app.models.guardrails.enums import (
    GuardrailOnFailEnum,
    StageEnum,
    ValidatorTypeEnum,
)


class ValidatorConfigBase(SQLModel):
    name: str = Field(
        ...,
        description=(
            "Unique name for this config within the project. Uniqueness is "
            "enforced on the name alone, so the same validator type may be "
            "registered more than once under different names."
        ),
    )
    type: ValidatorTypeEnum = Field(
        ..., description="Which validator implementation to run."
    )
    stage: StageEnum = Field(
        ...,
        description=(
            "Which text this config is intended for. Advisory only at run "
            "time: POST /guardrails routes on the text it is given, not on "
            "this field."
        ),
    )
    on_fail_action: GuardrailOnFailEnum = Field(
        default=GuardrailOnFailEnum.FIX,
        description="Action taken when this validator fails.",
    )
    is_enabled: bool = Field(
        default=True, description="Whether this config is eligible to run."
    )


class ValidatorConfigCreate(ValidatorConfigBase):
    """Request body for ``POST /api/v1/guardrails/validators/configs``.

    Extra keys are allowed and are stored as the validator's tuning config —
    e.g. ``entity_types``/``threshold`` for ``pii_remover``, ``languages``/
    ``severity`` for ``uli_slur_match``. Call ``GET /api/v1/guardrails`` for
    the JSON schema of each validator type's accepted keys.

    ``organization_id``/``project_id`` must not be sent: the guardrails
    service derives the tenant from the authenticated context and rejects
    those keys in the body with a 422.
    """

    model_config = ConfigDict(extra="allow")


class ValidatorConfigUpdate(SQLModel):
    """Request body for ``PATCH /guardrails/validators/configs/{config_id}``.

    Only the base fields can be patched. The guardrails service forbids extra
    keys here, so validator-specific tuning cannot be changed through this
    route — delete and recreate the config instead.

    Extras are accepted and forwarded rather than rejected locally, so the
    upstream 422 is what the caller sees. Dropping them silently would be
    worse: the caller would believe a tuning change had been applied.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    type: ValidatorTypeEnum | None = None
    stage: StageEnum | None = None
    on_fail_action: GuardrailOnFailEnum | None = None
    is_enabled: bool | None = None


class ValidatorConfigPublic(ValidatorConfigBase):
    """A validator config as returned by the guardrails service.

    The service returns the stored row flattened together with its JSONB
    tuning config, so responses carry additional validator-specific keys
    beyond the ones declared here.
    """

    model_config = ConfigDict(extra="allow")

    id: UUID
    organization_id: int
    project_id: int
    created_at: datetime
    updated_at: datetime


class ValidatorTypePublic(SQLModel):
    """One entry of ``GET /api/v1/guardrails``."""

    type: str = Field(..., description="Validator type discriminator.")
    config: dict[str, Any] = Field(
        ...,
        description="JSON Schema of the tuning fields this validator accepts.",
    )


class ValidatorTypeListPublic(SQLModel):
    """Response body of ``GET /api/v1/guardrails``.

    This route is the one guardrails endpoint that is *not* wrapped in the
    standard ``APIResponse`` envelope; the service returns this bare object.
    """

    validators: list[ValidatorTypePublic]
