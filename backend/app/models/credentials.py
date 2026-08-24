from datetime import datetime

import sqlalchemy as sa
from pydantic import field_validator, model_validator
from sqlmodel import Field, Relationship, SQLModel

from app.core.providers import (
    CredentialPayload,
    Provider,
    ProviderCredentials,
    mask_credential_fields,
    parse_provider_credentials,
    validate_provider,
)
from app.core.util import now
from app.models.organization import Organization
from app.models.project import Project


class CredsBase(SQLModel):
    """Base model for credentials with foreign keys and common fields."""

    is_active: bool = Field(
        default=True,
        nullable=False,
        sa_column_kwargs={
            "comment": "Flag indicating if this credential is currently active and usable"
        },
    )

    # Foreign keys
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )


class CredsCreate(SQLModel):
    """Create new credentials for an organization.
    The credential field maps each provider to its own credential payload.
    Example: {"openai": {"api_key": "..."}, "langfuse": {"public_key": "..."}}
    """

    is_active: bool = True
    credential: dict[Provider, ProviderCredentials] | None = Field(
        default=None,
        description="Credential payload per provider, keyed by provider name",
    )

    @field_validator("credential", mode="before")
    @classmethod
    def _parse_credential(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        return {
            validate_provider(provider).value: parse_provider_credentials(
                provider, payload
            )
            for provider, payload in value.items()
        }

    def credential_payloads(self) -> dict[str, CredentialPayload]:
        """Provider name -> credential dict, exactly as submitted."""
        return {
            provider.value: payload.model_dump(exclude_unset=True)
            for provider, payload in (self.credential or {}).items()
        }


class CredsUpdate(SQLModel):
    """Update credentials for an organization.
    Can update a specific provider's credentials or add a new provider.
    """

    provider: Provider = Field(
        description="Name of the provider to update/add credentials for"
    )
    credential: ProviderCredentials = Field(
        description="Credentials for the specified provider",
    )
    is_active: bool | None = Field(
        default=None, description="Whether the credentials are active"
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_credential(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        provider = data.get("provider")
        credential = data.get("credential")
        if not isinstance(provider, str) or credential is None:
            return data

        provider_key = validate_provider(provider).value
        nested = credential.get(provider_key) if isinstance(credential, dict) else None
        if isinstance(nested, dict):
            credential = nested

        return {
            **data,
            "provider": provider_key,
            "credential": parse_provider_credentials(provider_key, credential),
        }

    def credential_payload(self) -> CredentialPayload:
        """Credential dict for `provider`, exactly as submitted."""
        return self.credential.model_dump(exclude_unset=True)


class Credential(CredsBase, table=True):
    """Database model for storing provider credentials.
    Each row represents credentials for a single provider.
    """

    __table_args__ = (
        sa.UniqueConstraint(
            "organization_id",
            "project_id",
            "provider",
            name="uq_credential_org_project_provider",
        ),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique ID for the credential"},
    )
    provider: str = Field(
        index=True,
        nullable=False,
        description="Provider name like 'openai', 'google'",
        sa_column_kwargs={"comment": "Provider name like 'openai', 'google'"},
    )
    credential: str = Field(
        nullable=False,
        description="Encrypted JSON string containing provider-specific API credentials",
        sa_column_kwargs={
            "comment": "Encrypted JSON string containing provider-specific API credentials"
        },
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the credential was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the credential was last updated"},
    )

    # Relationships
    organization: Organization | None = Relationship(back_populates="creds")
    project: Project | None = Relationship(back_populates="creds")

    def to_public(self, mask: bool = True) -> "CredsPublic":
        """Convert the database model to a public model with decrypted credentials.

        By default, sensitive fields (e.g., api_key, secret_key) are masked so
        the response is safe to return via the API.
        """
        # Local import to avoid circular dependency (security imports app.models)
        from app.core.security import decrypt_credentials

        decrypted = decrypt_credentials(self.credential) if self.credential else None
        if mask and decrypted:
            decrypted = mask_credential_fields(self.provider, decrypted)

        return CredsPublic(
            id=self.id,
            organization_id=self.organization_id,
            project_id=self.project_id,
            is_active=self.is_active,
            provider=Provider(self.provider),
            credential=decrypted,
            inserted_at=self.inserted_at,
            updated_at=self.updated_at,
        )


class CredsPublic(CredsBase):
    """Public representation of credentials, excluding sensitive information."""

    id: int
    provider: Provider
    credential: CredentialPayload | None = Field(
        default=None,
        description="Provider credential with its sensitive fields masked",
    )
    inserted_at: datetime
    updated_at: datetime
