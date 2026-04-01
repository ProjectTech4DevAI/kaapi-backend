import re
import secrets
from typing import Any

from sqlmodel import SQLModel, Field
from pydantic import EmailStr, model_validator, field_validator

from app.core.providers import validate_provider, validate_provider_credentials


class OnboardingRequest(SQLModel):
    """
    Request model for onboarding an organization, project, and user.

    Behavior:
    - **organization_name**: Required. If it does not exist, a new organization will be created.
    - **project_name**: Required. Must be unique within the organization.
    - **user_name**: Optional. If not provided, defaults to `<project_name> User`.
    - **email**: Optional. If not provided, an email will be auto-generated using
      a normalized username + random suffix (e.g., `project_user.ab12cd@kaapi.org`).
    - **password**: Optional. If not provided, a secure random password is generated.
    - **openai_api_key**: Optional. If provided, it will be encrypted and stored with the project.

    Notes:
    - Some users may not need a full user module and only want to interact using an API key.
      For those cases, user-related fields are optional and safe defaults are generated automatically.
    """

    organization_name: str = Field(
        description="Name of the organization to be created or linked",
        min_length=1,
        max_length=100,
    )
    project_name: str = Field(
        description="Name of the project under the organization",
        min_length=1,
        max_length=100,
    )
    email: EmailStr | None = Field(
        default=None,
        description="Email address of the primary user",
    )
    password: str | None = Field(
        default=None,
        description="Password for the primary user (must be at least 8 characters)",
        min_length=8,
        max_length=128,
    )
    user_name: str | None = Field(
        default=None,
        description="Full name of the primary user",
        min_length=3,
        max_length=50,
    )
    credentials: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional credential(s) to link with the project",
    )

    @staticmethod
    def _clean_username(raw: str, max_len: int = 200) -> str:
        """
        Normalize a string into a safe username that can also be used
        as the local part of an email address.
        """
        username = re.sub(r"[^A-Za-z0-9._]", "_", raw.strip().lower())
        username = re.sub(r"[._]{2,}", "_", username)  # collapse repeats
        username = username.strip("._")  # remove leading/trailing
        return username[:max_len]

    @model_validator(mode="after")
    def set_defaults(self):
        if self.user_name is None:
            self.user_name = self.project_name + " User"

        if self.email is None:
            local_part = self._clean_username(self.user_name, max_len=200)
            suffix = secrets.token_hex(3)
            self.email = f"{local_part}.{suffix}@kaapi.org"

        if self.password is None:
            self.password = secrets.token_urlsafe(12)
        return self

    @field_validator("credentials")
    @classmethod
    def _validate_credential_list(cls, v: list[dict[str, dict[str, str]]] | None):
        if v is None:
            return v

        if not isinstance(v, list):
            raise TypeError(
                "Credential must be a list of single-key dicts (e.g., {'openai': {...}})."
            )

        for item in v:
            if not isinstance(item, dict):
                raise TypeError(
                    "Credential must be a dict with a single provider key like {'openai': {...}}."
                )
            if len(item) != 1:
                raise ValueError(
                    "Credential must have exactly one provider key like {'openai': {...}}."
                )

            (provider_key,) = item.keys()
            values = item[provider_key]

            validate_provider(provider_key)

            if not isinstance(values, dict):
                raise ValueError(
                    f"Value for provider '{provider_key}' must be an object/dict."
                )

            validate_provider_credentials(provider_key, values)

        return v


class OnboardingResponse(SQLModel):
    """
    Response model for the Onboarding API.

    Contains the identifiers and credentials created or linked during onboarding.
    """

    organization_id: int = Field(description="Unique ID of the organization")
    organization_name: str = Field(description="Name of the organization")
    project_id: int = Field(
        description="Unique ID of the project within the organization"
    )
    project_name: str = Field(description="Name of the project")
    user_id: int = Field(
        description="Unique ID of the user.",
    )
    user_email: EmailStr = Field(
        description="Email of the user.",
    )
    api_key: str = Field(description="Generated internal API key for the project.")
