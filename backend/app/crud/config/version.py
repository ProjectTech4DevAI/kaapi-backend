import logging
from uuid import UUID
from typing import Any

from sqlmodel import Session, select, and_, func
from fastapi import HTTPException
from sqlalchemy.orm import defer
from pydantic import ValidationError

from .config import ConfigCrud
from app.core.util import now
from app.models import (
    Config,
    ConfigVersion,
    ConfigVersionCreate,
    ConfigVersionUpdate,
    ConfigVersionItems,
)
from app.models.llm.request import ConfigBlob

logger = logging.getLogger(__name__)


class ConfigVersionCrud:
    """
    CRUD operations for configuration versions scoped to a project.
    """

    def __init__(self, session: Session, config_id: UUID, project_id: int):
        self.session = session
        self.project_id = project_id
        self.config_id = config_id

    def create_or_raise(self, version_create: ConfigVersionUpdate) -> ConfigVersion:
        """
        Create a new version from a partial config update.

        Fetches the latest version, merges the partial config with it,
        validates the result, and creates the new version.

        Fields 'type' is inherited from the existing config
        and cannot be changed.
        """
        self._config_exists_or_raise(self.config_id)

        # Get the latest version (required for partial updates)
        latest_version = self._get_latest_version()
        if latest_version is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot create partial version: no existing version found. Use full config for initial version.",
            )

        # Merge partial config with existing config
        merged_config = self._deep_merge(
            base=latest_version.config_blob,
            updates=version_create.config_blob,
        )

        # Validate that provider and type haven't been changed
        self._validate_immutable_fields(latest_version.config_blob, merged_config)

        # Validate the merged config as ConfigBlob
        try:
            validated_blob = ConfigBlob.model_validate(merged_config)
        except ValidationError as e:
            logger.error(
                f"[ConfigVersionCrud.create_from_partial] Validation failed | "
                f"{{'config_id': '{self.config_id}', 'error': '{str(e)}'}}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid config after merge: {str(e)}",
            )

        try:
            next_version = self._get_next_version(self.config_id)

            version = ConfigVersion(
                config_id=self.config_id,
                version=next_version,
                config_blob=validated_blob.model_dump(mode="json"),
                commit_message=version_create.commit_message,
            )

            self.session.add(version)
            self.session.commit()
            self.session.refresh(version)

            logger.info(
                f"[ConfigVersionCrud.create_from_partial] Version created successfully | "
                f"{{'config_id': '{self.config_id}', 'version_id': '{version.id}'}}"
            )

            return version

        except Exception as e:
            self.session.rollback()
            logger.error(
                f"[ConfigVersionCrud.create_from_partial] Failed to create version | "
                f"{{'config_id': '{self.config_id}', 'error': '{str(e)}'}}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Unexpected error occurred: failed to create version",
            )

    def _get_latest_version(self) -> ConfigVersion | None:
        """Get the latest version for the config."""
        stmt = (
            select(ConfigVersion)
            .where(
                and_(
                    ConfigVersion.config_id == self.config_id,
                    ConfigVersion.deleted_at.is_(None),
                )
            )
            .order_by(ConfigVersion.version.desc())
            .limit(1)
        )
        return self.session.exec(stmt).first()

    def _deep_merge(
        self, base: dict[str, Any], updates: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Deep merge two dictionaries.
        Values from 'updates' override values in 'base'.
        Nested dicts are merged recursively.
        """
        result = base.copy()

        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_immutable_fields(
        self, existing: dict[str, Any], merged: dict[str, Any]
    ) -> None:
        """
        Validate that immutable fields (type) haven't been changed.
        Provider and model can change between versions.
        """
        existing_completion = existing.get("completion", {})
        merged_completion = merged.get("completion", {})

        existing_type = existing_completion.get("type")
        merged_type = merged_completion.get("type")

        # Legacy configs predate the 'type' field; all were text-only
        if existing_type is None:
            existing_type = "text"

        if existing_type != merged_type:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot change config type from '{existing_type}' to '{merged_type}'. Type is immutable.",
            )

    def read_one(self, version_number: int) -> ConfigVersion | None:
        """
        Read a specific configuration version by its version number.
        """
        self._config_exists_or_raise(self.config_id)
        statement = select(ConfigVersion).where(
            and_(
                ConfigVersion.version == version_number,
                ConfigVersion.config_id == self.config_id,
                ConfigVersion.deleted_at.is_(None),
            )
        )
        return self.session.exec(statement).one_or_none()

    def read_all(self, skip: int = 0, limit: int = 100) -> list[ConfigVersionItems]:
        """
        Read all versions for a specific configuration with pagination.
        """
        self._config_exists_or_raise(self.config_id)

        statement = (
            select(ConfigVersion)
            .where(
                and_(
                    ConfigVersion.config_id == self.config_id,
                    ConfigVersion.deleted_at.is_(None),
                )
            )
            .options(
                defer(ConfigVersion.config_blob),
            )
            .order_by(ConfigVersion.version.desc())
            .offset(skip)
            .limit(limit)
        )
        results = self.session.exec(statement).all()
        return [ConfigVersionItems.model_validate(item) for item in results]

    def delete_or_raise(self, version_number: int) -> None:
        """
        Soft delete a configuration version by setting its deleted_at timestamp.
        """
        version = self.exists_or_raise(version_number)

        version.deleted_at = now()
        self.session.add(version)
        self.session.commit()
        self.session.refresh(version)

    def exists_or_raise(self, version_number: int) -> ConfigVersion:
        """
        Check if a configuration version exists; raise 404 if not found.
        """
        version = self.read_one(version_number=version_number)
        if version is None:
            raise HTTPException(
                status_code=404,
                detail=f"Version with number '{version_number}' not found for config '{self.config_id}'",
            )
        return version

    def _get_next_version(self, config_id: UUID) -> int | None:
        """Get the next version number for a config."""
        stmt = (
            select(ConfigVersion.version)
            .where(ConfigVersion.config_id == config_id)
            .order_by(ConfigVersion.version.desc())
            .limit(1)
        )
        latest = self.session.exec(stmt).first()
        if latest is None:
            return 1

        return latest + 1

    def _config_exists_or_raise(self, config_id: UUID) -> Config:
        """Check if a config exists in the project."""
        config_crud = ConfigCrud(session=self.session, project_id=self.project_id)
        config_crud.exists_or_raise(config_id)

    def _validate_config_type_unchanged(
        self, version_create: ConfigVersionCreate
    ) -> None:
        """
        Validate that the config type (text/stt/tts) in the new version matches
        the type from the latest existing version.
        Raises HTTPException if types don't match.
        """
        # Get the latest version
        stmt = (
            select(ConfigVersion)
            .where(
                and_(
                    ConfigVersion.config_id == self.config_id,
                    ConfigVersion.deleted_at.is_(None),
                )
            )
            .order_by(ConfigVersion.version.desc())
            .limit(1)
        )
        latest_version = self.session.exec(stmt).first()

        # If this is the first version, no validation needed
        if latest_version is None:
            return

        # Extract types from config blobs
        old_type = latest_version.config_blob.get("completion", {}).get("type")
        new_type = (
            version_create.config_blob.model_dump().get("completion", {}).get("type")
        )

        # Legacy configs predate the 'type' field; all were text-only
        if old_type is None:
            old_type = "text"

        if new_type is None:
            logger.error(
                f"[ConfigVersionCrud._validate_config_type_unchanged] Missing type field | "
                f"{{'config_id': '{self.config_id}', 'old_type': {old_type}, 'new_type': {new_type}}}"
            )
            raise HTTPException(
                status_code=400,
                detail="Config type field is missing in configuration blob",
            )

        if old_type != new_type:
            logger.warning(
                f"[ConfigVersionCrud._validate_config_type_unchanged] Type mismatch | "
                f"{{'config_id': '{self.config_id}', 'old_type': '{old_type}', 'new_type': '{new_type}'}}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Cannot change config type from '{old_type}' to '{new_type}'. Config type must remain consistent across versions.",
            )
