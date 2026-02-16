import logging
from typing import Optional
from uuid import UUID

from sqlmodel import Session, select, and_, func
from fastapi import HTTPException
from sqlalchemy.orm import defer

from .config import ConfigCrud
from app.core.util import now
from app.models import Config, ConfigVersion, ConfigVersionCreate, ConfigVersionItems

logger = logging.getLogger(__name__)


class ConfigVersionCrud:
    """
    CRUD operations for configuration versions scoped to a project.
    """

    def __init__(
        self,
        session: Session,
        config_id: UUID,
        project_id: int,
        organization_id: Optional[int] = None,
    ):
        self.session = session
        self.project_id = project_id
        self.config_id = config_id
        self.organization_id = organization_id

    def create_or_raise(self, version_create: ConfigVersionCreate) -> ConfigVersion:
        """
        Create a new version for an existing configuration.
        Automatically increments the version number.
        """
        self._config_exists_or_raise(self.config_id)
        try:
            next_version = self._get_next_version(self.config_id)

            version = ConfigVersion(
                config_id=self.config_id,
                version=next_version,
                config_blob=version_create.config_blob.model_dump(),
                commit_message=version_create.commit_message,
            )

            self.session.add(version)
            self.session.commit()
            self.session.refresh(version)

            logger.info(
                f"[ConfigVersionCrud.create] Version created successfully | "
                f"{{'config_id': '{self.config_id}', 'version_id': '{version.id}'}}"
            )

            return version

        except Exception as e:
            self.session.rollback()
            logger.error(
                f"[ConfigVersionCrud.create] Failed to create version | "
                f"{{'config_id': '{self.config_id}', 'error': '{str(e)}'}}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Unexpected error occurred: failed to create version",
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
