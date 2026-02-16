import logging
from uuid import UUID
from typing import Tuple

from sqlmodel import Session, select, and_
from fastapi import HTTPException

from app.models import (
    Config,
    ConfigCreate,
    ConfigUpdate,
    ConfigVersion,
)
from app.core.util import now

logger = logging.getLogger(__name__)


class ConfigCrud:
    """
    CRUD operations for configurations scoped to a project.
    """

    def __init__(self, session: Session, project_id: int):
        self.session = session
        self.project_id = project_id

    def create_or_raise(
        self, config_create: ConfigCreate
    ) -> Tuple[Config, ConfigVersion]:
        """
        Create a new configuration with an initial version.
        """
        self._check_unique_name_or_raise(config_create.name)

        try:
            config = Config(
                name=config_create.name,
                description=config_create.description,
                project_id=self.project_id,
            )

            self.session.add(config)
            self.session.flush()  # Flush to get the config.id

            # Create the initial version
            version = ConfigVersion(
                config_id=config.id,
                version=1,
                config_blob=config_create.config_blob.model_dump(),
                commit_message=config_create.commit_message,
            )

            self.session.add(version)
            self.session.commit()
            self.session.refresh(config)
            self.session.refresh(version)

            logger.info(
                f"[ConfigCrud.create] Configuration created successfully | "
                f"{{'config_id': '{config.id}', 'config_version_id': '{version.id}', 'project_id': {self.project_id}}}"
            )

            return config, version

        except Exception as e:
            self.session.rollback()
            logger.error(
                f"[ConfigCrud.create] Failed to create configuration | "
                f"{{'name': '{config_create.name}', 'project_id': {self.project_id}, 'error': '{str(e)}'}}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error occurred: failed to create config",
            )

    def read_one(self, config_id: UUID) -> Config | None:
        statement = select(Config).where(
            and_(
                Config.id == config_id,
                Config.project_id == self.project_id,
                Config.deleted_at.is_(None),
            )
        )
        return self.session.exec(statement).one_or_none()

    def read_all(self, skip: int = 0, limit: int = 100) -> list[Config]:
        statement = (
            select(Config)
            .where(
                and_(
                    Config.project_id == self.project_id,
                    Config.deleted_at.is_(None),
                )
            )
            .order_by(Config.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return self.session.exec(statement).all()

    def update_or_raise(self, config_id: UUID, config_update: ConfigUpdate) -> Config:
        config = self.exists_or_raise(config_id)

        config_update = config_update.model_dump(exclude_none=True)

        if config_update.get("name") and config_update["name"] != config.name:
            self._check_unique_name_or_raise(config_update["name"])

        for key, value in config_update.items():
            setattr(config, key, value)

        config.updated_at = now()

        self.session.add(config)
        self.session.commit()
        self.session.refresh(config)

        logger.info(
            f"[ConfigCrud.update] Config updated successfully | "
            f"{{'config_id': '{config.id}', 'project_id': {self.project_id}}}"
        )
        return config

    def delete_or_raise(self, config_id: UUID) -> None:
        config = self.exists_or_raise(config_id)

        config.deleted_at = now()
        self.session.add(config)
        self.session.commit()
        self.session.refresh(config)

    def exists_or_raise(self, config_id: UUID) -> Config:
        config = self.read_one(config_id)
        if config is None:
            raise HTTPException(
                status_code=404,
                detail=f"config with id '{config_id}' not found",
            )

        return config

    def _check_unique_name_or_raise(self, name: str) -> None:
        if self._read_by_name(name):
            raise HTTPException(
                status_code=409,
                detail=f"Config with name '{name}' already exists in this project",
            )

    def _read_by_name(self, name: str) -> Config | None:
        statement = select(Config).where(
            and_(
                Config.name == name,
                Config.project_id == self.project_id,
                Config.deleted_at.is_(None),
            )
        )
        return self.session.exec(statement).one_or_none()
