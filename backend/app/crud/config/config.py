import logging
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session, and_, select

from app.core.util import now
from app.crud.model_config import validate_blob_model_or_raise
from app.models import (
    Config,
    ConfigCreate,
    ConfigUpdate,
    ConfigVersion,
)
from app.models.config.config import ConfigTag

logger = logging.getLogger(__name__)


def get_config_by_id(
    *,
    session: Session,
    config_id: UUID,
    project_id: int,
) -> Config | None:
    """Return a non-deleted Config scoped to project_id, or None if absent/soft-deleted."""
    statement = select(Config).where(
        and_(
            Config.id == config_id,
            Config.project_id == project_id,
            Config.deleted_at.is_(None),
        )
    )
    config = session.exec(statement).one_or_none()
    if not config:
        logger.warning(
            f"[get_config_by_id] Not found or soft-deleted | "
            f"config_id={config_id} project_id={project_id}"
        )
    return config


class ConfigCrud:
    """
    CRUD operations for configurations scoped to a project.
    """

    def __init__(self, session: Session, project_id: int):
        self.session = session
        self.project_id = project_id

    def create_or_raise(
        self, config_create: ConfigCreate
    ) -> tuple[Config, ConfigVersion]:
        """
        Create a new configuration with an initial version.
        """
        self._check_unique_name_or_raise(config_create.name)

        validate_blob_model_or_raise(self.session, config_create.config_blob)

        try:
            config = Config(
                name=config_create.name,
                description=config_create.description,
                project_id=self.project_id,
                tag=config_create.tag,
            )

            self.session.add(config)
            self.session.flush()  # Flush to get the config.id

            # Create the initial version
            version = ConfigVersion(
                config_id=config.id,
                version=1,
                config_blob=config_create.config_blob.model_dump(mode="json"),
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
                detail="Unexpected error occurred: failed to create config",
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

    def read_all(
        self,
        query: str | None,
        skip: int = 0,
        limit: int = 100,
        tag: ConfigTag = ConfigTag.DEFAULT,
    ) -> tuple[list[Config], bool]:
        filters = [
            Config.project_id == self.project_id,
            Config.deleted_at.is_(None),
        ]

        if query:
            filters.append(Config.name.ilike(f"{query}%"))

        filters.append(self._tag_scope_filter(tag))

        statement = (
            select(Config)
            .where(and_(*filters))
            .order_by(Config.updated_at.desc())
            .offset(skip)
            .limit(limit + 1)
        )
        configs = self.session.exec(statement).all()
        has_more = False
        if limit is not None and len(configs) > limit:
            has_more = True
            configs = configs[:limit]

        return configs, has_more

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

    def exists_in_tag_scope_or_raise(
        self, config_id: UUID, tag: ConfigTag = ConfigTag.DEFAULT
    ) -> Config:
        statement = select(Config).where(
            and_(
                Config.id == config_id,
                Config.project_id == self.project_id,
                Config.deleted_at.is_(None),
                self._tag_scope_filter(tag),
            )
        )
        config = self.session.exec(statement).one_or_none()
        if config is None:
            raise HTTPException(
                status_code=404,
                detail=f"config with id '{config_id}' not found",
            )

        return config

    def _tag_scope_filter(self, tag: ConfigTag):
        return Config.tag == tag

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
