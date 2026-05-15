import logging
from uuid import UUID
from typing import Tuple

from sqlmodel import Session, select, and_
from fastapi import HTTPException

from app.models import APIKey, User
from app.crud.project import get_project_by_id
from app.core.util import now
from app.core.security import api_key_manager

logger = logging.getLogger(__name__)


class APIKeyCrud:
    """
    CRUD operations for API keys scoped to a project.
    """

    def __init__(self, session: Session, project_id: int):
        self.session = session
        self.project_id = project_id

    def read_one(self, key_id: UUID) -> APIKey | None:
        """
        Retrieve a single non-deleted API key by its id.
        """
        statement = select(APIKey).where(
            and_(
                APIKey.id == key_id,
                APIKey.project_id == self.project_id,
                APIKey.deleted_at.is_(None),
            )
        )
        return self.session.exec(statement).one_or_none()

    def read_all(self, skip: int = 0, limit: int = 100) -> list[APIKey]:
        """
        Read all non-deleted API keys for the project.
        """
        statement = (
            select(APIKey)
            .where(
                and_(
                    APIKey.project_id == self.project_id,
                    APIKey.deleted_at.is_(None),
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return self.session.exec(statement).all()

    def create(self, user_id: int, project_id: int) -> Tuple[str, APIKey]:
        """
        Create a new API key for the project.
        """
        project = get_project_by_id(session=self.session, project_id=project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        user = self.session.get(User, user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        try:
            raw_key, key_prefix, key_hash = api_key_manager.generate()

            api_key = APIKey(
                key_prefix=key_prefix,
                key_hash=key_hash,
                user_id=user_id,
                organization_id=project.organization_id,
                project_id=project_id,
            )

            self.session.add(api_key)
            self.session.commit()
            self.session.refresh(api_key)

            logger.info(
                f"[APIKeyCrud.create_api_key] API key created successfully | "
                f"{{'api_key_id': '{api_key.id}', 'project_id': {project_id}, 'user_id': {user_id}}}"
            )

            return raw_key, api_key

        except Exception as e:
            logger.error(
                f"[APIKeyCrud.create_api_key] Failed to create API key | "
                f"{{'project_id': {project_id}, 'user_id': {user_id}, 'error': '{str(e)}'}}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to create API key: {str(e)}"
            )

    def delete(self, key_id: UUID) -> None:
        """
        Soft delete an API key by marking it as deleted.
        """
        api_key = self.read_one(key_id)
        if not api_key:
            raise HTTPException(status_code=404, detail="API Key not found")

        api_key.deleted_at = now()
        api_key.updated_at = now()
        self.session.add(api_key)
        self.session.commit()
        self.session.refresh(api_key)

        logger.info(
            f"[APIKeyCrud.delete_api_key] API key deleted successfully | "
            f"{{'api_key_id': '{api_key.id}', 'project_id': {self.project_id}}}"
        )
