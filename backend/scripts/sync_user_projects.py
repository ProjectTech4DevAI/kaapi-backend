"""
One-time script to sync user-project mappings from the APIKey table
into the UserProject table.

Usage:
    cd backend && uv run python scripts/sync_user_projects.py
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.load_env import load_environment

load_environment()

from sqlmodel import Session, select, and_
from app.core.db import engine
from app.models import APIKey, UserProject

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def sync_user_projects() -> None:
    with Session(engine) as session:
        # Get distinct user/org/project combos from active API keys
        statement = (
            select(APIKey.user_id, APIKey.organization_id, APIKey.project_id)
            .where(APIKey.is_deleted.is_(False))
            .distinct()
        )
        api_key_mappings = session.exec(statement).all()

        added = 0
        skipped = 0

        for user_id, org_id, project_id in api_key_mappings:
            # Check if mapping already exists in UserProject
            existing = session.exec(
                select(UserProject).where(
                    and_(
                        UserProject.user_id == user_id,
                        UserProject.project_id == project_id,
                    )
                )
            ).first()

            if existing:
                skipped += 1
                logger.info(
                    f"  SKIP: user_id={user_id}, project_id={project_id} (already exists)"
                )
                continue

            user_project = UserProject(
                user_id=user_id,
                organization_id=org_id,
                project_id=project_id,
            )
            session.add(user_project)
            added += 1
            logger.info(
                f"  ADD:  user_id={user_id}, org_id={org_id}, project_id={project_id}"
            )

        session.commit()
        logger.info(f"\nSync complete: {added} added, {skipped} skipped")


if __name__ == "__main__":
    logger.info("Syncing user-project mappings from APIKey table...\n")
    sync_user_projects()
