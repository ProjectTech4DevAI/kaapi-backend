from sqlmodel import Session, select

from app.core.util import now
from app.models import FeatureFlag


def get_feature_flag(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int | None,
) -> FeatureFlag | None:
    statement = select(FeatureFlag).where(
        FeatureFlag.key == key,
        FeatureFlag.organization_id == organization_id,
        FeatureFlag.project_id == project_id,
    )
    return session.exec(statement).first()


def get_feature_flag_enabled(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int | None,
) -> bool | None:
    statement = select(FeatureFlag.enabled).where(
        FeatureFlag.key == key,
        FeatureFlag.organization_id == organization_id,
        FeatureFlag.project_id == project_id,
    )
    return session.exec(statement).first()


def create_feature_flag(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int | None,
    enabled: bool,
) -> FeatureFlag | None:
    feature_flag = get_feature_flag(
        session=session,
        key=key,
        organization_id=organization_id,
        project_id=project_id,
    )
    if feature_flag is not None:
        return None

    feature_flag = FeatureFlag(
        key=key,
        organization_id=organization_id,
        project_id=project_id,
        enabled=enabled,
    )
    session.add(feature_flag)
    session.commit()
    session.refresh(feature_flag)
    return feature_flag


def update_feature_flag(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int | None,
    enabled: bool,
) -> FeatureFlag | None:
    feature_flag = get_feature_flag(
        session=session,
        key=key,
        organization_id=organization_id,
        project_id=project_id,
    )
    if feature_flag is None:
        return None

    feature_flag.enabled = enabled
    feature_flag.updated_at = now()
    session.add(feature_flag)
    session.commit()
    session.refresh(feature_flag)
    return feature_flag


def delete_feature_flag(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int | None,
) -> bool:
    feature_flag = get_feature_flag(
        session=session,
        key=key,
        organization_id=organization_id,
        project_id=project_id,
    )
    if feature_flag is None:
        return False

    session.delete(feature_flag)
    session.commit()
    return True


def list_feature_flags(
    *,
    session: Session,
    key: str | None = None,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> list[FeatureFlag]:
    statement = select(FeatureFlag)
    if key is not None:
        statement = statement.where(FeatureFlag.key == key)
    if organization_id is not None:
        statement = statement.where(FeatureFlag.organization_id == organization_id)
    if project_id is not None:
        statement = statement.where(FeatureFlag.project_id == project_id)
    statement = statement.order_by(FeatureFlag.key.asc())
    return list(session.exec(statement).all())
