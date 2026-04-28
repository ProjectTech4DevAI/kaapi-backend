from sqlmodel import Session, select

from app.core.util import now
from app.models import FeatureFlag as FeatureFlagModel


def get_feature_flag(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int,
) -> FeatureFlagModel | None:
    statement = select(FeatureFlagModel).where(
        FeatureFlagModel.key == key,
        FeatureFlagModel.organization_id == organization_id,
        FeatureFlagModel.project_id == project_id,
    )
    return session.exec(statement).first()


def get_feature_flag_enabled(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int,
) -> bool | None:
    statement = select(FeatureFlagModel.enabled).where(
        FeatureFlagModel.key == key,
        FeatureFlagModel.organization_id == organization_id,
        FeatureFlagModel.project_id == project_id,
    )
    return session.exec(statement).first()


def create_feature_flag(
    *,
    session: Session,
    key: str,
    organization_id: int,
    project_id: int,
    enabled: bool,
) -> FeatureFlagModel | None:
    feature_flag = get_feature_flag(
        session=session,
        key=key,
        organization_id=organization_id,
        project_id=project_id,
    )
    if feature_flag is not None:
        return None

    feature_flag = FeatureFlagModel(
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
    project_id: int,
    enabled: bool,
) -> FeatureFlagModel | None:
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
    project_id: int,
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
) -> list[FeatureFlagModel]:
    statement = select(FeatureFlagModel)
    if key is not None:
        statement = statement.where(FeatureFlagModel.key == key)
    if organization_id is not None:
        statement = statement.where(FeatureFlagModel.organization_id == organization_id)
    if project_id is not None:
        statement = statement.where(FeatureFlagModel.project_id == project_id)
    statement = statement.order_by(FeatureFlagModel.key.asc())
    return list(session.exec(statement).all())
