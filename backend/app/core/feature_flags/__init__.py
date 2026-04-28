from sqlmodel import Session

from app.core.feature_flags.constants import FeatureFlag


def is_enabled(
    session: Session,
    flag: str,
    organization_id: int,
    project_id: int,
) -> bool:
    """Check whether *flag* is enabled for the given org + project scope."""
    from app.crud.feature_flag import get_feature_flag_enabled

    result = get_feature_flag_enabled(
        session=session,
        key=flag,
        organization_id=organization_id,
        project_id=project_id,
    )
    return bool(result) if result is not None else False


def resolve_all_flags(
    session: Session,
    organization_id: int,
    project_id: int,
) -> dict[str, bool]:
    """Return all flags resolved for the given org + project scope."""
    from app.crud.feature_flag import list_feature_flags

    flags = list_feature_flags(
        session=session,
        organization_id=organization_id,
        project_id=project_id,
    )
    return {flag_row.key: flag_row.enabled for flag_row in flags}


__all__ = [
    "FeatureFlag",
    "is_enabled",
    "resolve_all_flags",
]
