from sqlmodel import Session

from app.core.feature_flags.constants import FeatureFlag
from app.crud.feature_flag import get_feature_flag_enabled, list_feature_flags


def parse_feature_flag(flag_key: str) -> FeatureFlag:
    """Parse a feature flag from its exact key."""
    try:
        return FeatureFlag(flag_key)
    except ValueError as exc:
        raise ValueError(f"Unknown feature flag: {flag_key}") from exc


def is_enabled(
    session: Session,
    flag: FeatureFlag,
    organization_id: int,
    project_id: int | None = None,
    user_id: int | None = None,
) -> bool:
    """Check whether *flag* is enabled for the given scope."""
    _ = (organization_id, project_id, user_id)
    resolved_flag = get_feature_flag_enabled(
        session=session,
        key=flag.value,
        organization_id=organization_id,
        project_id=project_id,
    )
    return bool(resolved_flag) if resolved_flag is not None else False


def resolve_all_flags(
    session: Session,
    organization_id: int,
    project_id: int | None = None,
    user_id: int | None = None,
) -> dict[str, bool]:
    """Return persisted known flags for the given org/project scope."""
    _ = user_id
    flags = list_feature_flags(
        session=session,
        organization_id=organization_id,
        project_id=project_id,
    )
    resolved: dict[str, bool] = {}
    for flag_row in flags:
        try:
            parse_feature_flag(flag_row.key)
        except ValueError:
            continue
        resolved[flag_row.key] = flag_row.enabled
    return resolved


__all__ = [
    "FeatureFlag",
    "is_enabled",
    "parse_feature_flag",
    "resolve_all_flags",
]
