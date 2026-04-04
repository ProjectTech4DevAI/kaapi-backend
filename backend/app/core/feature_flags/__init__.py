from enum import Enum

from app.core.feature_flags.client import (
    get_client,
    init_unleash,
    shutdown_unleash,
)
from app.core.feature_flags.context import build_context


class FeatureFlag(str, Enum):
    ASSESSMENT = "Assessment"


def is_enabled(
    flag: FeatureFlag,
    organization_id: int,
    project_id: int | None = None,
    user_id: int | None = None,
    default: bool = False,
) -> bool:
    """Check whether *flag* is enabled for the given scope."""
    ctx = build_context(organization_id, project_id, user_id)

    def _fallback(_feature_name: str, _context: dict | None) -> bool:
        return default

    return get_client().is_enabled(flag.value, ctx, fallback_function=_fallback)


def resolve_all_flags(
    organization_id: int,
    project_id: int | None = None,
    user_id: int | None = None,
) -> dict[str, bool]:
    """Evaluate every registered flag for the given scope."""
    return {
        flag.name: is_enabled(
            flag,
            organization_id=organization_id,
            project_id=project_id,
            user_id=user_id,
        )
        for flag in FeatureFlag
    }


__all__ = [
    "FeatureFlag",
    "init_unleash",
    "is_enabled",
    "resolve_all_flags",
    "shutdown_unleash",
]
