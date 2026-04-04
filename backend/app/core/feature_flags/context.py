"""Build Unleash evaluation context from auth information."""


def build_context(
    organization_id: int,
    project_id: int | None = None,
    user_id: int | None = None,
) -> dict[str, str]:
    """Build an Unleash context dict from auth dimensions.

    Unleash strategies use these fields for targeting:
    - organizationId  → gate at org level
    - projectId       → drill down to org + project
    - userId          → drill down to individual user (future)
    """
    ctx: dict[str, str] = {"organizationId": str(organization_id)}
    if project_id is not None:
        ctx["projectId"] = str(project_id)
    if user_id is not None:
        ctx["userId"] = str(user_id)
    return ctx
