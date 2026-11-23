"""
Per-organization request rate limiting implemented as a FastAPI dependency.
Uses Redis to track request counts per organization and the `limits` package
to enforce rate limits.

Reusable across endpoints by adding it to the dependencies:
```
Depends(enforce_rate_limit("evaluations"))
```
"""

import logging
import math
import time
from collections.abc import Callable
from typing import Literal

from fastapi import HTTPException
from limits import RateLimitItem, RateLimitItemPerMinute
from limits.errors import StorageError
from limits.storage import RedisStorage
from limits.strategies import (
    MovingWindowRateLimiter,
)  # using MovingWindowRateLimiter for more accurate rate limiting

from app.api.deps import AuthContextDep
from app.core.config import settings

logger = logging.getLogger(__name__)

# Extendable for other categories in the future
RateLimitCategory = Literal["evaluations"]

RATE_LIMITS: dict[RateLimitCategory, RateLimitItem] = {
    "evaluations": RateLimitItemPerMinute(settings.EVAL_RATE_LIMIT_PER_MINUTE),
}

_limiter = MovingWindowRateLimiter(
    RedisStorage(
        settings.REDIS_URL,
        key_prefix="rate_limit",
        wrap_exceptions=True,
        socket_timeout=1,
        socket_connect_timeout=1,
    )
)


def enforce_rate_limit(category: RateLimitCategory) -> Callable[[AuthContextDep], None]:
    """Reject with 429 once the caller's organization exceeds the category's per-minute limit.

    Fails open when Redis is unreachable so an outage never blocks requests.
    """
    rate_limit_item = RATE_LIMITS[category]

    def _enforce(auth_context: AuthContextDep) -> None:
        org_id = auth_context.organization_.id
        key = f"{category}:{org_id}"

        try:
            if _limiter.hit(rate_limit_item, key):
                return
            reset_time = _limiter.get_window_stats(rate_limit_item, key).reset_time
        except StorageError as exc:
            logger.error(
                f"[enforce_rate_limit] Redis unavailable, allowing request | "
                f"category: {category} | org_id: {org_id} | error: {exc}"
            )
            return

        retry_after = max(1, math.ceil(reset_time - time.time()))
        project_id = auth_context.project.id if auth_context.project else None
        logger.warning(
            f"[enforce_rate_limit] Rate limit exceeded | category: {category} | "
            f"org_id: {org_id} | project_id: {project_id} | retry_after: {retry_after}s"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: at most {rate_limit_item.amount} {category} "
                f"requests per minute per organization. Try again in {retry_after} seconds."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    return _enforce
