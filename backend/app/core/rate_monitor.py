import logging
import time

from collections.abc import Callable
from typing import Literal

import redis

from app.api.deps import AuthContextDep
from app.core.config import settings

from app.core.telemetry import record_rate_threshold

logger = logging.getLogger(__name__)

# Categories of rates we want to monitor
RateCategory = Literal["llm_call", "collections", "evaluations"]

# THRESHOLD NUMBERS
THRESHOLDS: dict[RateCategory, int] = {
    "llm_call": settings.THRESHOLD_LLM_CALL_RATE,
    "collections": settings.THRESHOLD_COLLECTIONS_RATE,
    "evaluations": settings.THRESHOLD_EVALUATIONS_RATE,
}

# Delete record after 2 minutes from redis
_EXPIRATION_SECONDS = 120

_redis_client: redis.Redis = redis.from_url(settings.REDIS_URL, decode_responses=True)


# count incrementor after each request and get count
def increment_and_get_count(key: str) -> int | None:
    """Increment the count for the given key and return the new count.
    The count will automatically expire after _EXPIRATION_SECONDS.
    """
    try:
        pipe = _redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, _EXPIRATION_SECONDS)
        count, _ = pipe.execute()
        return count
    except Exception as e:
        logger.error(
            f"[increment_and_get_count] Error incrementing count for {key}: {e}"
        )
        return None


def monitor_rate(category: RateCategory) -> Callable[[AuthContextDep], None]:
    """Monitor the rate of events for the given category. If the rate exceeds the threshold, record it in telemetry.

    Usage:
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("{category}")),
    ]
    """

    def _checker(auth_context: AuthContextDep) -> None:
        project = auth_context.project
        if project is None:
            return

        threshold = THRESHOLDS.get(category, None)
        if threshold is None:
            logger.warning(
                f"[monitor_rate] No threshold defined for category {category}"
            )
            return

        minute_bucket = int(time.time() // 60)
        redis_key = f"rate_monitor:{category}:{project.id}:{minute_bucket}"

        count = increment_and_get_count(redis_key)
        if count is not None and count == threshold + 1:
            logger.warning(
                f"[monitor_rate] Rate threshold exceeded for {category} in project {project.id}: count={count}"
            )
            record_rate_threshold(
                project_id=project.id,
                project_name=project.name,
                category=category,
                request_count=count,
                threshold=threshold,
            )

    return _checker
