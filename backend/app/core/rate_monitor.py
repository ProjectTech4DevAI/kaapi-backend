import logging
import time

from typing import Literal

import redis

from app.api.deps import AuthContextDep
from app.core.config import settings

from app.core.telemetry import record_rate_threshold

logger = logging.getLogger(__name__)

# Categores of rates we want to monitor
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


def monitor_rate(category: RateCategory):
    """Monitor the rate of events for the given category. If the rate exceeds the threshold, record it in telemetry.

    Usage:
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("llm")),
    ]
    """

    def _checker(auth_context: AuthContextDep) -> None:
        org = auth_context.organization
        if org is None:
            return

        threshold = THRESHOLDS.get(category, None)
        if threshold is None:
            logger.warning(
                f"[monitor_rate] No threshold defined for category {category}"
            )
            return

        minute_bucket = int(time.time() // 60)
        redis_key = f"rate_monitor:{category}:{org.id}:{minute_bucket}"

        try:
            count = increment_and_get_count(redis_key)
            if count is not None and count > threshold:
                logger.warning(
                    f"[monitor_rate] Rate threshold exceeded for {category} in org {org.id}: count={count}"
                )
                record_rate_threshold(
                    org_id=org.id,
                    org_name=org.name,
                    category=category,
                    request_count=count,
                    threshold=threshold,
                )
        except redis.RedisError as e:
            logger.error(
                "[monitor_rate] Redis unavailable, skipping rate check "
                "(org_id=%s category=%s)",
                org.id,
                category,
                exc_info=e,
            )

    return _checker
