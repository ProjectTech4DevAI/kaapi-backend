"""Redis counting semaphore that caps concurrent fast-eval chunk tasks.

Issue #1019: fast-eval chunk tasks (priority 6) and interactive LLM jobs
(priority 9) share the single `default` queue. RabbitMQ priority orders the
*queue*, but never preempts a task that is already running -- so a burst of
chunk tasks can fill every worker slot and stall LLM jobs behind a 300s chunk.
This semaphore caps how many chunks execute at once so LLM jobs keep headroom;
excess chunks re-queue themselves.

Crash-safe: permits are ZSET members scored by expiry epoch seconds, so a
worker that dies mid-chunk without releasing has its permit reclaimed once the
score passes. Chunks are idempotent, so a re-run after expiry is safe.
"""

import logging
import time

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis_client: redis.Redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

_SLOTS_KEY = "eval:fast:chunk:slots"

# Backstop TTL for a permit whose holder dies without releasing; must outlive
# the hard task limit. Chunks are idempotent, so reclaiming + re-running is safe.
_PERMIT_TTL_SECONDS = settings.CELERY_TASK_TIME_LIMIT + 60

# Evict expired permits, then grant one only if we're under the cap -- all in a
# single round-trip so the check-and-add can't race across workers.
_ACQUIRE_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[2]) then
    redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
    return 1
end
return 0
"""


def acquire_slot(token: str) -> bool:
    """Try to claim a concurrency permit for a chunk task, keyed by `token`."""
    now = time.time()
    try:
        granted = _redis_client.eval(
            _ACQUIRE_LUA,
            1,
            _SLOTS_KEY,
            now,
            settings.EVAL_FAST_MAX_CONCURRENCY,
            now + _PERMIT_TTL_SECONDS,
            token,
        )
        return bool(granted)
    except redis.RedisError as exc:
        # Fail open: never block eval work on a Redis outage. Losing the cap
        # temporarily is far cheaper than stalling every fast-eval run.
        logger.warning(
            f"[acquire_slot] Redis error, failing open | token={token} | error={exc}"
        )
        return True


def release_slot(token: str) -> None:
    """Release the permit held by `token`; the TTL is the backstop if this fails."""
    try:
        _redis_client.zrem(_SLOTS_KEY, token)
    except redis.RedisError as exc:
        logger.warning(
            f"[release_slot] Redis error, permit will expire via TTL | "
            f"token={token} | error={exc}"
        )
