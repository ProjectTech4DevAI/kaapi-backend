# Cross-cutting: Rate Limiting

All paths relative to `backend/app/`.

## Limiter vs monitor
| | `core/rate_limit.py` → `enforce_rate_limit(category)` | `core/rate_monitor.py` → `monitor_rate(category)` |
|---|---|---|
| Effect | Blocks with `429` + `Retry-After` header | Alert only (Sentry `threshold_rate_monitor`) |
| Scope | per organization (all projects of an org share one window) | per project |
| Window | moving 60 s window, `limits` library (`MovingWindowRateLimiter` + `RedisStorage`, atomic Lua) | fixed 1-minute bucket, raw redis-py INCR |
| Config | `RATE_LIMITS` map ← `*_RATE_LIMIT_PER_MINUTE` settings | `THRESHOLD_*_RATE` settings |
| Redis down | fails open: logs an error, request proceeds (1 s socket timeout) | skips silently |

Both attach as FastAPI route dependencies; neither touches the DB.

## Enforced limits
| Category | Setting | Default | Endpoints |
|---|---|---|---|
| `evaluations` | `EVAL_RATE_LIMIT_PER_MINUTE` | 3 | `POST /api/v2/evaluations`, `POST /api/v2/evaluations/iterations` (one shared bucket). v1 `POST /api/v1/evaluations` is not limited. |

## Response contract
- Status `429`, header `Retry-After: <seconds>`, body is the standard `APIResponse` failure with a generic `error`: `Too many requests. Please try again in N seconds.` The category, the limit value and the per-org scope are never exposed.
- Each rejection logs `[enforce_rate_limit]` at warning with `category`, `org_id`, `project_id` and `retry_after`.

## Adding a limit to an endpoint
1. Add a `*_RATE_LIMIT_PER_MINUTE` setting in `core/config.py`.
2. Add the category to `RateLimitCategory` and an entry to `RATE_LIMITS` in `core/rate_limit.py`. A missing entry raises `KeyError` when the route module is imported, not on a request.
3. Add `Depends(enforce_rate_limit("<category>"))` to the route's `dependencies`, after `require_permission` (it needs the org) and after `monitor_rate`, so rejected attempts still count toward the alert.

## Gotchas
- The check runs before body validation, so a request that later fails with 422 or 404 still uses a slot. Rejected (429) hits do not consume the window.
- Redis keys: `rate_limit:LIMITER/<category>:<org_id>/<amount>/1/minute`. Changing the limit switches to a new key; nothing to migrate.
- Tests: the autouse `isolated_rate_limiter` fixture in `tests/conftest.py` swaps in a fresh in-memory limiter per test, so route tests sharing one org never trip 429 or write to real Redis.
