# Celery task conventions (`app/celery/tasks/`)

Authoritative conventions for Celery tasks in kaapi-backend. Tasks live in `app/celery/tasks/`.
Celery uses RabbitMQ as broker. There is a **single `default` queue** declared with
`x-max-priority=10`; tasks are ordered by a per-task `priority` (higher drains first, FIFO within
a band). Read `app/celery/tasks/job_execution.py` before writing — it shows the full pattern
(decorator + timeout + OTel propagation + delegation to a service).

## Canonical decorator stack

```python
@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_my_job")
def run_my_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.my_domain.jobs import do_the_work  # late import to avoid cycles

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: do_the_work(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )
```

`_set_trace`, `_run_with_otel_parent`, and `gevent_timeout` already exist in this module /
`app/celery/utils.py` — reuse them, don't reinvent.

## Priority choice — be explicit

All tasks share the one `default` queue; set `priority` to place the task in the right band. Match
the bands already in use (see the `job_execution.py` module docstring):

| Priority | When |
|---|---|
| `9` | User-blocking, interactive — LLM call / chain / response jobs |
| `6` | Fast evaluation |
| `2` | Default — doctransform, collections, STT/TTS evaluation, assessment |
| `1` | Notifications and other fire-and-forget background work |

Pick the band that matches the task's user-facing urgency; document the choice in a comment if it's
not obvious. `task_inherit_parent_priority=True` is set, so a task enqueued from another task
inherits its priority unless you override it.

## Hard rules

- **`bind=True`** so you have `self` (the task instance) for retries, IDs, etc.
- **Always pass `queue="default"` and an explicit `priority`** — there is only one queue; the priority is what matters.
- **Pass `trace_id` explicitly** as a parameter and call `_set_trace(trace_id)` first thing. This wires `asgi_correlation_id` so logs from inside the task match the originating request.
- **Wrap the work in `_run_with_otel_parent(self, lambda: ...)`** so OpenTelemetry parent context propagates from the enqueueing process.
- **Delegate to a service.** The task body should be a thin shim over `app/services/<domain>/`. No DB queries, no external HTTP, no business logic inside the task itself.
- **Late-import the service inside the function body** (as the canonical pattern does). Celery workers boot faster and you avoid model-import cycles.
- **Idempotency.** Celery will redeliver. Either:
  - The work is naturally idempotent (`UPDATE ... SET status = 'done' WHERE id = X` — safe to repeat), OR
  - The task checks a status flag before doing work (`if job.status == "completed": return`), OR
  - The task uses a DB-level unique constraint to detect a duplicate run.
  Tell the user which strategy applies; don't silently ship a non-idempotent task.
- **Retries.** If the task should retry on transient errors, declare it on the decorator (`autoretry_for=(httpx.HTTPError,), retry_backoff=True, retry_kwargs={"max_retries": 3}`). Don't catch-and-re-raise.
- **No blocking calls in `async def`.** Celery tasks are sync; never mix.
- **Timeouts:** rely on the `@gevent_timeout(...)` decorator (or Celery's `soft_time_limit` / `time_limit` on the decorator). External HTTP inside the service should also have its own timeout.

## Registering the task

- New task files under `app/celery/tasks/` must be imported somewhere Celery's autodiscover picks them up. Read `app/celery/celery_app.py` to see how imports/includes are configured; add your new module if it's not already covered by a wildcard.
- The Celery Beat schedule (recurring tasks) lives in `app/celery/beat.py`. If your task should run on a cron, add the entry there.

## Logging

- `logger = logging.getLogger(__name__)` at the module top.
- Every line is `logger.info(f"[task_name] Message | key: {value}")`. Log start, finish, and any retry. Mask PII / credentials with `mask_string` from `app.utils` — **never log raw payloads** if they may contain sensitive data.

## What you DO NOT do

- Don't write SQL or call CRUD directly from the task body.
- Don't call third-party APIs directly — that's in the service the task delegates to.
- Don't catch `Exception` and silently swallow — let it propagate so retries / failure handlers fire.
- Don't run `.delay(...)` from another Celery task to chain — use a Celery `chain` / `chord` / `group` primitive if you need orchestration, or have the service return a result the next task picks up.
- Don't use `time.sleep(...)` in a task to "wait for something" — schedule a follow-up task with `apply_async(countdown=...)`.
