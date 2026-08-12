# Runtime Topology

Who talks to whom. Details of each hop live in the module pages.

```
Client / kaapi-frontend
   │  JWT or API key
   ▼
FastAPI (backend/app/main.py, api/routes/*)
   │
   ├── Postgres (SQLModel, core/db.py, alembic migrations)
   ├── Celery workers ── RabbitMQ (broker) + Redis (results)
   │      └── app/celery/tasks/job_execution.py, 2 queues (default, evaluations) + priority bands
   ├── Object storage        core/cloud/storage.py
   ├── Langfuse              core/langfuse/langfuse.py (traces, scores)
   ├── Sentry                core/sentry_filters.py
   └── Providers
         ├── OpenAI    (llm calls, embeddings, batch, vector stores, responses API)
         ├── Gemini    (llm calls, batch: core/batch/gemini.py)
         └── Anthropic (core/batch/anthropic.py)
```

## Processes

| Process | Entry | Notes |
|---|---|---|
| API server | `fastapi run app/main.py` | routes in `app/api/routes/` |
| Celery worker | `app/celery/` config | `default` queue (most tasks) + dedicated `evaluations` queue/pool (fast-eval chunk/aggregate, isolated so eval bursts can't delay LLM jobs), priority bands within each |
| Celery beat / cron | `app/api/routes/cron.py` + `crud/evaluations/cron.py` | batch polling for eval/assessment runs |

## Environment

- `.env` (app), `.env.test` (tests). Settings in `app/core/config.py`.
- Provider credentials per org/project in the `credential` table, falling back to platform env keys.
