# Kaapi Wiki Index

Router for LLM sessions. Load this file, then open ONLY the page(s) the task touches.
Deep design narrative lives in `docs/architecture/*.md`; open those only for design-level questions.

## How to use

1. Find the module/domain below.
2. Read its `modules/*.md` page (lean: tables, models, services, paths).
3. Only if you need the "why" (design rationale, flows, trade-offs), follow the page's deep-dive link.

## Pages

### Maps
- [domain-map.md](domain-map.md) — entities + FK/consumed-by edges. Source of truth for blast-radius analysis when planning a feature.
- [services.md](services.md) — runtime topology: FastAPI, Celery/RabbitMQ/Redis, Postgres, object storage, Langfuse, providers.

### Modules
- [modules/llm-call.md](modules/llm-call.md) — `POST /llm/call` pipeline, configs (`Config`/`ConfigVersion`, `LLMCallConfig`), guardrails, chains. Deep dive: `docs/architecture/kaapi-llm-call-ARCHITECTURE.md`
- [modules/evaluations.md](modules/evaluations.md) — text/STT/TTS evals, datasets, runs, batch + cron scoring, fast evals. Deep dive: `docs/architecture/kaapi-evaluations-ARCHITECTURE.md`
- [modules/knowledge-base.md](modules/knowledge-base.md) — documents, collections, transforms, vector-store providers. Deep dive: `docs/architecture/kaapi-knowledge-base-ARCHITECTURE.md`
- [modules/responses.md](modules/responses.md) — OpenAI Responses API integration, conversations, threads, assistants. No deep-dive doc yet.
- [modules/assessment.md](modules/assessment.md) — assessments and assessment runs. No deep-dive doc yet.
- [modules/tenancy.md](modules/tenancy.md) — users, orgs, projects, API keys, onboarding, login.
- [modules/platform.md](modules/platform.md) — analytics, notifications, feature flags, languages, credentials, model config, cron.

### Cross-cutting
- [cross-cutting/auth.md](cross-cutting/auth.md) — JWT, API keys, org/project permission model.
- [cross-cutting/exceptions.md](cross-cutting/exceptions.md) — global exception handlers, provider error handling convention.
- [cross-cutting/observability.md](cross-cutting/observability.md) — Langfuse tracing, logging convention, Sentry, telemetry.

## Maintenance rule

A PR that changes a module's routes/tables/models/services updates that module's wiki page in the same PR. Pages hold names and paths, never line numbers.
