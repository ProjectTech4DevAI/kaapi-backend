# Guardrails

Text safety validation. Two distinct things live under `/api/v1/guardrails`:

1. **A job-based apply path** — Kaapi owns it, runs it through Celery, and persists it as a `Job`.
2. **A thin proxy** to the external `kaapi-guardrails` service for managing the configs those runs reference.

Knowing which of the two a route belongs to explains almost every behavioural quirk below.

## Routes

`backend/app/api/routes/guardrails.py`, tag `Guardrails`, no router prefix — every path is spelled in full.

| Route | Kind | Notes |
|---|---|---|
| `POST /guardrails` | job | Creates a `JobType.LLM_GUARDRAILS` job, returns `job_id` immediately. Publishes its webhook contract via `guardrails_callback_router`. |
| `GET /guardrails/{job_id}` | job | Polls the job; rehydrates the sanitised text from `job.meta`. |
| `GET /guardrails` | proxy | Validator types + their JSON schemas. **Not** `APIResponse`-wrapped. |
| `{POST,GET}/guardrails/ban_lists`, `{GET,PATCH,DELETE} /guardrails/ban_lists/{ban_list_id}` | proxy | Ban list CRUD. |
| `{POST,GET} /guardrails/validators/configs`, `{GET,PATCH,DELETE} /guardrails/validators/configs/{config_id}` | proxy | Validator config CRUD. |
| `{POST,GET} /guardrails/llm_prompt_configs`, `{GET,PATCH,DELETE} /guardrails/llm_prompt_configs/{prompt_config_id}` | proxy | Stored prompts for the LLM-backed validators. |

**Route ordering matters:** every fixed `/guardrails/...` path must stay declared above `GET /guardrails/{job_id}`, or the job route swallows them. `TestProxyRouteOrdering` guards this.

## Models

`backend/app/models/guardrails/`

- `request.py` / `response.py` — Kaapi-owned shapes for the job path.
- `ban_list.py`, `validator_config.py`, `llm_prompt_config.py`, `enums.py`, `common.py` — **mirrors** of the upstream service's schemas, added so Swagger can render real request/response schemas instead of `dict[str, Any]`.

The mirrors are a documentation surface, not a source of truth. The upstream service owns validation; anything it accepts but a mirror rejects turns a working call into a spurious 422. They keep the upstream `created_at`/`updated_at` spelling rather than the Kaapi-wide `inserted_at`, because the objects are echoed verbatim from the other service.

Two shapes are deliberately open (`extra="allow"`): `ValidatorConfigCreate` and `ValidatorConfigPublic`. A validator config is base columns plus a JSONB tuning blob that the service flattens back into one object, so the real payload always carries keys beyond the declared schema.

## Services

- `app/services/guardrails/jobs.py` — `start_job` and the Celery `execute_job`. Dedupes validator ids, calls the service, delivers or stores the result. Everything lands on `job.meta` under `request` / `response` / `callback`.
- `app/services/llm/guardrails.py` — the HTTP client. `proxy_guardrails_request` backs the CRUD routes; `apply_guardrails` / `run_guardrails_validation` back the job path and the `/llm/call` pipeline.

### Two different failure policies

This trips people up. The same file contains both:

- **`proxy_guardrails_request` does not fail open.** Unreachable or non-JSON upstream raises `502`. A config read returning stale silence would be worse than an error.
- **`run_guardrails_validation` does fail open.** If the service is unreachable the job still succeeds, carrying the *original text unchanged*, and a warning is attached. Callers must check `warnings` — on the webhook under `metadata.warnings`, on the poll route under `warnings` — or they will silently treat unvalidated text as sanitised.

Tenant always travels in `X-ORGANIZATION-ID` / `X-PROJECT-ID` headers derived from the auth context, never from a caller-supplied body or query field.

## Swagger docs

Each route's prose lives in `backend/app/api/docs/guardrails/<action>.md`, loaded via `load_description`. `TestOpenAPIDocumentation` asserts every operation keeps a summary, a description, a 200 schema, and a typed request body — add the markdown file when you add a route, or that test fails.

## Gotchas

- The upstream service returns **200 for creates and deletes**, never 201/204. Deletes carry a confirmation body.
- Validator config uniqueness is enforced on **`name` alone**, not on type/stage.
- `PATCH /guardrails/validators/configs/{id}` **cannot** change validator-specific tuning — upstream forbids extra keys there. Delete and recreate.
- PATCH bodies are forwarded with `exclude_unset=True`; without it, omitted optional fields would serialise as `null` and blank out stored data.
- `stage` on a validator config is advisory. `POST /guardrails` routes on the text it is handed, so one config can serve both directions.
