# Module: Assessment

Assessments of datasets/inputs against ASSESSMENT-tagged configs. Three methods, chosen by input shape: **RESPONSE** (object input → `job` → llm_call/llm_chain), **BATCH** (list input → assessment_run → batch_job), **RUN** (legacy UI pipeline). No deep-dive doc yet.

All paths relative to `backend/app/`.

## Routes
- `api/routes/assessment/` — legacy UI (assessments, runs, datasets)
- `api/routes/assessment/api.py` — API-client route, mounted top-level at `/assessments` (**`POST /assessments` only** — no status/result poll endpoint; the result is delivered by webhook to the request's required `callback_url`); method inferred from input shape. BATCH wired; RESPONSE returns 501 (deferred)

## Tables (SQLModel)
`models/assessment/` is a package, split by surface: `assessment.py` holds the shared DB tables + `AssessmentStatus`/`AssessmentMethod` enums + `AssessmentConfigRef` + the legacy RUN (UI) models; `assessment_api.py` holds the API-client request/response models. Both are re-exported from the package `__init__`, so `from app.models.assessment import X` resolves either.

| Table | Model |
|---|---|
| `assessment` (Assessment; parent — `method`, data source; FK → job (RESPONSE), evaluation_dataset, org, project) | `models/assessment/assessment.py` |
| `assessment_run` (AssessmentRun; child — one config execution, BATCH/RUN; FK → assessment, config, batch_job) | `models/assessment/assessment.py` |

Config version (tag=ASSESSMENT, `models/config/assessment_blob.py`) owns system / pre-filters / params / schemas: `input_schema` (**mandatory, non-empty** per-column spec `{type, format}` for BATCH submissions — `type` is required per column; every declared column must be present in every submission row; attachment columns are url-format only) and `json_output_schema` (object-typed structured-output schema, omit for free text). Each pre-filter (`topic_relevance`, `duplicate_detection`) carries its own `provider` (default `openai`) + `params` (TextLLMParams: model, temperature, ...) and runs its own llm call; its criteria live in `params.instructions` (a **mandatory** field, same shape as the assessment call — pre-filters no longer have a top-level `prompt`/`content`). The request input carries the user-message. Strict input types `ResponseInput` (RESPONSE, `{query, attachments}`) / `BatchInput` (BATCH, `{query, data}` where `data` is a **list of submission rows** — each a flat column→string map, an attachment column's value being a url string — and `query` is a `{column}` template) both carry `query` and discriminate structurally on the other key (`data` ⇒ BATCH, else RESPONSE) with `extra=forbid` keeping them disjoint — no `mode` tag (`models/assessment/assessment_api.py`). Legacy RUN runtime lives in `assessment_run.execution` (`RunExecution`).

`assessment.id` is a **UUID** (like config/job/llm_call). Per-item result = `AssessmentResult {output: {assessment, pre_filter}, error}` (no `metadata` — the provider/model/usage block was removed from the API-client output) where `output.assessment` = the LLM output parsed to an object when the config has a `json_output_schema`, else string (null for gated/failed rows), and `output.pre_filter` groups the two verdicts — `{topic_relevance, duplicate_detection}`, each `{verdict, reasoning}` or null — and is itself null when no pre-filter ran. Delivery is **webhook-only**: the `POST /assessments` ack is the flat `AssessmentSubmitResponse {assessment_id, status, message, inserted_at, updated_at}`, and the result is delivered solely by POSTing the `AssessmentCallback {assessment_id, status, data, request_metadata}` to the request's required `callback_url` on completion — where `data` is a single `AssessmentResult` (RESPONSE) or an `AssessmentBatchResult {total_items, counts, items}` (BATCH); `status` lives on the envelope only. Pre-filter `stop_on_fail` flag (`config/assessment_blob.py`) drives which filters hard-stop the chain on a failing verdict vs pass-through (record only).

## Services / CRUD
- `services/assessment/utils/attachments.py` — cell→provider attachment conversion (Drive URL normalization; OpenAI/Anthropic/Gemini part builders). `rewrite_gcs_attachment_urls` bulk-resolves `gs://` cells to provider-reachable URLs via `services/buckets/` (Path A native passthrough for google-gcp / Path B signed HTTPS otherwise) **before** JSONL build. Called in: `services/assessment/api/batch.py::_submit_provider_batch` (API pipeline, all stages), `crud/assessment/batch.py::submit_assessment_batch` (legacy L2), `services/assessment/tasks.py` (legacy prefilter). Submit-time validation (`services/assessment/api/submission.py`) allows `gs://` alongside `http(s)://`.
- `services/assessment/` — legacy RUN pipeline (service, stages, processing, batch, cron, tasks)
- `services/assessment/api/` — API-client pipeline: `submission.py` (submit), `batch.py` (staged provider batches — gate pre-filters → pass-through → assessment, over `core/batch`; `PREFILTER_VERDICT_SCHEMA`), `results.py` (builds `AssessmentBatchResult`), `callbacks.py` (webhook)
- `crud/assessment/api.py` — new API-client crud (method-based Assessment/AssessmentRun writes): `create_assessment`, `set_assessment_job`, `create_execution`, `set_execution_batch_job`, `update_status`, `list_executions` (no `get_assessment` — delivery is webhook-only, so there is no request-time fetch). Namespaced under `api` (`from app.crud.assessment import api`) to avoid colliding with the legacy `create_assessment`.
- `crud/assessment/{core,cron,processing,batch}.py` — legacy RUN pipeline crud. RUN runtime for the dropped columns (`stage`/`stage_status`/`pipeline`/`stage_batches`/`prefilter_total_*`/`object_store_url`) now lives in the `assessment_run.execution` bag via `core._read_exec`/`_write_exec`; `status` is the `AssessmentStatus` enum; the RUN input binding lives on the parent `assessment.input`. `batch.py` builds per-row prompts (`{column}` substitution from the parent `InputBinding.prompt`).

## Async
- Rides shared `core/batch/` provider batch infra + cron polling (same as evaluations).
- API-client BATCH: `celery/tasks/job_execution.py::run_assessment_api_batch` drives the staged pipeline (poll → parse verdict/gate → advance/finalize → callback), self-re-enqueuing between stages. Staged state + `callback_url`/`request_metadata` live in the `assessment_run.execution` bag.

## External
- Provider Batch APIs, object storage for attachments (incl. `gs://` attachments resolved via `services/buckets/`).
- Gemini-family batch provider is chosen inline in `api/batch.py` (`_submit_provider_batch` / `_build_batch_provider`): `google-gcp` -> `GoogleGCPBatchProvider` (Vertex, GCS in/out, `core/batch/google_gcp.py`, built from the `google-gcp` credential), `google` -> `GeminiBatchProvider` (AI-Studio, File API, via `GeminiClient`).
