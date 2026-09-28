# Module: Guardrails

Handoff/context page for changing anything guardrails-related. Guardrails themselves run in a **separate service** (`kaapi-guardrails`); this backend only proxies to it and orchestrates jobs around it. Nothing in this repo evaluates text.

All paths relative to `backend/app/`. Related: [llm-call.md](llm-call.md) (guardrails are part of the `/llm/call` pipeline), [../cross-cutting/auth.md](../cross-cutting/auth.md). Deep dive: `docs/architecture/kaapi-llm-call-ARCHITECTURE.md` §7 — but see §12, it predates the fail-closed-on-auth change.

## 1. Two entry points, one transport

| Entry point | Shape | Where |
|---|---|---|
| `POST /guardrails` | Standalone async job. Caller supplies text + validator IDs, gets `job_id`, result via webhook or poll. For callers running their own LLM workflow. | `api/routes/guardrails.py`, `services/guardrails/jobs.py` |
| `/llm/call` + `/llm/chain` + **fast evaluation** | Inline hooks around the provider call, driven by `config_blob.input_guardrails` / `output_guardrails`. | `services/llm/jobs.py` (`apply_input_guardrails`, `apply_output_guardrails`) |

Both funnel into `apply_guardrails()` in `services/llm/guardrails.py`. That module is the **only** place that talks HTTP to the guardrails service.

## 2. File map

| Concern | File |
|---|---|
| Transport, outcome type, proxy helper | `services/llm/guardrails.py` |
| Standalone job orchestration (dedupe, warnings, callbacks) | `services/guardrails/jobs.py` |
| Routes: async job + poll + management proxies | `api/routes/guardrails.py` |
| Public endpoint contract (rendered into OpenAPI) | `api/docs/guardrails/apply_guardrails.md` |
| Request/response schemas | `models/guardrails/request.py`, `models/guardrails/response.py` |
| Inline adapters + call sites | `services/llm/jobs.py`, `services/llm/chain/chain.py` |
| Celery task + enqueue helper | `celery/tasks/job_execution.py` (`run_guardrails_job`), `celery/utils.py` (`start_guardrails_job`) |
| `JobType.LLM_GUARDRAILS`, `job.meta` | `models/job.py` |
| Settings | `core/config.py` — `KAAPI_GUARDRAILS_URL`, `KAAPI_GUARDRAILS_AUTH` |
| Sentry scrubbing | `core/sentry_filters.py` (`before_send_filter`) |
| Migrations | `alembic/versions/070_add_meta_and_guardrails_jobtype.py` (`job.meta` + enum value), `alembic/versions/083_add_llm_call_metadata.py` (`llm_call.metadata`) |

Timeouts, all hardcoded in `services/llm/guardrails.py`: validator-config fetch 10s, validation POST 45s, management proxy `GUARDRAILS_PROXY_TIMEOUT_SECONDS` 30s.

## 3. Transport layer (`services/llm/guardrails.py`)

`_guardrails_headers()` builds every outbound request: `Authorization: Bearer {KAAPI_GUARDRAILS_AUTH}` plus tenant in `X-ORGANIZATION-ID` / `X-PROJECT-ID`. Tenant values come from the auth context only — never from request body or query.

`apply_guardrails(text, validators, job_id, project_id, organization_id, output_text=None)` makes two upstream hops:

1. `GET /validators/configs/?ids=...` via `list_validators_config()` — resolves config IDs to full validator configs. `output_text is not None` routes the IDs through the output slot instead of the input slot.
2. `POST /` via `run_guardrails_validation()` with `{request_id, input, validators}`, plus `output` when output guardrails are running (for validators that judge an input/output pair). Query param `suppress_pass_logs=false`.

Returns `GuardrailsOutcome(safe_text, error, bypassed, rephrase_needed, raw)`. Property `.applied` = "ran and was not bypassed" — it is the gate for metadata emission.

`summarize_validator_results(outcome)` flattens `raw.data.validator_results` into per-validator `{name, outcome, error, input_text, output_text}`.

`proxy_guardrails_request(method, path, ...)` is a separate, dumber path used only by the management CRUD routes: forwards verbatim, returns `(status_code, parsed_json_or_None)`.

## 4. Failure semantics (the thing most changes get wrong)

| Upstream condition | Behaviour |
|---|---|
| Network error / timeout / 5xx during validation | **Fail open.** `bypassed=True`, original text passes through. |
| 401 / 403 / 422 on validation or on config fetch | **Fail closed.** Job FAILED. `_AUTH_ERROR_STATUS_CODES` includes 422 deliberately: a missing/invalid tenant header is a broken deploy, not a transient outage, so it must not fall open like one. |
| Config fetch fails for any other reason | Returns `([], [])` → `apply_guardrails` short-circuits with a no-op outcome where `bypassed=False`. See §9. |
| Upstream responds `success=false` | **Hard block.** `outcome.error` set; job FAILED. |
| Sanitised text is empty/whitespace | Blocked by the inline adapters ("left no usable content") — an empty prompt would otherwise produce a confusing provider-side error. |

`outcome.error` therefore covers two very different situations, and `GuardrailsOutcome.blocked` is the discriminator: it is `True` only for a content verdict. The fail-closed auth dict carries `auth_error: True`, so `blocked` stays `False` there and callers keep treating it as a plain job failure rather than "the text was rejected".

Error strings handed back to clients are status-only (`"Guardrails service rejected the request (HTTP 403)"`). Never put `str(e)` in them: it embeds the internal service URL and the string surfaces publicly via `job.error_message`.

## 5. Standalone `POST /guardrails`

Route validates `callback_url` (SSRF guard, `utils.validate_callback_url`), then `services/guardrails/jobs.py::start_job` creates a `JobType.LLM_GUARDRAILS` row with `meta={"request": ...}` and enqueues `run_guardrails_job`. HTTP 200 returns `GuardrailsJobImmediatePublic` immediately.

Worker (`execute_job`): PROCESSING → `_dedupe_validators` → `apply_guardrails` → hard block means FAILED + failure callback, otherwise SUCCESS + success callback.

Endpoint-specific behaviour, none of which exists on the `/llm/call` path:

- **Dedupe.** `_dedupe_validators` drops repeated `validator_config_id`s (order-preserving) so callers are not double-billed upstream.
- **Warnings channel.** Bypass, dedupe, and "no validators resolved" become human-readable strings. Delivered as `metadata.warnings` in the callback and `warnings[]` on the poll response. A caller-supplied `warnings` key in `request_metadata` is overwritten (documented in the model docstring).
- **`job.meta`** holds `{request, response, callback}`. Raw unsafe text is persisted on purpose — this endpoint exists to inspect unsafe content.
- **`response_id` is server-minted** (`uuid4`), so callers always have a stable correlation handle even when upstream returns none.
- `rephrase_needed` is ignored here. Per-validator results are not exposed here.

`GET /guardrails/{job_id}` **rebuilds** the payload from `job.meta` rather than replaying the stored callback body — the two representations can drift if only one is changed.

## 6. Inline guardrails in `/llm/call` and `/llm/chain`

`execute_llm_call` in `services/llm/jobs.py` is shared by three callers: `/llm/call`, `/llm/chain` (through `ChainBlock.execute` in `services/llm/chain/chain.py`), and fast evaluation (`crud/evaluations/fast.py::_llm_call_for_item`, with `record_call=False`). A guardrails change therefore changes what an eval measures, not just what production serves.

Order inside `execute_llm_call`: resolve config → capture `original_input_value` → interpolate `prompt_template` → **input guardrails** → create the `LlmCall` row → provider call → **output guardrails** → update that row.

- `apply_input_guardrails` returns a **5-tuple** `(query, error, guardrail_direct_response, metadata, guardrail_outcome)`. The last element is `"blocked"` / `"rephrased"` / `None` and lands on `BlockResult.guardrail_outcome`, so a caller can tell a content verdict from a provider failure — both of which arrive as a bare string in `error`. It is `None` on a fail-closed auth error (see §4).
  - Hard block → job error.
  - `rephrase_needed=True` → `safe_text` is returned to the user **without calling the LLM**; the call is recorded via `save_rephrase_guardrail_call`, and `query.input` is restored to `original_input_value` first.
  - Otherwise `safe_text` overwrites `query.input.content.value`.
  - Non-text input is skipped entirely.
- `apply_output_guardrails` returns `(BlockResult, error, guardrail_outcome)`. An output block now also carries the `usage` actually spent — the provider was called and billed before the verdict arrived. It passes `text=input_text` (the pre-template user input) and `output_text=`the LLM output. On success, sanitised text overwrites the response content, then `persist_output_guardrail_result` re-writes the already-created `LlmCall` row **in its own session** (the caller's session may already be closed).
- **There are two output-guardrail call sites** inside `execute_llm_call`: one in the proxy-provider branch, one in the main provider path. Any change to output guardrails must touch both.

## 7. Guardrail metadata (`include_guardrail_metadata`, PR #1195)

Opt-in flag on `LLMCallRequest`, default `False`. Gated on `outcome.applied`, so bypassed and no-op runs emit **nothing** — absence of metadata does not mean "guardrails passed".

- Input side produces `{"input_guardrail": {input_from_user, input_to_llm, validators}}`, merged into `request_metadata`, which is then stored on the new `llm_call.metadata` column (model field `metadata_`, renamed to dodge SQLAlchemy's reserved `metadata`; migration 083).
- Output side produces `{"output_guardrail": {output_from_llm, output_to_user, validators}}` on `result.metadata`, persisted through `update_llm_call_response` (merge, not replace).
- `GET /llm/call/{job_id}` returns `llm_call.metadata_` as the response envelope's `metadata`.
- **`/llm/call` only.** `LLMChainRequest` has no such field and `ChainBlock.execute` does not forward the flag, so chains never emit guardrail metadata.

## 8. Management proxy routes (PR #1135)

Thin pass-throughs over the upstream management API, all under `require_permission(Permission.REQUIRE_PROJECT)`:

- `GET /guardrails` — validator catalogue
- `/guardrails/ban_lists` — POST, GET (`domain`, `offset`, `limit`); `/{id}` GET/PATCH/DELETE
- `/guardrails/llm_prompt_configs` — POST, GET (`validator_name`, `offset`, `limit`); `/{id}` GET/PATCH/DELETE
- `/guardrails/validators/configs` — POST, GET (`ids`, `stage`, `type`); `/{id}` GET/PATCH/DELETE

These do **not** fail open: upstream status codes and bodies (including its 422s) are returned unchanged. `_upstream_response` sets `Cache-Control: no-store` (tenant-scoped data must not sit in a shared cache, CWE-525) and keeps an empty upstream body empty, since a 204 cannot carry one. A non-JSON upstream body becomes a 502.

## 9. Invariants — do not break these

1. Tenant identity travels in headers derived from the auth context. Never accept `organization_id`/`project_id` from a request body or query string.
2. 401/403/422 fail closed; network/5xx fail open. Changing either side changes the security posture — say so explicitly in the PR.
3. Client-visible error strings stay status-only, never `str(e)`.
4. Fixed `/guardrails/*` paths stay declared **above** `GET /guardrails/{job_id}`. FastAPI matches in declaration order and will not fall through on a UUID parse failure.
5. Management proxies return upstream status/body verbatim with `no-store`.
6. Metadata emission stays gated on `outcome.applied`.
7. Raw text persistence is intentional on the `/guardrails` job, not an oversight.
8. `guardrail_outcome` marks content verdicts only. Never label a fail-closed auth/transport error `"blocked"` — a caller that treats blocks as benign (evaluation does) would silently swallow a broken deploy.

## 10. Touch-map: common changes

| Change | Files, in order |
|---|---|
| Add/modify an output guardrail behaviour | `apply_output_guardrails`, then **both** call sites in `execute_llm_call` (proxy branch + main provider path) |
| Change `apply_input_guardrails`' return | It is a 4-tuple — update the unpack in `execute_llm_call` |
| Add a field to the `/guardrails` result | `models/guardrails/response.py` → `_build_callback_payload` (callback path) → `get_guardrails_job_status` (poll path, rebuilds independently) → `api/docs/guardrails/apply_guardrails.md` |
| Add a management proxy route | `api/routes/guardrails.py` using `proxy_guardrails_request` + `_upstream_response`, declared above `/{job_id}` |
| Change upstream request/response shape | `run_guardrails_validation` and/or `list_validators_config`; check `summarize_validator_results` still finds `data.validator_results` |
| Add a new warning | `_dedupe_validators` or `_outcome_warnings` in `services/guardrails/jobs.py`; document it in the api doc |
| Change timeouts | `services/llm/guardrails.py` (three separate values, see §2) |
| Forward `include_guardrail_metadata` to chains | `LLMChainRequest`, `ChainBlock.__init__`/`execute`, `LLMChain`, `execute_chain_job`. **Hazard:** blocks share one `context.request_metadata`, and `apply_input_guardrails`' result is merged with an in-place `.update()` — every block would overwrite the previous block's `input_guardrail`. Key it per block before wiring this up. |

## 11. Inferred upstream contract

Not authoritative — reconstructed from call sites in this repo. The real contract lives in `kaapi-guardrails`.

```
POST /            -> {success, bypassed?, error, data: {safe_text, rephrase_needed, validator_results[], usage{input_tokens,output_tokens,total_tokens,reasoning_tokens}}}
GET  /validators/configs/?ids=... -> {success, data: [ {...validator config...} ]}
```

`validator_results[]` entries are read for `name`, `outcome`, `error`, `input_text`, `output_text`.

## 12. Known issues / open questions

- **`input_from_user` is post-template.** `prompt_template` interpolation runs before input guardrails inside `execute_llm_call`, so the metadata field holds the templated prompt, not the raw user text (`original_input_value`, captured before interpolation, is what output guardrails receive).
- **The architecture doc is stale on failure semantics.** `docs/architecture/kaapi-llm-call-ARCHITECTURE.md` §5, §7 and §10 state guardrails are fail-open unconditionally. That predates PR #1135, which made 401/403/422 fail closed. Trust the §4 matrix here.
- **Guardrail bypass is invisible to the caller.** When the service is unreachable the outcome is `bypassed=True` (§4), metadata emission is gated on `outcome.applied`, and neither adapter propagates the bypass flag. A row that silently skipped guardrails looks identical to one that passed them. This matters most for evaluation, where a whole run can quietly measure the bare model. Carrying a `"bypassed"` value through both adapters is the fix; it was deliberately left out of the eval-guardrail change to keep that diff contained.
- **Guardrail metadata reaches failure callbacks only by accident.** A hard block returns `BlockResult(error=..., llm_call_id=...)` with no `metadata`, and the failure callback in `execute_job` is built from `request.request_metadata`. Input metadata still shows up there when the caller supplied a `request_metadata` dict (even `{}`), because `apply_input_guardrails`' result is merged with an in-place `.update()` on that same object; it is dropped when the caller sent none. The alias also breaks on the main provider path whenever `transform_kaapi_config_to_native` rebinds `request_metadata` to a new dict to attach warnings. Do not rely on either behaviour — make it explicit if failure payloads need the metadata.
- **Silent no-op when the config fetch fails non-auth.** `list_validators_config` returns `[]` and the request proceeds with no guardrails at all, signalled only by a log line on the `/llm/call` path. On `/guardrails` it surfaces as a "none resolved" warning, indistinguishable from genuinely bad validator IDs.
- **Unverified:** in output mode the success branch does `data.get("safe_text", text)` where `text` is the *user input*. If upstream ever omits `safe_text`, the LLM output would be replaced by the user's input. Depends on the upstream contract, which is not visible from this repo.
- **Sentry redaction does not cover `/guardrails`.** `_LLM_JOB_TASK_NAMES` lists `run_llm_job`, `run_llm_chain_job`, `run_response_job` — not `run_guardrails_job` — and `_SENSITIVE_REQUEST_DATA_KEYS` does not include `text`. So the standalone endpoint's raw text and `callback_url` reach Sentry unredacted via the Celery integration.

## 13. Tests

| File | Covers |
|---|---|
| `tests/api/routes/test_guardrails.py` | Routes, proxy behaviour, route ordering, poll |
| `tests/services/llm/test_guardrails.py` | Transport, fail-open/fail-closed matrix |
| `tests/services/guardrails/test_jobs.py` | Standalone job worker, dedupe, warnings, callbacks |
| `tests/services/llm/test_jobs.py` | Inline adapters, metadata emission |
| `tests/core/test_sentry_filters.py` | Redaction |

Mocking style is `unittest.mock.patch`, not `respx`. Transport tests patch `app.services.llm.guardrails.httpx.Client`; route tests patch `app.api.routes.guardrails.start_job` or stub upstream via a local `_mock_upstream` helper; worker tests patch `Session` and `JobCrud` inside `app.services.guardrails.jobs` (the real ones would escape the test fixture's savepoint rollback).

Run: `uv run bash scripts/tests-start.sh`.

## 14. Workflow

- Layer edits (model/crud/service/route/migration/celery) go through the `senior-engineer` subagent; tests through `test-writer`. See root `CLAUDE.md`.
- Run `/pr-review` on the full diff before committing.
- Maintenance rule: a change to guardrails routes/models/services updates **this page** in the same PR (and `domain-map.md` if entities or edges changed).
