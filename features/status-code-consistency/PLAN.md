# Status Code and Error Message Consistency (issue #834): Implementation Plan

Source spec: [GitHub issue #834](https://github.com/ProjectTech4DevAI/kaapi-backend/issues/834), "Observability: Improve status codes consistency"

## Summary

Kaapi returns HTTP 500 for conditions that are not server bugs: a missing Langfuse dataset, a provider or storage failure, a KMS encrypt that failed, a broker that could not accept a job. Callers cannot tell "you asked for something that does not exist" from "we broke", and ops alerting on 5xx rate is polluted by caller errors. This change introduces a central domain exception taxonomy in `app/core/exceptions.py` with registered handlers, then converts the mis-coded raise sites across eight domains, and completes the log-level and log-prefix audit on the same lines. No schema change, so no Alembic migration.

Domains in scope: the five the issue names (LLM Call, LLM Chain, Collections, Documents, Evaluations including STT/TTS), plus Onboarding, Config Management, and Credentials added by later scope decision.

### Note on the issue's two cited line numbers

Both citations are stale against `main`; the paths also moved under `backend/`.

- `app/crud/assistants.py:68` is already correct. [backend/app/crud/assistants.py:64-73](backend/app/crud/assistants.py#L64-L73) raises 404 on `openai.NotFoundError` and 502 on `openai.OpenAIError`. No change needed, and `assistants` belongs to the Responses module, which is not among the in-scope domains.
- `app/crud/evaluations/batch.py:47` is now a docstring line, and no `success` boolean exists in that file. The underlying defect survives in a different shape: [backend/app/crud/evaluations/batch.py:49-58](backend/app/crud/evaluations/batch.py#L49-L58) raises a bare `ValueError` when Langfuse cannot return the dataset, which nothing catches on the fast-eval path, so the generic handler turns it into a 500. Production evidence: `app/logs/app.log.3:9975` logs `status_code: 404, body: {'message': 'Dataset not found', 'error': 'LangfuseNotFoundError'}` on a request that returned 500. Step 3 fixes it.

### The taxonomy this plan applies

Every code change below is an application of this table. It is derived from `.claude/conventions/route.md` ("Status codes (the ones to get right)") and `.claude/conventions/error-handling.md`.

| Condition | Code | Log level |
|---|---|---|
| Resource absent, ours or an upstream-held resource we reference | 404 | `warning` |
| State conflict (already exists, already deleted) | 409 | `warning` |
| Payload unparseable or wrong shape (bad CSV, unreadable file) | 422 | `warning` |
| Valid shape, unacceptable value (unsupported transform pair) | 400 | `warning` |
| Upstream returned an error or was unreachable (Langfuse, OpenAI, Gemini, object storage) | 502 | `error` |
| Our async infrastructure could not accept the work (Celery broker, RabbitMQ) | 503 | `error` |
| Genuinely our bug (DB write failed, unexpected exception) | 500 | `error` |

## Blast Radius

Primary entities: this is a cross-cutting change to the HTTP error contract, not to any table. Entities whose read/write paths change their failure responses: `EvaluationDataset`, `EvaluationRun`, `STTSample`/`STTResult`, `TTSResult`, `Collection`, `CollectionJob`, `Document`, `DocTransformationJob`, `Job`, `LlmCall`, `LlmChain`, `Config`, `ConfigVersion`, `Credential`, `Organization`, `Project`, `User`, `APIKey`.

| Surface | Hop | Impact | Decision |
|---|---|---|---|
| `app/core/exception_handlers.py` | 1 | New typed handler for `KaapiError`; `generic_error_handler` stops echoing `str(exc)` | in scope |
| Evaluations routes/services/crud (text, STT, TTS) | 1 | 500 becomes 404 / 422 / 502 / 503 at named sites | in scope |
| Collections crud/routes | 1 | "already deleted" 400 becomes 409; `CollectionNameConflictError` re-based on `ConflictError` | in scope |
| Documents services/crud | 1 | Unsupported file format 400 becomes 422; wrong-module `HTTPException` import corrected | in scope |
| LLM Call, LLM Chain, Guardrails job start | 1 | Celery enqueue failure 500 becomes 503 | in scope |
| Config Management (`crud/config/`) | 1 | Catch-alls swallow `IntegrityError` into 500; `ConfigBlob` validation returns 400 where the request boundary returns 422 | in scope |
| Credentials (`crud/credentials.py`, `routes/credentials.py`) | 1 | Unique-constraint 400 becomes 409; delete-miss 500 becomes 404; bare `ValueError` sites typed | in scope |
| `app/core/security.py` KMS encrypt/decrypt | 1 | `ValueError` on AWS KMS failure surfaces as 500; becomes 502 | in scope |
| Onboarding (`crud/onboarding.py`) | 1 | Unguarded `encrypt_credentials` and unguarded final `session.commit()` both surface as 500 | in scope |
| `app/tests/` status-code assertions | 1 | 8 assertions encode the current wrong codes and must flip | in scope |
| `docs/wiki/cross-cutting/exceptions.md` | 1 | Must document the new taxonomy module (wiki maintenance rule) | in scope |
| `app/api/docs/**/*.md` swagger copy | 2 | 3 files in the touched domains mention status codes and may go stale | in scope |
| Sentry (`app/core/sentry_filters.py`) | 2 | Reclassifying caller errors off 5xx reduces event volume; no filter code change needed | in scope, verify only |
| kaapi-frontend console | 2 | Breaking if it branches on status code; `APIResponse` body shape is unchanged | **out of scope** (user decision) |
| Langfuse | 2 | Trace and score shape unchanged | unaffected |
| Provider Batch APIs (OpenAI, Gemini, Anthropic) | 2 | Batch payload shape unchanged | unaffected |
| Object storage | 2 | Upload failures resurface as 502 instead of 500; no storage code change | in scope, read path only |
| auth, fine_tuning, assessment, response, api_key | 2 | Same defect class present but outside the eight in-scope domains | **out of scope**, follow-up issue (step 13) |

## Steps

### 1. Core: add the domain exception taxonomy
- Files: `backend/app/core/exceptions.py` (new)
- Define `KaapiError(Exception)` carrying `status_code: int` and `detail: str`, plus subclasses `NotFoundError` (404), `ConflictError` (409), `InvalidPayloadError` (422), `InvalidValueError` (400), `UpstreamError` (502), `ServiceUnavailableError` (503).
- `UpstreamError` takes a `provider: str` so the message can name who failed.
- Precedent to follow, not duplicate: the four existing ad-hoc exceptions (`CloudStorageError` in `core/cloud/storage.py:34`, `GeminiClientError` in `core/batch/client.py:15`, `CollectionNameConflictError` in `crud/collection/collection.py:18`, `TransformationError` in `services/doctransform/registry.py:9`) stay where they are; only `CollectionNameConflictError` is re-based in step 6.
- Depends on: nothing

### 2. Core: register the handler and stop the 500 body leak
- Files: `backend/app/core/exception_handlers.py` (change)
- Add `@app.exception_handler(KaapiError)` inside `register_exception_handlers`, returning `JSONResponse(status_code=exc.status_code, content=APIResponse.failure_response(exc.detail).model_dump())`, matching the existing `http_exception_handler` shape at lines 98-108.
- Change `generic_error_handler` (lines 110-117) to log the exception with `exc_info=True` and the correlation id, and return a fixed message instead of `str(exc)`. This is what currently exposes raw `ValueError` text (including Langfuse response bodies) to API clients.
- Handler registration order matters: register the `KaapiError` handler before the bare `Exception` handler.
- Depends on: step 1

### 3. Evaluations crud: dataset fetch raises typed errors
- Files: `backend/app/crud/evaluations/batch.py` (change)
- `fetch_dataset_items` (lines 35-70): in the `except Exception` at line 51, branch on the Langfuse error. Not-found raises `NotFoundError(f"Dataset '{dataset_name}' not found")`; any other Langfuse failure raises `UpstreamError(provider="langfuse", ...)`. Log level follows the taxonomy table (`warning` for not-found, `error` with `exc_info=True` for upstream).
- Line 58 empty-dataset `ValueError` becomes `InvalidPayloadError`.
- The three other `raise ValueError` sites in this file get the same treatment; the file has 5 total.
- Depends on: step 1

### 4. Evaluations services: upstream and broker failures
- Files: `backend/app/services/evaluations/dataset.py`, `backend/app/services/evaluations/fast.py`, `backend/app/services/evaluations/prompt_improvement.py` (change)
- `dataset.py:131-138`: Langfuse upload failure becomes `UpstreamError` (502) instead of `HTTPException(500)`.
- `fast.py:211-215`: the broad `except` currently swallows step 3's typed errors into a flat 500. Re-raise `KaapiError` untouched before the catch-all, so a missing dataset reaches the caller as 404. The run is still marked `failed` in both branches.
- `prompt_improvement.py:207-210`: Celery enqueue failure becomes `ServiceUnavailableError` (503).
- Leave `crud/evaluations/dataset.py:102-105` at 500. A failed DB write is genuinely our fault and already logs at `error` with `exc_info=True`.
- Depends on: steps 1, 3

### 5. STT/TTS evaluations: queue, storage, and empty-dataset codes
- Files: `backend/app/api/routes/stt_evaluations/evaluation.py`, `backend/app/api/routes/tts_evaluations/evaluation.py`, `backend/app/services/stt_evaluations/audio.py` (change)
- `stt_evaluations/evaluation.py:108-111` and `tts_evaluations/evaluation.py:112-115`: "Failed to queue batch submission" becomes 503.
- `stt_evaluations/evaluation.py:65` and `tts_evaluations/evaluation.py:69`: "Dataset has no samples" moves 400 to 422, matching the empty-dataset code chosen in step 3.
- `audio.py:126-129`: object storage upload failure becomes `UpstreamError` (502). The `except HTTPException: raise` guard at line 119 becomes `except (HTTPException, KaapiError): raise`.
- Depends on: steps 1, 3

### 6. Collections: conflict semantics
- Files: `backend/app/crud/collection/collection.py`, `backend/app/api/routes/collections.py` (change)
- `collection.py:114`: "Collection already deleted" moves 400 to 409. It is a state conflict, not bad input.
- `collection.py:18`: `CollectionNameConflictError` subclasses `ConflictError` and carries its own detail, so the manual translation at `collections.py:220-224` can be deleted and the handler from step 2 covers it. Keep the `name` attribute; it is read to build the message.
- Depends on: steps 1, 2

### 7. Documents: format errors and the wrong-module import
- Files: `backend/app/services/documents/helpers.py`, `backend/app/crud/document/document.py`, `backend/app/crud/document/doc_transformation_job.py` (change)
- `helpers.py:62-64`: unsupported or unreadable source file format moves 400 to 422, matching `services/evaluations/validators.py:84-97`, which already returns 422 for the same class of problem. See Open Questions.
- `helpers.py:67-72` and `helpers.py:76-83` stay at 400. An unsupported transform pair and an unknown transformer are valid-shape, unacceptable-value cases.
- `helpers.py:59`: the docstring says `Raises: HTTPException(400)` and goes stale; update it.
- `document.py:8` and `doc_transformation_job.py:16` import `HTTPException` from `app.core.exception_handlers` rather than `fastapi`. Correct both imports. Behaviour is identical today, but it couples crud to the handler module.
- Depends on: step 1

### 8. LLM Call, LLM Chain, Guardrails: broker failures
- Files: `backend/app/services/llm/jobs.py`, `backend/app/services/guardrails/jobs.py` (change)
- `llm/jobs.py:181-183` (`start_job`), `llm/jobs.py:241-244` (`start_chain_job`), `guardrails/jobs.py:87-90` (`start_job`): all three catch a failed Celery enqueue and return 500. All three become `ServiceUnavailableError` (503). The job is still marked `FAILED` first, unchanged.
- Guardrails is included because `docs/wiki/modules/llm-call.md` places `api/routes/guardrails.py` in the LLM Call module.
- Depends on: step 1

### 9. Config Management: catch-all scope and validation code
- Files: `backend/app/crud/config/config.py`, `backend/app/crud/config/version.py` (change)
- `config.py:70-80` (`ConfigCrud.create`) and `version.py:110-120` (`ConfigVersionCrud.create_from_partial`) both wrap their DB writes in a bare `except Exception` that returns 500. A duplicate config name raises `IntegrityError` inside that block and is flattened to 500, even though `config.py:191` already returns 409 for the same conflict on another path. Add an `except IntegrityError` branch returning `ConflictError` ahead of the catch-all, and a `except (HTTPException, KaapiError): raise` guard so typed errors raised inside are not swallowed.
- `version.py:78-85`: a `ConfigBlob` Pydantic `ValidationError` returns `HTTPException(400, detail=validation_errors)`. The same `ConfigBlob` failing at the request boundary returns 422 through `validation_error_handler`. Move this to 422 so one config blob has one rejection code regardless of entry path. The list `detail` still routes through `_sanitize_validation_errors` at `exception_handlers.py:103-104`, so the body shape is unchanged.
- Leave the immutability checks at `version.py:201`, `version.py:330`, and `version.py:340` at 400 for now. See Open Questions.
- Depends on: steps 1, 2

### 10. Credentials: conflict, not-found, and KMS failures
- Files: `backend/app/crud/credentials.py`, `backend/app/api/routes/credentials.py`, `backend/app/core/security.py` (change)
- `credentials.py:66-69`: the unique-constraint branch (`uq_credential_org_project_provider`) returns 400. `.claude/conventions/route.md` names 409 for exactly this case. Move to `ConflictError`.
- `credentials.py:70-72`: the non-duplicate `IntegrityError` path raises a bare `ValueError` carrying the raw DB error, which reaches the client as a 500 body. Raise a typed error with a sanitized message; the DB text stays in the log only.
- `credentials.py:190`: `raise ValueError("Provider and credential must be provided")` surfaces as 500, while `routes/credentials.py:122` and `routes/credentials.py:263` raise 400 for the identical condition. Make the crud site 400 to match.
- `credentials.py:285-289` (`remove_provider_credential`): `rowcount == 0` means nothing matched, so it is a 404, not the current 500. Change the code and drop the log from `error` to `warning`, since a caller deleting a non-existent credential is not an outage.
- `credentials.py:322-329` (`remove_creds_for_org`): split the branch. Zero rows deleted returns 404; a genuine partial delete (`0 < rows_deleted < expected_count`) keeps 500 and stays at `logger.error`, because it is a real anomaly.
- `security.py:213-216` and `security.py:233-236`: `encrypt_credentials` and `decrypt_credentials` raise `ValueError` when AWS KMS fails. KMS is upstream, so raise `UpstreamError(provider="kms")` (502). Keep the existing behaviour of never surfacing the underlying message, which may carry AWS ARNs, and add the missing `exc_info=True` to both `logger.error` calls.
- Leave `routes/credentials.py:49` at 500. An empty result from `set_creds_for_org` after no exception is a genuine internal anomaly.
- Leave the `validate_provider_credentials` 400s at `credentials.py:36`, `credentials.py:160`, and `credentials.py:208` unchanged. See Open Questions.
- Depends on: steps 1, 2

### 11. Onboarding: KMS and the unguarded commit
- Files: `backend/app/crud/onboarding.py` (change)
- `onboarding.py:126`: `encrypt_credentials(values)` is called with no guard, so a KMS failure becomes a 500. Step 10 makes it raise `UpstreamError`, so this site only needs to stop being the reason a 500 escapes; verify no local `except` re-flattens it.
- `onboarding.py:139` (`session.commit()`): the whole multi-step onboarding has no `try`. The pre-checks at `onboarding.py:67-71` and `onboarding.py:100-105` are read-then-write races, so two concurrent onboards of the same org and project name hit an `IntegrityError` at commit and return 500 with raw DB text. Wrap the commit in `except IntegrityError` returning `ConflictError` with the same wording the pre-check uses, and roll back.
- The two existing 409s (`onboarding.py:68`, `onboarding.py:102`) are already correct and stay as they are.
- Depends on: steps 1, 2, 10

### 12. Observability: log-level and prefix audit in the in-scope domains
- Files (change): `backend/app/api/routes/stt_evaluations/evaluation.py:98`, `backend/app/api/routes/tts_evaluations/evaluation.py:102`, `backend/app/crud/evaluations/cron_utils.py:325`, `backend/app/crud/evaluations/embeddings.py:325`, `backend/app/crud/evaluations/langfuse.py:197,599`, `backend/app/crud/evaluations/processing.py:328,1027,1078,1201`, `backend/app/crud/stt_evaluations/batch.py:228`, `backend/app/crud/tts_evaluations/batch.py:137`, `backend/app/services/collections/providers/gemini.py:148,173`
- Add `exc_info=True` to these 14 `logger.error` calls that caught a real exception without it, per `.claude/conventions/error-handling.md` ("`exc_info=True` on any path that caught a real exception").
- Files (change): `backend/app/crud/evaluations/core.py:106,239` and `backend/app/crud/evaluations/embeddings.py:105,111,120,163,174,232,296,331,387`
- Add the missing `[function_name]` bracket prefix required by CLAUDE.md. 12 lines.
- Log levels already match the fault table at the status-code sites this plan touches; verify rather than assume when editing each one. The two credentials delete sites are the exception and are handled in step 10.
- Depends on: steps 3 through 11 (same lines, edit once)

### 13. Docs: wiki, swagger, and the follow-up issue
- Files: `docs/wiki/cross-cutting/exceptions.md` (change), `backend/app/api/docs/evaluation/improve_prompt.md`, `backend/app/api/docs/evaluation/create_evaluation.md`, `backend/app/api/docs/collections/update.md`, `backend/app/api/docs/onboarding/onboarding.md` (change if their stated codes went stale)
- Add an "Domain exceptions" section to `exceptions.md` pointing at `core/exceptions.py`, listing the taxonomy table above as the source of truth, and noting that the `KaapiError` handler is registered ahead of the generic one.
- The touched module pages (`llm-call.md`, `evaluations.md`, `knowledge-base.md`, `tenancy.md`, `platform.md`) hold no status codes, so they need no edit.
- Open a follow-up issue listing the out-of-scope sites found during the survey: `app/api/routes/auth.py:65,306,357`, `app/api/routes/fine_tuning.py:269`, `app/crud/api_key.py:97`, `app/crud/assessment/dataset.py:72`, `app/services/assessment/dataset.py:293`, `app/services/assessment/utils/export.py:329`, `app/services/response/jobs.py:42`.
- Depends on: steps 1 through 12

## Tests

Existing assertions that encode the current wrong codes and must flip:

| Test | Now | After |
|---|---|---|
| `app/tests/api/routes/test_evaluation.py:477` | 500 | per step 3/4 |
| `app/tests/api/routes/test_evaluation_fast.py:1271` | 500 | 404 (missing dataset) |
| `app/tests/api/routes/test_tts_evaluation.py:629` | 500 | 503 |
| `app/tests/api/routes/test_improve_prompt.py:517` | 500 | 503 |
| `app/tests/services/llm/test_jobs.py:101,2012` | 500 | 503 |
| `app/tests/services/stt_evaluations/test_audio.py:188` | 500 | 502 |
| `app/tests/services/guardrails/test_jobs.py:159` | 500 | 503 |
| `app/tests/api/routes/test_creds.py:260` (plus its docstring at line 240, which says "fails with 400") | 400 | 409 |

Out of scope and untouched: `app/tests/api/test_auth.py:49,313,412` and `app/tests/assessment/test_dataset.py:304`.

Assertions deliberately left alone: `app/tests/crud/test_credentials.py:273` ("Unsupported provider") and `:332` ("Missing required fields for langfuse") stay at 400, matching the Open Questions decision below.

New coverage:
- `app/tests/core/test_exceptions.py` (new): each `KaapiError` subclass maps to its status code through the registered handler, and `generic_error_handler` no longer returns `str(exc)` in the body.
- Evaluations: Langfuse dataset-not-found returns 404 and Langfuse unreachable returns 502, on both the batch path and the fast path. Mock the Langfuse client boundary (existing patch target `app.services.evaluations.fast.fetch_dataset_items` and `app.crud.evaluations.batch.fetch_dataset_items`, already used at `test_evaluation_fast.py:322` and `test_evaluation.py:1036`).
- Collections: deleting an already-deleted collection returns 409; duplicate name still returns 409 after the handler takes over from the manual translation.
- Documents: unsupported source format returns 422; unsupported transform pair still returns 400.
- Config: creating a config whose name already exists returns 409 rather than 500; a `ConfigBlob` that fails validation during partial-version merge returns 422 with the same body shape the request boundary produces.
- Credentials: duplicate provider credential returns 409; deleting a credential that does not exist returns 404; a KMS failure during create returns 502 and the response body does not contain the AWS error text.
- Onboarding: a KMS failure during onboarding returns 502; a concurrent onboard hitting `IntegrityError` at commit returns 409, and no partial org/project/user rows survive.

HTTP boundaries mocked: Langfuse client, OpenAI and Gemini SDK clients, object storage (`core/cloud/storage.py`), the AWS KMS client (`get_kms_client` in `core/security.py`), and the Celery `.delay`/`apply_async` call for the 503 paths.

`app/tests/services/credentials/` already exists and is the home for the KMS-failure cases.

## Open Questions

- **Documents 422 versus 400 for an unsupported file format (step 7).** Two live conventions disagree: `services/documents/helpers.py:64` returns 400, `services/evaluations/validators.py:84-97` returns 422 for the same class of problem. This plan picks 422 on the reading that the uploaded file is the unparseable payload, which also makes evaluations the larger surface that does not have to change. Reverse it if the reviewer reads an unsupported extension as a value problem rather than a shape problem, in which case `validators.py` moves to 400 instead and evaluations tests change.
- **503 versus 500 for a failed Celery enqueue (steps 4, 5, 8).** RabbitMQ being unable to accept a job is an availability problem the caller can retry, which 503 signals and 500 does not. Assumed, not stated in the issue.
- **Config immutability violations: 400 or 409 (step 9).** `version.py:201`, `version.py:330`, and `version.py:340` reject a change to an immutable field (`type`, `provider`) with 400. `.claude/conventions/error-handling.md` describes 409 as "request conflicts with current resource state", which is literally what this is. This plan leaves them at 400 to keep the config diff to the two defects that have no defence. Flip all three to 409 if the reviewer wants the taxonomy applied strictly.
- **Credentials validation: 400 or 422 (step 10).** `validate_provider_credentials` rejects both an unknown provider ("Unsupported provider", a value problem, 400) and a well-formed request missing provider-specific keys ("Missing required fields for langfuse", arguably a shape problem, 422) with the same 400. This plan keeps both at 400 rather than splitting one helper's output across two codes. Splitting them is defensible and would change `app/tests/crud/test_credentials.py:332`.
- **Onboarding bypasses credential validation.** `crud/onboarding.py:118-133` builds `Credential` rows inline instead of calling `set_creds_for_org`, so it never runs `validate_provider_credentials`. The same bad payload gets 400 through `POST /credentials` and is silently stored through `POST /onboard`. This is an error-message consistency defect, but fixing it changes onboarding behaviour rather than its status codes, so it is called out here rather than planned. Confirm whether it belongs in this PR or its own.
- **Frontend coordination is out of scope by decision.** Eleven status codes now change on endpoints the console calls, including `POST /credentials` and `POST /onboard`. The `APIResponse` body shape is untouched, so only code-branching logic can break. Worth a line in the PR description even though no investigation is planned.
