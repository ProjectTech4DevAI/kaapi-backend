# Postman Collection — Evaluation Section Changelog

Companion document for the Postman PR that lifts the Evaluation, STT Evaluation, and TTS Evaluation folders to a quality bar where a non-technical NGO operator can run the full lifecycle using only API calls.

**Scope:** Evaluation (text/Langfuse), STT Evaluation, TTS Evaluation.
**Out of scope:** Model Evaluation (untouched in this PR).

---

## Top-level additions

| Item | Location | Purpose |
|---|---|---|
| `Errors & Troubleshooting` | New top-level item (first in the collection) | Global error matrix, common traps, capability gaps, auth note |
| `Getting Started — Text Evaluation` | New first item inside `Evaluation` folder | End-to-end cURL walkthrough |
| `Getting Started — STT Evaluation` | New first item inside `STT Evaluation` folder | End-to-end cURL walkthrough |
| `Getting Started — TTS Evaluation` | New first item inside `TTS Evaluation` folder | End-to-end cURL walkthrough |

---

## Bugs fixed in the JSON itself

| # | Endpoint | Bug (before) | Fix (after) |
|---|---|---|---|
| 1 | `Evaluation` → `Get Dataset` | URL path `…/datasets/dataset_id={dataset_id}` (query string mashed into path) | URL path `…/datasets/:dataset_id` with `include_signed_url` moved to query; `dataset_id` declared in `variable` |
| 2 | `Evaluation` → `Delete Dataset` | Same URL path bug as #1 | Path `…/datasets/:dataset_id`, `dataset_id` declared in `variable`, stale query removed |
| 3 | `STT Evaluation` → `Upload audio file` | Form-data `file` field had `type: "text"`, `value: "string"` — request would not send as multipart file | `type: "file"`, `src: ""` (Postman shows native file picker) |
| 4 | `STT Evaluation` → `Start STT evaluation` | Body `"models": ["string"]` → guaranteed 422 | `"models": ["gemini-2.5-pro"]` |
| 5 | `TTS Evaluation` → `Start TTS evaluation` | Body `"models": ["string"]` → guaranteed 422 | `"models": ["gemini-2.5-pro-preview-tts"]` |

### Bonus polish to request bodies

| Endpoint | Before | After |
|---|---|---|
| STT `Create STT dataset` | Generic placeholders (`"name": "string"`, random `language_id: 8609`) | Realistic Hindi call-center example with `language_id: 5` |
| TTS `Create TTS dataset` | Generic `"text": "string"` | Realistic Project Tech4Dev support greeting |
| STT `Update STT sample` | `"language_id": null, "ground_truth": "string"` | `"language_id": 5, "ground_truth": "corrected transcription text"` |

---

## Per-endpoint description rewrites

Every endpoint's description was replaced. Below is the **delta** for each — what the user gains compared to the previous text.

### Evaluation folder (7 endpoints)

| Endpoint | Key additions |
|---|---|
| `Upload Dataset` | 1 MB max size, MIME whitelist, sample CSV inline, errors table (422 / 413 / 409 / 500), explicit field-by-field schema |
| `List Dataset` | Pagination params (`limit` 1–100 default 50, `offset` ≥0), response example |
| `Get Dataset` | Path-param/query-param split, 404 behaviour, signed URL 1-hr TTL |
| `Delete Dataset` | Cascade behaviour (CSV retained, past runs preserved, Langfuse not auto-deleted), 404 |
| `Evaluate` | Prereq link to `POST /configs`, full field reference, async lifecycle, errors (404 / 400 / 422 / 500), only-`openai`-provider trap |
| `List Evaluation Runs` | Pagination defaults, status enum with semantics |
| `Get Evaluation Run Status` | Already strong — now also documents both `400` query-combination errors, the soft trace-info error on non-completed runs, how to read `error_message` on failure |

### STT Evaluation folder (12 endpoints)

| Endpoint | Key additions |
|---|---|
| `Upload audio file` | Supported formats and 200 MB cap (already there) + sample cURL, response example, errors (400 / 500), pointer to next step |
| `List audio files` | Pagination, response shape, signed URL TTL |
| `Get audio file by ID` | Path param, query param, 404 |
| `Create STT dataset` | Languages lookup pointer, prereq, full schema, errors (400 file_ids / 400 dup name / 404 language / 422 empty) |
| `List STT datasets` | Pagination params with limits, response metadata |
| `Get STT dataset` | Query params (`include_samples`, `include_signed_url`, sample pagination), 404 |
| `Update STT sample` | Realistic example payload, null vs omitted semantics, 404 |
| `Start STT evaluation` | Single-model-supported call-out, errors (404 / 400 / 422 / 500), polling note, status enum |
| `List STT evaluation runs` | Filters (`dataset_id`, `status`), pagination, status enum |
| `Get STT evaluation run` | Polling note, all query params with defaults, failure-reading guidance |
| `Get STT result` | Full response schema, score keys (WER/CER/MER/WIL) explained, status enum, 404 |
| `Update human feedback` | Idempotency, null-clear semantics, 404 |

### TTS Evaluation folder (8 endpoints)

| Endpoint | Key additions |
|---|---|
| `Create TTS dataset` | 5000-char per-sample cap, languages lookup pointer, full JSON example, errors (400 / 404 / 422) |
| `List TTS datasets` | Pagination, `include_signed_url`, response metadata |
| `Get TTS dataset` | Path/query params, signed URL TTL, 404 |
| `Start TTS evaluation` | Default voice (`Kore`), default style prompt, single-model call-out, errors, polling note |
| `List TTS evaluation runs` | Filters, pagination, status enum |
| `Get TTS evaluation run` | Polling note, query params, failure-reading guidance |
| `Get TTS result` | Full response schema with `duration_seconds`/`size_bytes`/audio URLs, signed URL TTL, 404 |
| `Update human feedback` | **Critical** `status == SUCCESS` requirement (otherwise 400), null semantics, suggested score keys, illustrative-only score schema, 404 |

---

## Silent invariants now explicit somewhere

- 1 MB max CSV upload (text eval)
- 200 MB max audio (STT)
- 5000 char max per TTS sample
- Signed URL TTL = 1 hour
- Supported STT model: `gemini-2.5-pro` only
- Supported TTS model: `gemini-2.5-pro-preview-tts` only
- TTS feedback only on `status == SUCCESS`
- Dataset-name sanitization rule
- Run status lifecycle: `pending → processing → completed | failed`
- Per-result status (STT/TTS): `PENDING`, `SUCCESS`, `FAILED`

## Capability gaps surfaced (in Errors & Troubleshooting + each Getting Started)

- Cancel an in-flight run — not supported
- Retry a failed run — must POST again
- DELETE STT/TTS datasets — not supported (text eval has DELETE)
- Append samples to existing dataset — not supported
- Rename a dataset — not supported
- Bulk feedback — not supported
- CSV/JSON results export — not supported
- Completion webhook — not supported (poll only)

---

## How to apply this to your Postman fork

1. Open your forked Kaapi API'S collection in Postman Cloud.
2. **Import** → choose `Kaapi API'S.postman_collection.json` from this PR's branch → **Replace** (scoped to your fork only).
3. Walk the folders top-to-bottom, comparing the description of each endpoint against the deltas above. Adjust wording where you want a local change.
4. Verify the 5 bug fixes by sending each affected request as-shipped (no manual body edits required) against `api-staging.kaapi.ai`.
5. In your fork's menu, **Create pull request** → choose the parent collection → confirm.

## Source-of-truth files I cross-checked while writing the descriptions

- `backend/app/api/routes/evaluations/{evaluation,dataset}.py`
- `backend/app/api/routes/stt_evaluations/{router,evaluation,dataset,files,result}.py`
- `backend/app/api/routes/tts_evaluations/{router,evaluation,dataset,result}.py`
- `backend/app/services/evaluations/{validators,evaluation,dataset}.py`
- `backend/app/services/stt_evaluations/{audio,dataset,constants}.py`
- `backend/app/services/tts_evaluations/constants.py`
- `backend/app/models/{stt_evaluation,tts_evaluation,evaluation}.py`
- `backend/app/crud/{evaluations,stt_evaluations,tts_evaluations}/*.py`
