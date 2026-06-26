# AI-Assisted Prompt Improvement from Evaluation Results SRD

## Introduction & Purpose

This SRD defines an AI-assisted prompt-improvement capability for Kaapi: from a single
completed evaluation run, a user generates a new, improved prompt iteration grounded in
that evaluation's own results — the full trace file the run already produced and stored in
S3, covering every question, every score, and every category.

Today the loop is open. An evaluation produces exactly the evidence needed to improve a
prompt (per-question scores, categories, the assistant's actual answers vs. ground truth),
but turning that evidence into a better prompt is entirely manual: a user reads the failing
questions, spots patterns, and hand-writes a new prompt. That work is slow, needs
prompt-engineering skill, and does not scale across many configurations or frequent
evaluation cycles. This feature closes the loop — one explicit action turns a finished
evaluation into a concrete, improved prompt iteration.

Rather than selecting "weak" questions and categories in code, the feature hands the
run's **entire** evaluation trace file to Claude via the Anthropic Files API and lets the
model find the patterns itself. The trace file carries, per question, the assistant's
answer, the ground-truth answer, the category, and **all** of the run's scores — the
built-in Cosine Similarity score plus any number of Langfuse "LLM-as-judge" scores. Claude
reads every metric and every category together to understand how the generated answers
performed against ground truth, then rewrites the system prompt accordingly. No threshold,
no metric selection, no caps.

What the feature produces, at minimum:
- A new `config_version` for the evaluated configuration, identical to the evaluated
  version except for the prompt (`completion.params.instructions`) text.
- Provenance on that version, recorded entirely in the existing `commit_message` field
  (no schema changes): an `[AI Generated]` marker, the source evaluation run id, and a
  short rationale describing what the improvement targeted.

Phasing:
- **Phase 1 (this SRD):** Synchronous, user-initiated prompt improvement from one
  completed evaluation. No request parameters — the service reads the run's stored trace
  file from S3 and hands the whole file to the LLM, which analyzes all scores and all
  categories. Prompt-only change; prior iterations preserved.
- **Phase 2+ (deferred):** Knowledge-base staleness diagnosis; async/background execution
  with completion notification; one-click re-evaluation of the new iteration; generating
  and comparing multiple candidate prompts at once.

Quality bar: every generated iteration is **traceable** (to its source evaluation),
**auditable** (the rationale is recorded), and **non-destructive** (added alongside, never
overwriting prior iterations).

## Goals

- Generate an improved prompt iteration from a completed evaluation run in a single,
  explicit user action — no automatic triggering and no request parameters.
- Ground the improvement in the run's complete evaluation evidence: the entire stored trace
  file (every question, its assistant answer vs. ground truth, its category, and **all** of
  its scores), handed to the LLM so the model — not hand-tuned code heuristics — identifies
  the low-performing answers and the patterns across them.
- Use every metric the run recorded: the built-in Cosine Similarity score plus any number
  of Langfuse LLM-as-judge scores, all read together to judge how generated answers
  performed against ground truth. Likewise, every category present in the run is available
  to the analysis.
- Preserve everything about the configuration except the prompt — same model, knowledge
  base, and other `config_blob` settings — so the change is an apples-to-apples prompt swap.
- Persist the new prompt as the next `config_version` via the existing version-creation
  path — no schema changes. Mark it AI-generated, traceable to the source evaluation, and
  carry the rationale, all in the version's `commit_message`. All prior versions preserved.

## Assumptions & Constraints

- **Out of scope:** No automatic triggering on evaluation completion. No knowledge-base
  diagnosis or changes. No change to model, knowledge base, or any other `config_blob`
  setting — only `completion.params.instructions` changes. No automatic re-evaluation of
  the new iteration. No async/background processing or completion notifications in v1.
- **Synchronous execution:** The endpoint downloads the trace file, makes the LLM call, and
  returns the new version inline. No Celery task in v1 (a faster async version is a likely
  follow-up — see Design Decisions).
- **No request parameters:** The endpoint takes only the path `evaluation_id`. There is no
  metric to choose and no threshold to set — the model reads every score and every category
  in the trace file.
- **Eligibility:** Improvement can run only against an `evaluation_run` with
  `status = "completed"`. Any other status is rejected (409, `evaluation_not_completed`).
- **Trace file is required:** The run must carry a non-empty `score_trace_url` (the S3
  location of its stored trace file, written by the evaluation pipeline). If it is empty or
  null, the request is rejected (422, `traces_not_available`) — there is nothing to analyze.
- **Analysis is delegated to the LLM, not done in code:** The service does not group,
  filter, score-select, or cap anything. It downloads the whole trace file and uploads it to
  the Anthropic Files API as a document attached to the improvement request. The model
  identifies the low-performing answers (low scores or large divergence from ground truth)
  across the full result set and rewrites the prompt to address them.
- **All metrics, all categories:** Each trace in the file carries a `scores` list of
  `{name, value, unscoreable}` objects — the built-in Cosine Similarity score plus any
  number of Langfuse LLM-as-judge scores whose names are not known ahead of time. The model
  weighs all of them, alongside every `category` present, to understand performance against
  ground truth. No metric is privileged and no category count is bounded.
- **Reuse:** No new tables and no schema changes. Reuse `evaluation_run` (read-only) and
  `config_version` (one new row, existing columns only). Provenance is recorded in the new
  row's `commit_message`. Reuse the existing version-creation CRUD path
  (`create_or_raise`) so version numbering and soft-delete semantics stay consistent; it
  deep-merges the prompt-only change onto the latest version's `config_blob`.
- **Multi-tenancy:** `evaluation_run` carries `organization_id` + `project_id`; `config`
  and `config_version` are project-scoped (`project_id`, no `organization_id`). The run's
  `config_id` must resolve to a config in the caller's project, else the request is rejected.
- **Trace download via cloud storage:** The trace file is fetched through the project's
  configured cloud storage (`get_cloud_storage(...).get(url)`). A storage failure (missing
  object, bad credentials) is mapped to `502 trace_download_failed`.
- **Anthropic Files API:** The trace bytes are uploaded with the `files-api-2025-04-14`
  beta and attached as a `document` content block (uploaded as `text/plain`, which the Files
  API accepts reliably where `application/json` is sometimes rejected). The uploaded file is
  always deleted after the call, even on error.
- **Pricing:** The improvement makes one LLM call per invocation (paid). The whole trace
  file is sent as a document, so cost scales with the run's size.
- **Starting provider/model:** Claude (default `claude-opus-4-8`), configurable through
  settings (see Configuration).
- **Credentials:** The feature uses a single platform-owned Anthropic key
  (`ANTHROPIC_API_KEY` env var) shared by every org/project, so it works
  without per-project credentials. If the key is unset, the request fails with
  `502 prompt_generation_failed`.

## Detailed Design (Execution Flow)

### Improvement Flow (synchronous)

1. **Request.** User calls `POST /evaluations/{evaluation_id}/improve-prompt` with no body.
   `Permission.REQUIRE_PROJECT` resolves the caller's org + project.
2. **Load & validate the run.** Fetch the `evaluation_run` by id, scoped to the caller's
   `organization_id` + `project_id`. Reject if missing (404, `evaluation_not_found`) or if
   `status != "completed"` (409, `evaluation_not_completed`).
3. **Require the trace file.** If the run's `score_trace_url` is empty or null, reject (422,
   `traces_not_available`) — there is no stored result set to analyze.
4. **Resolve the evaluated config version.** Read `config_id` + `config_version` off the
   run. Load that `config_version` (and parent `config`, scoped to the caller's project).
   Reject if either is missing or soft-deleted (409, `source_config_unavailable`).
5. **Read the source context.** Extract the current prompt from the source version's
   `config_blob.completion.params.instructions`, and (optionally) the model name from
   `config_blob.completion.params.model` to give the LLM context.
6. **Download the trace file.** Fetch the trace bytes from S3 via the project's cloud
   storage (`get_cloud_storage(...).get(score_trace_url)`). On any storage error, reject
   (502, `trace_download_failed`).
7. **Draft the improved prompt.** Verify the platform Anthropic key is set (else 502,
   `prompt_generation_failed`). Upload the trace bytes to the Anthropic Files API
   (`files-api-2025-04-14` beta) as `text/plain`. Call Claude with a single user message
   carrying (a) a text block — the current prompt, the model context, and the task: identify
   answers that scored low or diverge from ground truth across *all* scores and categories in
   the file, then rewrite the system prompt to address them while preserving what works — and
   (b) a `document` content block referencing the uploaded file. The model returns a JSON
   object with `improved_instructions` (the full rewritten prompt) and `rationale` (one
   short paragraph). The uploaded file is deleted afterward, even on error. SDK/transport
   failures and unparseable or incomplete responses all map to 502, `prompt_generation_failed`.
8. **Compose the prompt-only change.** Pass a partial `config_blob` containing only
   `completion.params.instructions = improved_instructions` to the version-creation path,
   which deep-merges it onto the latest version's blob. Every other field (model,
   `knowledge_base_ids`, temperature, etc.) is preserved.
9. **Persist the new version.** Create a `config_version` via `create_or_raise`:
   `version = latest active version + 1`, the merged `config_blob`, and `commit_message` =
   the provenance string (truncated to 512 chars):
   `"[AI Generated] (source_evaluation_run_id={evaluation_id}) {rationale}"`. No new columns
   are written.
10. **Respond.** Return the new version (id, version number, `commit_message`, `config_blob`).

> Sequence Flow Diagram: _TBD_

## Functional Requirements (Testing)

| ID | What (user-facing behavior) | Acceptance criteria | Status |
|----|-----------------------------|---------------------|--------|
| FR-1 | Generate an improved prompt from a completed evaluation | `POST /evaluations/{id}/improve-prompt` (no body) on a `completed` run with a trace file returns 201 with a new `config_version` whose `version` = previous latest + 1 | Not Started |
| FR-2 | Reject improvement on a non-completed run | Run with status `pending`/`processing`/`failed` returns 409 `evaluation_not_completed` | Not Started |
| FR-3 | Reject when the trace file is missing | A completed run with empty/null `score_trace_url` returns 422 `traces_not_available` and creates no version | Not Started |
| FR-4 | Reject when source config/version is unavailable | If the run's `config_version` (or parent `config`) is missing or soft-deleted, returns 409 `source_config_unavailable` | Not Started |
| FR-5 | Full trace file drives the analysis | The run's entire trace file is downloaded from `score_trace_url` and sent to the LLM as a document — no in-code question selection, scoring, or capping | Not Started |
| FR-6 | All scores and categories used | The model receives every trace's `scores` (Cosine Similarity plus any Langfuse scores) and `category`; no metric is chosen by the caller and no category count is bounded | Not Started |
| FR-7 | Trace download failure surfaced | If the trace file cannot be retrieved from storage, returns 502 `trace_download_failed` and creates no version | Not Started |
| FR-8 | Uploaded file cleaned up | The trace file uploaded to the Anthropic Files API is deleted after the LLM call, including when the call errors | Not Started |
| FR-9 | Prompt-only change | New version's `config_blob` equals the prior blob except `completion.params.instructions`; model, `knowledge_base_ids`, and all other params are unchanged | Not Started |
| FR-10 | New version marked AI-generated | New `config_version.commit_message` begins with the `[AI Generated]` marker | Not Started |
| FR-11 | New version traceable to source evaluation | New `config_version.commit_message` contains `source_evaluation_run_id={the evaluation id used}` | Not Started |
| FR-12 | Rationale recorded | New version's `commit_message` contains the LLM's improvement rationale | Not Started |
| FR-13 | Prior iterations preserved | All pre-existing `config_version` rows for the config are unchanged and still retrievable after generation | Not Started |
| FR-14 | LLM failure surfaced | A failed or unparseable LLM call (or an unset platform Anthropic key) returns 502 `prompt_generation_failed` and creates no version | Not Started |
| FR-15 | Tenant isolation | A run not belonging to the caller's org+project returns 404; a run whose config is outside the caller's project returns 409 | Not Started |
| FR-16 | Repeatable iteration | Running improvement again (e.g. after re-evaluating) creates a further `config_version` at the next version number | Not Started |

## Endpoints

### `POST /evaluations/{evaluation_id}/improve-prompt`
Generate a new, AI-improved prompt iteration for the configuration evaluated by this run.
Runs synchronously and returns the new `config_version`.

**Path parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| evaluation_id | integer | Yes | The completed `evaluation_run` to improve from |

**Request body:** None. The service reads the run's stored trace file and hands it to the
LLM; there is no metric or threshold to supply.

**Response (201):**

```json
{
  "success": true,
  "data": {
    "id": "9f1c0b8e-3a2d-4c77-9b6a-2e5f1d7c4a10",
    "config_id": "3d2b1a09-8765-4321-fedc-ba9876543210",
    "version": 4,
    "commit_message": "[AI Generated] (source_evaluation_run_id=812) Tightened answer scoping and added explicit grounding instructions to address the low-scoring answers, concentrated in the 'Eligibility' and 'Payments' categories.",
    "config_blob": { "...": "merged blob — prompt-only change" },
    "inserted_at": "2026-06-24T10:15:00Z",
    "updated_at": "2026-06-24T10:15:00Z"
  }
}
```

**Error responses:**

| Status | Code | When |
|--------|------|------|
| 404 | evaluation_not_found | No `evaluation_run` with this id in the caller's org+project |
| 409 | evaluation_not_completed | Run status is not `completed` |
| 409 | source_config_unavailable | The run's `config`/`config_version` is missing, soft-deleted, or outside the caller's project |
| 422 | traces_not_available | The run has no `score_trace_url`; the trace file is required |
| 502 | trace_download_failed | The trace file could not be retrieved from storage |
| 502 | prompt_generation_failed | The platform Anthropic key is unset, or the LLM call failed / returned an unusable result |

## Database Schema

**No schema changes.** No new tables, no new columns, no new enum types, and no migration.
The feature reads `evaluation_run` and adds one new `config_version` row using the existing
columns only.

### `config_version` (existing — unchanged)
Stores prompt iterations for a configuration. The AI-generated iteration is a normal new
row created through the existing version-creation path. Its AI provenance — the
`[AI Generated]` marker and the source evaluation run id — plus the rationale live in the
existing `commit_message` field (truncated to its 512-char limit). Nothing about prior rows
changes.

### `evaluation_run` (existing — read-only)
No changes. Read fields: `status`, `config_id`, `config_version`, `score_trace_url`,
`organization_id`, `project_id`. The analysis runs off the trace file at `score_trace_url`
(downloaded from S3), not the `score` JSONB column. The trace file is a JSON array of
trace objects, each with `question`, `ground_truth_answer`, `llm_answer`, `category`, and a
`scores` list of `{name, value, unscoreable}` (the built-in Cosine Similarity score plus
any Langfuse scores).

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| ANTHROPIC_API_KEY | str | `""` | Platform-owned Anthropic key shared by all orgs/projects; required for the feature |
| PROMPT_IMPROVEMENT_MODEL | str | `claude-opus-4-8` | LLM used to draft the improved prompt |

## Design Decisions / Known Limitations

- **Synchronous in v1:** The PRD calls for no async/notifications in v1, so the endpoint
  downloads the trace file and makes the single LLM call inline. Risk: a slow LLM call (the
  whole trace file is sent as a document) holds the request open. Async via Celery is the
  planned follow-up; the endpoint contract can stay the same with a `202 + polling` variant
  later.
- **Analysis delegated to the LLM, not done in code:** Instead of selecting "weak" questions
  by a consistency ratio, capping the set, and computing per-category averages in Python,
  the service ships the run's entire trace file to Claude via the Files API and lets the
  model find the patterns. This removes every tuning knob (consistency ratio, weak-question
  cap, weak-category cap, metric choice, threshold) and the associated heuristics. The model
  sees all answers, all scores, and all categories at once, so it can weigh signals a
  single-metric threshold would miss and is not blinded by truncation. Trade-off: token cost
  scales with the run's size, and the analysis is non-deterministic — re-running may produce
  a different rewrite.
- **Whole-file input via the Files API:** The trace bytes are uploaded once and attached as
  a `document` block (as `text/plain`, which the Files API accepts where `application/json`
  is sometimes rejected). The upload is always deleted in a `finally` block so transient or
  failed runs don't leak files. This keeps the message payload small (a file reference, not
  inlined JSON) and lets large result sets through.
- **New version reuses the existing version-creation path** rather than writing
  `config_version` directly, so version numbering, soft-delete, and validation stay
  consistent with human-authored versions. The service passes only the changed
  `completion.params.instructions`; `create_or_raise` deep-merges it onto the latest
  version's `config_blob`, guaranteeing the prompt-only change. The AI iteration is identical
  in shape to a human-authored one — only the `commit_message` distinguishes it.
- **Provenance in `commit_message`, not dedicated columns:** Provenance ("AI Generated",
  the source run, and the rationale) is folded into the existing `commit_message` rather
  than first-class columns. This keeps the change schema-free — no migration, no enum, no
  new fields — and the AI iteration stays a plain `config_version` the UI already knows how
  to render. Trade-off: provenance is not independently queryable or filterable (e.g. "show
  all AI versions" or "versions from run 812" would require text search on `commit_message`);
  promoting it to columns is a deferred follow-up if those queries become needed.
- **All metrics, no privileged score:** A run carries one built-in cosine score plus any
  number of Langfuse LLM-as-judge scores. Rather than asking the caller to pick one and set
  a numeric threshold, the model reads every score (and the `unscoreable` flag) on every
  trace and judges performance against ground truth holistically. This also sidesteps the
  numeric-vs-categorical distinction — the model interprets each score in context.
- **Known limitation — prompt-only:** Poor answers caused by stale/irrelevant knowledge-base
  content cannot be fixed here; the prompt rewrite may not move the metric if retrieval is
  the real problem. KB diagnosis is deferred to Phase 2.
- **Known limitation — no auto re-evaluation:** The lift from the new prompt is not measured
  automatically; the user must re-run an evaluation to see whether quality improved.
