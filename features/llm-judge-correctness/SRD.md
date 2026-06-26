# Native LLM-as-a-Judge Correctness Score SRD

## Introduction & Purpose

This SRD defines a native, reference-based LLM-as-a-judge **correctness** score for
Kaapi fast evaluations. Each evaluated row already returns a *cosine similarity*
score (how close the generated answer is to the ground truth). This feature adds a
second, independent **correctness** score (0 to 1) plus a short reasoning string,
produced by an LLM inside Kaapi with no manual Langfuse configuration.

Today a correctness judgment is only available by hand-configuring a model-based
evaluator inside the third-party Langfuse dashboard, per project, outside the
platform. That setup is easily forgotten, the judging logic lives outside Kaapi
(unversioned and untailorable without engineering), and Kaapi keeps no record of
how a row was judged. Early users are NGO eval teams running fast evaluations on
their bots.

The feature produces, per evaluated row: a correctness score, a reasoning string,
both persisted by Kaapi and written to the row's Langfuse trace alongside the
existing similarity score, plus a per-project judge configuration the team can
create, view, update, and delete.

- **Phase 1 (this release):** automatic correctness judging on fast evaluations
  only; zero-config default (built-in prompt + fallback model); per-project judge
  config CRUD effective on the next run; both scores persisted and synced to
  Langfuse; reasoning per row.
- **Phase 2+ (deferred):** batch-mode judging; judge error/retry handling;
  confirmed performance budget and row-count guidance; rating-example entry UX and
  final score labels.

Intent: judging is automatic, zero-config, tailorable, explainable, and reversible,
owned inside Kaapi so eval owners never open Langfuse.

## Goals

- Correctness judging runs natively inside Kaapi with zero manual Langfuse setup.
- The judge is reference-based: it compares generated answer against ground truth
  (and the question) and returns a 0 to 1 score plus reasoning.
- Judging is automatic, with no new trigger, flag, or opt-in on the run path.
- It works out of the box from a built-in default prompt and a fallback model when
  no project config exists.
- A project team can tailor the judge (rating examples, model, model settings) and
  have it take effect on the next run with no deploy.
- Deleting a project's judge config reverts to default prompt + fallback model.

## Assumptions & Constraints

- **Out of scope:** batch-mode judging (fast only); retiring Langfuse (results keep
  syncing to it); robust judge error/retry handling. A failed or malformed judge
  call must not block the similarity score or the run; the row's correctness is left
  unscoreable and the run still completes.
- **Trigger:** judging runs inside the existing fast-eval pipeline. No new endpoint,
  request field, or flag on the run path.
- **Limits:** fast eval is capped at `EVAL_FAST_MAX_UNIQUE_ROWS` (10) unique rows;
  the judge adds one model call per evaluated row within that cap.
- **Per-row independence:** one row's judge failure must not fail sibling rows.
- **Reuse:** no new per-row table. Correctness rides the existing
  `EvaluationRun.score` record (per-trace `scores` list + `summary_scores`) and a new
  durable per-row map mirroring `per_item_scores`. One new table,
  `evaluation_judge_config`, holds per-project judge configuration. Judge model +
  sampling settings reuse `TextLLMParams` (the same shape the eval config uses); the
  results read path already aggregates a `Correctness` score, so surfacing needs no
  change.
- **Starting provider/model:** OpenAI, matching the fast-eval response/embedding
  path. The fallback model is a configurable default; project config may override
  the model and its settings.
- **Pricing:** the judge adds one LLM completion per row (paid), tracked under a
  `judge` stage in `EvaluationRun.cost`.

## Detailed Design (Execution Flow)

The judge slots into the fast-eval pipeline as a scoring step that runs after
similarity is computed, so a judge failure can never block the cosine score. The run
is marked complete with both summary scores; per-row scores are then written to
Langfuse, and a failed write is recoverable from the durable maps on resync.

### Judged fast-eval run

---

**>> PLACE IMAGE HERE: `assets/flow-a.png`, judged fast-eval run.**
System-level sequence: `User`, `Kaapi Backend`, `OpenAI`, `Langfuse` (arrows in and out).

---

Each judged row's correctness score and reasoning are appended to that trace's
`scores` list (reasoning carried in the score `comment`), and a `Correctness` summary
score is added to `EvaluationRun.score.summary_scores` next to `Cosine Similarity`.
The durable per-row correctness map is the source of truth for Langfuse resync.

### Tailor the judge (self-service config)

An admin creates, views, updates, or deletes the project's judge config through the
endpoints below; no diagram is needed for plain CRUD. Config is resolved once per run, at the judge step, so an in-flight run is unaffected
by a config change. An active (`deleted_at IS NULL`) config for the run's (org,
project) supplies its `llm_params`, `rating_examples`, and `prompt_override` (falling
back to the built-in prompt when unset); absence uses the fallback model and built-in
prompt with no rating examples.

## Functional Requirements (Testing)

| ID | What (user-facing behavior) | Acceptance criteria | Status |
|----|-----------------------------|---------------------|--------|
| FR-1 | Every evaluated row gets a correctness score | After a fast run completes, each scoreable row's persisted trace `scores` carries both a `Cosine Similarity` and a `Correctness` entry | Not Started |
| FR-2 | Correctness is 0 to 1 with reasoning | Each `Correctness` trace score has a numeric value in [0,1] and a non-empty reasoning `comment` | Not Started |
| FR-3 | Judging is automatic, no new trigger | Running the existing fast-eval endpoint with no new fields produces correctness scores; no request flag toggles it | Not Started |
| FR-4 | Zero-config default works | A project with no judge config still scores every scoreable row using the fallback model + built-in prompt | Not Started |
| FR-5 | Two summary scores persisted | `EvaluationRun.score.summary_scores` contains both a `Cosine Similarity` and a `Correctness` entry after completion | Not Started |
| FR-6 | Both scores on the Langfuse trace | Each evaluated row's Langfuse trace shows two distinctly-named scores | Not Started |
| FR-7 | Create judge config | `POST /evaluations/judge-config` returns 201 with the saved config (model, settings, rating examples) | Not Started |
| FR-8 | View judge config | `GET /evaluations/judge-config` returns the active config, or a default indicator when none exists | Not Started |
| FR-9 | Update takes effect next run | After `PATCH`, the next run's judge uses the updated model/settings/examples; an in-flight run is unaffected | Not Started |
| FR-10 | Delete reverts to default | After `DELETE`, the next run uses the fallback model + built-in prompt with no rating examples | Not Started |
| FR-11 | One active config per project | A second `POST` while an active config exists returns 409; never two active rows | Not Started |
| FR-12 | Per-row judge failure isolation | If a row's judge call fails or returns malformed output, that row's correctness is left unscoreable, its cosine score is unaffected, and the run still completes | Not Started |
| FR-13 | Judge cost tracked | After a run, `EvaluationRun.cost` includes a `judge` stage with token counts and USD | Not Started |
| FR-14 | Tenant isolation | A judge config for (org A, project A) is never resolved for any other (org, project) | Not Started |

## Endpoints

All judge-config endpoints are project-scoped under the evaluations router and
require the existing project permission. The fast-eval run trigger
(`POST /evaluations`) is unchanged: no new request fields, correctness surfaces
through the already-present run results.

### `GET /evaluations/judge-config`
Return the calling project's active judge configuration, or a default indicator.

**Response (config exists):**

```json
{
  "data": {
    "id": "b3f1c2d4-5e6a-7b8c-9d0e-1f2a3b4c5d6e",
    "llm_params": { "model": "gpt-4o", "temperature": 0.0 },
    "rating_examples": [
      {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "ground_truth": "Paris",
        "score": 1.0,
        "reasoning": "Exact, correct answer."
      }
    ],
    "prompt_override": null,
    "organization_id": 12,
    "project_id": 34,
    "inserted_at": "2026-06-25T10:00:00Z",
    "updated_at": "2026-06-25T10:00:00Z"
  }
}
```

**Response (no config, project on default):**

```json
{ "data": { "is_default": true, "llm_params": { "model": "gpt-4o-mini" } } }
```

### `POST /evaluations/judge-config`
Create the project's judge configuration.

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| llm_params | object (`TextLLMParams`) | No | fallback model | Judge model and sampling settings |
| rating_examples | array | No | `[]` | Example-graded rows guiding the judge |
| prompt_override | string \| null | No | null | Replaces the built-in judge prompt |

```json
{
  "llm_params": { "model": "gpt-4o", "temperature": 0.0 },
  "rating_examples": [
    { "question": "...", "answer": "...", "ground_truth": "...", "score": 1.0, "reasoning": "..." }
  ],
  "prompt_override": null
}
```

**Response:** `201 Created`, the saved config (same shape as `GET`).

**Error responses:**

| Status | Code | Message |
|--------|------|---------|
| 409 | judge_config_exists | "An active judge configuration already exists for this project." |
| 422 | invalid_judge_config | "rating_examples[0].score must be between 0 and 1." |

### `PATCH /evaluations/judge-config`
Partially update the project's judge configuration. Effective on the next run.

**Request body:** any subset of the `POST` fields.

```json
{ "llm_params": { "temperature": 0.2 } }
```

**Response:** `200 OK`, the updated config.

**Error responses:**

| Status | Code | Message |
|--------|------|---------|
| 404 | judge_config_not_found | "No active judge configuration for this project." |

### `DELETE /evaluations/judge-config`
Soft-delete the project's judge configuration; reverts the project to default.

**Response:** `204 No Content`.

**Error responses:**

| Status | Code | Message |
|--------|------|---------|
| 404 | judge_config_not_found | "No active judge configuration for this project." |

## Database Schema

One new table plus one reused table with a new column, so the schema is presented as
tables below (no diagram needed at this complexity).

### `evaluation_judge_config` (new)
Per-project judge configuration. At most one active row per (org, project); absence
means the platform default prompt + fallback model.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | UUID (PK) | NO | uuid4 | Unique identifier |
| llm_params | JSONB | NO | `{}` | `TextLLMParams` (judge model + sampling settings) |
| rating_examples | JSONB | NO | `[]` | Example-graded rows guiding the judge |
| prompt_override | TEXT | YES | NULL | Replaces the built-in judge prompt; NULL means default |
| organization_id | INTEGER (FK) | NO | n/a | Reference to the organization |
| project_id | INTEGER (FK) | NO | n/a | Reference to the project |
| deleted_at | TIMESTAMP | YES | NULL | Soft-delete marker; NULL means active |
| inserted_at | TIMESTAMP | NO | now() | Created timestamp |
| updated_at | TIMESTAMP | NO | now() | Last-updated timestamp |

**Constraints:**
- `uq_evaluation_judge_config_org_project_active`: UNIQUE on (`organization_id`, `project_id`) WHERE `deleted_at IS NULL`
- FK `organization_id` → `organization.id` (ON DELETE CASCADE)
- FK `project_id` → `project.id` (ON DELETE CASCADE)
- Index on (`project_id`) WHERE `deleted_at IS NULL` for run-time resolution

### `evaluation_run` (existing, reused)
Correctness rides the existing score columns; one new column holds the durable
correctness map.

| Column | Type | Now carries |
|--------|------|-------------|
| score | JSONB | `summary_scores` gains a `Correctness` entry; each `traces[].scores` gains a `Correctness` score with value + reasoning `comment` |
| per_item_correctness | JSONB (YES, default NULL) | New column mirroring `per_item_scores`: durable `{trace_id: correctness}` map, source of truth for Langfuse resync |
| unscoreable | JSONB | Reused for rows the judge could not score, alongside existing cosine reasons |
| cost | JSONB | Gains a `judge` stage (tokens + USD) |

**Backfill plan:** `per_item_correctness` is nullable with default NULL; pre-feature
runs need no backfill (they carry no correctness data).

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| EVAL_JUDGE_FALLBACK_MODEL | str | `gpt-4o-mini` | Judge model when a project has no config |
| EVAL_JUDGE_DEFAULT_TEMPERATURE | float | `0.0` | Default judge temperature when not overridden |

The built-in default judge prompt and the `Correctness` score name (mirroring
`COSINE_SCORE_NAME`) are owned in Kaapi code as constants.

## Design Decisions / Known Limitations

- **No new per-row table.** Correctness reuses the per-trace `scores` list and the
  `EvaluationRun.score` summary, matching how cosine per-row data is already
  persisted to S3 and synced to Langfuse; reasoning is stored in the score `comment`.
- **Dedicated `evaluation_judge_config` table** rather than the versioned
  `config`/`config_version` flow. The judge config needs plain CRUD with
  revert-on-delete, so a soft-deletable table keeps "next run reads the latest active
  row" trivial.
- **Reuses `TextLLMParams`** for the judge model + sampling settings rather than a
  parallel config shape, so the judge config validates and maps like the eval config.
- **`per_item_correctness` as a separate column** (vs folding into `per_item_scores`)
  keeps the two score families independently resyncable and mirrors the existing
  cosine map exactly.
- **Judge runs after similarity** so cosine is already computed and a judge failure
  can never block it.
- **Known limitation, error/retry (deferred):** a failed judge row is left
  unscoreable with no judge-specific retry; defined retry/backoff is Phase 2.
- **Open (from PRD):** judge error/retry behavior, performance budget and supported
  row count, the rating-example entry format, and the final user-facing score labels.
```
