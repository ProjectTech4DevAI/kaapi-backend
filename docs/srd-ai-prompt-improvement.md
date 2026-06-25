# AI-Assisted Prompt Improvement from Evaluation Results SRD

## Introduction & Purpose

This SRD defines an AI-assisted prompt-improvement capability for Kaapi: from a single
completed evaluation run, a user generates a new, improved prompt iteration grounded in
that evaluation's own evidence — the questions that scored low *consistently* and the
question categories that underperform.

Today the loop is open. An evaluation produces exactly the evidence needed to improve a
prompt (per-question scores, weak categories, the assistant's actual answers vs. ground
truth), but turning that evidence into a better prompt is entirely manual: a user reads
the failing questions, spots patterns, and hand-writes a new prompt. That work is slow,
needs prompt-engineering skill, and does not scale across many configurations or frequent
evaluation cycles. This feature closes the loop — one explicit action turns a finished
evaluation into a concrete, improved prompt iteration.

What the feature produces, at minimum:
- A new `config_version` for the evaluated configuration, identical to the evaluated
  version except for the prompt (`completion.params.instructions`) text.
- Provenance on that version: marked AI-generated, linked to the source evaluation run,
  and carrying a short rationale describing what the improvement targeted.

Phasing:
- **Phase 1 (this SRD):** Synchronous, user-initiated prompt improvement from one
  completed evaluation. User picks the quality metric and the low-score threshold. Weak
  questions are selected by *consistent* low performance; underperforming categories feed
  the analysis. Prompt-only change; prior iterations preserved.
- **Phase 2+ (deferred):** Knowledge-base staleness diagnosis; async/background execution
  with completion notification; one-click re-evaluation of the new iteration; generating
  and comparing multiple candidate prompts at once.

Quality bar: every generated iteration is **traceable** (to its source evaluation),
**auditable** (rationale + the metric/threshold used are recorded), and **non-destructive**
(added alongside, never overwriting prior iterations).

## Goals

- Generate an improved prompt iteration from a completed evaluation run in a single,
  explicit user action — no automatic triggering.
- Ground the improvement in evaluation evidence: questions that scored below the chosen
  threshold *consistently* across their repetitions, plus underperforming categories.
- Let the user control the analysis: choose the quality metric (cosine similarity or
  correctness) and the low-score threshold.
- Preserve everything about the configuration except the prompt — same model, knowledge
  base, and other `config_blob` settings — so the change is an apples-to-apples prompt swap.
- Persist the new prompt as the next `config_version`, clearly marked AI-generated,
  traceable to the source evaluation, with a short rationale. All prior versions preserved.

## Assumptions & Constraints

- **Out of scope:** No automatic triggering on evaluation completion. No knowledge-base
  diagnosis or changes. No change to model, knowledge base, or any other `config_blob`
  setting — only `completion.params.instructions` changes. No automatic re-evaluation of
  the new iteration. No async/background processing or completion notifications in v1.
- **Synchronous execution:** The endpoint runs the analysis and the LLM draft inline and
  returns the new version in the response. No Celery task in v1 (a faster async version is
  a likely follow-up — see Design Decisions).
- **Eligibility:** Improvement can run only against an `evaluation_run` with
  `status = "completed"`. Any other status is rejected.
- **"Consistently low" definition:** Traces are grouped by `question_id`. A question is
  selected as *weak* when the fraction of its repetitions scoring below `threshold` is at
  least `MIN_CONSISTENCY_RATIO` (default 0.5). This filters one-off bad answers. Questions
  with no scoreable repetitions for the chosen metric are ignored. (Repetition count comes
  from the dataset's `duplication_factor`, 1–5; with `duplication_factor = 1` a single
  sub-threshold score qualifies.)
- **Limits / caps:** At most `MAX_WEAK_QUESTIONS` (default 50) weak questions — the lowest
  scoring first — are sent to the LLM, to bound token cost and prompt size. At most
  `MAX_WEAK_CATEGORIES` (default 10) underperforming categories are included. When the cap
  truncates the set, the rationale records that the analysis was truncated.
- **Metric availability:** The chosen metric must be present in the run's recorded scores.
  Cosine similarity lives in `EvaluationRun.per_item_scores` / `summary_scores`;
  correctness is a `summary_scores` entry whose name matches "correctness". If the chosen
  metric has no scores in this run, the request is rejected.
- **Reuse:** No new tables. Reuse `evaluation_run` (read-only) and `config_version`
  (one new row + three new provenance columns). Reuse the existing version-creation CRUD
  path so version numbering and soft-delete semantics stay consistent.
- **Multi-tenancy:** `evaluation_run` carries `organization_id` + `project_id`; `config`
  and `config_version` are project-scoped (`project_id`, no `organization_id`). The run's
  `config_id` must resolve to a config in the caller's project, else the request is rejected.
- **Pricing:** The improvement makes one LLM call per invocation (paid). Bounded by the
  weak-question/category caps above.
- **Starting provider/model:** Claude (default `claude-opus-4-8`) via the existing provider
  abstraction, configurable through settings (see Configuration).

## Detailed Design (Execution Flow)

### Improvement Flow (synchronous)

1. **Request.** User calls `POST /evaluations/{evaluation_id}/improve-prompt` with `metric`
   and `threshold`. `Permission.REQUIRE_PROJECT` resolves the caller's org + project.
2. **Load & validate the run.** Fetch the `evaluation_run` by id, scoped to the caller's
   `organization_id` + `project_id`. Reject if missing (404) or if `status != "completed"`
   (409, `evaluation_not_completed`).
3. **Resolve the evaluated config version.** Read `config_id` + `config_version` off the
   run. Load that `config_version` (and parent `config`, scoped to the caller's project).
   Reject if either is missing or soft-deleted (409, `source_config_unavailable`).
4. **Verify metric presence.** Confirm the chosen `metric` exists in the run's
   `summary_scores` / `per_item_scores`. If absent, reject (422, `metric_not_available`).
5. **Select weak questions.** Group `score.traces` by `question_id`. For each group,
   compute the fraction of repetitions whose chosen-metric value is below `threshold`. Keep
   groups where that fraction ≥ `MIN_CONSISTENCY_RATIO`. Sort by mean sub-threshold severity
   (lowest mean first); truncate to `MAX_WEAK_QUESTIONS`. For each kept question, collect
   `{question, llm_answer, ground_truth_answer, category, mean_score}`.
6. **Select underperforming categories.** From `score.category_metrics`, keep categories
   whose chosen-metric average (`avg_cosine` or `avg_correctness`) is below `threshold`;
   sort ascending; truncate to `MAX_WEAK_CATEGORIES`.
7. **Guard: nothing to improve.** If both the weak-question and weak-category sets are
   empty, reject (422, `no_weak_signals`) — there is no evidence to improve against.
8. **Draft the improved prompt.** Read the current prompt from the source version's
   `config_blob.completion.params.instructions`. Build an improvement request to the LLM
   containing: the current prompt, the weak questions (with answers vs. ground truth and
   scores), the underperforming categories, and the metric/threshold context. The LLM
   returns the rewritten instructions plus a one-paragraph rationale of what it targeted.
9. **Compose the new `config_blob`.** Deep-copy the source version's `config_blob`; replace
   only `completion.params.instructions` with the new text. Every other field (model,
   `knowledge_base_ids`, temperature, etc.) is preserved byte-for-byte.
10. **Persist the new version.** Create a `config_version` via the existing version path:
    `version = latest active version + 1`, `config_blob` = the composed blob,
    `commit_message` = the rationale (truncated to 512 chars), and the new provenance
    columns: `source = AI_GENERATED`, `source_evaluation_run_id = evaluation_id`,
    `generation_metadata` = `{metric, threshold, weak_question_count, weak_category_count,
    truncated, model}`.
11. **Respond.** Return the new version (id, version number, provenance, rationale).

> Sequence Flow Diagram: _TBD_

## Functional Requirements (Testing)

| ID | What (user-facing behavior) | Acceptance criteria | Status |
|----|-----------------------------|---------------------|--------|
| FR-1 | Generate an improved prompt from a completed evaluation | `POST /evaluations/{id}/improve-prompt` on a `completed` run returns 201 with a new `config_version` whose `version` = previous latest + 1 | Not Started |
| FR-2 | Reject improvement on a non-completed run | Run with status `pending`/`processing`/`failed` returns 409 `evaluation_not_completed` | Not Started |
| FR-3 | Reject when source config/version is unavailable | If the run's `config_version` (or parent `config`) is missing or soft-deleted, returns 409 `source_config_unavailable` | Not Started |
| FR-4 | User selects the quality metric | `metric` accepts `cosine_similarity` or `correctness`; any other value returns 422 `invalid_metric` | Not Started |
| FR-5 | Reject a metric absent from this run | Choosing `correctness` on a run with no correctness scores returns 422 `metric_not_available` | Not Started |
| FR-6 | User selects the low-score threshold | `threshold` accepted in [0, 1]; values outside the range return 422 `invalid_threshold` | Not Started |
| FR-7 | Weak questions chosen by *consistent* low performance | A question whose sub-threshold repetition fraction ≥ `MIN_CONSISTENCY_RATIO` is selected; a question with one low repetition out of many (below ratio) is excluded | Not Started |
| FR-8 | Underperforming categories feed the analysis | Categories whose chosen-metric average < `threshold` are passed to the LLM; this is reflected in `generation_metadata.weak_category_count` | Not Started |
| FR-9 | Nothing-to-improve guard | When no weak questions and no weak categories are found, returns 422 `no_weak_signals` and creates no version | Not Started |
| FR-10 | Prompt-only change | New version's `config_blob` equals the source blob except `completion.params.instructions`; model, `knowledge_base_ids`, and all other params are unchanged | Not Started |
| FR-11 | New version marked AI-generated | New `config_version.source = AI_GENERATED` | Not Started |
| FR-12 | New version traceable to source evaluation | New `config_version.source_evaluation_run_id` = the evaluation id used | Not Started |
| FR-13 | Rationale recorded | New version's `commit_message` contains the improvement rationale; `generation_metadata` records metric, threshold, and counts | Not Started |
| FR-14 | Prior iterations preserved | All pre-existing `config_version` rows for the config are unchanged and still retrievable after generation | Not Started |
| FR-15 | Caps enforced and disclosed | With > `MAX_WEAK_QUESTIONS` weak questions, only the lowest-scoring `MAX_WEAK_QUESTIONS` are used and `generation_metadata.truncated = true` | Not Started |
| FR-16 | Tenant isolation | A run not belonging to the caller's org+project returns 404; a run whose config is outside the caller's project returns 409 | Not Started |
| FR-17 | Repeatable iteration | Running improvement again (e.g. after re-evaluating) creates a further `config_version` at the next version number | Not Started |

## Endpoints

### `POST /evaluations/{evaluation_id}/improve-prompt`
Generate a new, AI-improved prompt iteration for the configuration evaluated by this run.
Runs synchronously and returns the new `config_version`.

**Path parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| evaluation_id | integer | Yes | The completed `evaluation_run` to improve from |

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| metric | string (enum) | Yes | — | Quality metric to judge "low" on: `cosine_similarity` or `correctness` |
| threshold | number | Yes | — | Score in [0, 1]; a repetition below this is "low" |

```json
{
  "metric": "cosine_similarity",
  "threshold": 0.7
}
```

**Response (201):**

```json
{
  "success": true,
  "data": {
    "id": "9f1c0b8e-3a2d-4c77-9b6a-2e5f1d7c4a10",
    "config_id": "3d2b1a09-8765-4321-fedc-ba9876543210",
    "version": 4,
    "source": "ai_generated",
    "source_evaluation_run_id": 812,
    "commit_message": "Tightened answer scoping and added explicit grounding instructions to address 7 consistently low questions, concentrated in the 'Eligibility' and 'Payments' categories.",
    "generation_metadata": {
      "metric": "cosine_similarity",
      "threshold": 0.7,
      "weak_question_count": 7,
      "weak_category_count": 2,
      "truncated": false,
      "model": "claude-opus-4-8"
    },
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
| 422 | invalid_metric | `metric` is not one of the supported values |
| 422 | metric_not_available | The chosen metric has no scores recorded in this run |
| 422 | invalid_threshold | `threshold` is outside [0, 1] |
| 422 | no_weak_signals | No consistently-low questions and no underperforming categories found |
| 502 | prompt_generation_failed | The LLM call failed or returned an unusable result |

## Database Schema

No new tables. The feature reads `evaluation_run` and adds one row plus three columns to
`config_version`.

### `config_version` (existing — added columns)
Stores prompt iterations for a configuration. New columns capture AI provenance.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| source | ENUM(`config_version_source`) | NO | `human_authored` | Origin of the version: `human_authored` or `ai_generated` |
| source_evaluation_run_id | INTEGER | YES | NULL | FK to the `evaluation_run` this version was generated from (NULL for human-authored) |
| generation_metadata | JSONB | YES | NULL | AI-generation context: `metric`, `threshold`, `weak_question_count`, `weak_category_count`, `truncated`, `model` |

**Constraints:**
- New enum type `config_version_source` with values (`human_authored`, `ai_generated`).
- FK `source_evaluation_run_id` → `evaluation_run.id` (`ON DELETE SET NULL`, so deleting a
  run does not destroy the prompt it produced; provenance simply detaches).
- Index on `source_evaluation_run_id` (FK lookup — "which versions came from this run").
- **Backfill:** existing rows set `source = 'human_authored'`,
  `source_evaluation_run_id = NULL`, `generation_metadata = NULL`. `source` is non-null with
  a server default, so the backfill is the default; the other two columns are nullable.

### `evaluation_run` (existing — read-only)
No changes. Read fields: `status`, `config_id`, `config_version`, `score` (JSONB:
`traces`, `summary_scores`, `category_metrics`), `per_item_scores`, `organization_id`,
`project_id`.

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| PROMPT_IMPROVEMENT_MODEL | str | `claude-opus-4-8` | LLM used to draft the improved prompt |
| PROMPT_IMPROVEMENT_MIN_CONSISTENCY_RATIO | float | `0.5` | Min fraction of a question's repetitions below threshold to count it as consistently weak |
| PROMPT_IMPROVEMENT_MAX_WEAK_QUESTIONS | int | `50` | Cap on weak questions sent to the LLM |
| PROMPT_IMPROVEMENT_MAX_WEAK_CATEGORIES | int | `10` | Cap on underperforming categories sent to the LLM |

## Design Decisions / Known Limitations

- **Synchronous in v1:** The PRD calls for no async/notifications in v1, so the endpoint
  runs the analysis and the single LLM call inline. Risk: a slow LLM call holds the request
  open. Mitigated by the weak-question/category caps. Async via Celery is the planned
  follow-up; the endpoint contract can stay the same with a `202 + polling` variant later.
- **New version reuses the existing version-creation path** rather than writing
  `config_version` directly, so version numbering, soft-delete, and validation stay
  consistent with human-authored versions. The only divergence is the three provenance
  fields, which the manual `POST /config/{id}/versions` path leaves at their defaults.
- **Provenance as columns, not free-text `commit_message`:** "AI Generated" and the source
  run must be queryable and filterable (e.g. "show AI versions", "versions from run 812"),
  so they are first-class columns. The human-readable rationale still lands in
  `commit_message` for display continuity with existing versions.
- **"Consistently low" is repetition-fraction based:** A simple average could let a single
  catastrophic repetition drag an otherwise-fine question below threshold. Requiring a
  *fraction* of repetitions below threshold targets genuine, repeatable weakness. With
  `duplication_factor = 1` there is only one repetition, so the notion degrades gracefully
  to a single sub-threshold score.
- **Known limitation — prompt-only:** Poor answers caused by stale/irrelevant knowledge-base
  content cannot be fixed here; the prompt rewrite may not move the metric if retrieval is
  the real problem. KB diagnosis is deferred to Phase 2.
- **Known limitation — no auto re-evaluation:** The lift from the new prompt is not measured
  automatically; the user must re-run an evaluation to see whether quality improved.
