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
- Provenance on that version, recorded entirely in the existing `commit_message` field
  (no schema changes): an `[AI Generated]` marker, the source evaluation run id, the
  metric/threshold used, and a short rationale describing what the improvement targeted.

Phasing:
- **Phase 1 (this SRD):** Synchronous, user-initiated prompt improvement from one
  completed evaluation. User picks any **numeric** score recorded on the run and the
  low-score threshold. Weak questions are selected by *consistent* low performance;
  underperforming categories feed the analysis. Prompt-only change; prior iterations preserved.
- **Phase 2+ (deferred):** Categorical-score support (caller-supplied "failing values" in
  place of a numeric threshold); knowledge-base staleness diagnosis; async/background
  execution with completion notification; one-click re-evaluation of the new iteration;
  generating and comparing multiple candidate prompts at once.

Quality bar: every generated iteration is **traceable** (to its source evaluation),
**auditable** (rationale + the metric/threshold used are recorded), and **non-destructive**
(added alongside, never overwriting prior iterations).

## Goals

- Generate an improved prompt iteration from a completed evaluation run in a single,
  explicit user action — no automatic triggering.
- Ground the improvement in evaluation evidence: questions that scored below the chosen
  threshold *consistently* across their repetitions, plus underperforming categories.
- Let the user control the analysis: choose the quality metric (any numeric score recorded
  on the run — there is one built-in "Cosine Similarity" score plus any number of custom
  Langfuse scores) and the low-score threshold.
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
- **Dynamic metric selection:** `metric` is a free-form score *name*, not a fixed enum. A
  run records one built-in "Cosine Similarity" score plus any number of custom Langfuse
  scores whose names are not known ahead of time. The chosen name is matched
  (case-insensitive, trimmed) against the entries in `EvaluationRun.score.summary_scores`.
  If no matching score exists in this run, the request is rejected (422,
  `metric_not_available`). Per-repetition values are read from `score.traces[].scores[]`
  (matched by score name), not from the cosine-specific `per_item_scores`.
- **Numeric scores only (Phase 1):** Scores can be NUMERIC or CATEGORICAL. Phase 1 supports
  NUMERIC scores only — the threshold/below-threshold logic is inherently numeric. If the
  chosen score's `data_type` is CATEGORICAL, the request is rejected (422,
  `metric_not_numeric`). Categorical handling (e.g. caller-supplied "failing values") is
  deferred to Phase 2.
- **Threshold:** A plain number; a repetition with the chosen score's value below it is
  "low". There is **no fixed [0, 1] range** — custom numeric scores may use any scale (e.g.
  1–5, 0–100), so any numeric threshold is accepted and interpreted in the score's own units.
- **Reuse:** No new tables and no schema changes. Reuse `evaluation_run` (read-only) and
  `config_version` (one new row, existing columns only). Provenance is recorded in the new
  row's `commit_message`. Reuse the existing version-creation CRUD path
  (`create_or_raise`) so version numbering and soft-delete semantics stay consistent.
- **Multi-tenancy:** `evaluation_run` carries `organization_id` + `project_id`; `config`
  and `config_version` are project-scoped (`project_id`, no `organization_id`). The run's
  `config_id` must resolve to a config in the caller's project, else the request is rejected.
- **Pricing:** The improvement makes one LLM call per invocation (paid). Bounded by the
  weak-question/category caps above.
- **Starting provider/model:** Claude (default `claude-opus-4-8`), configurable through
  settings (see Configuration).
- **Credentials:** The feature uses a single platform-owned Anthropic key
  (`ANTHROPIC_API_KEY` env var) shared by every org/project, so it works
  without per-project credentials. If the key is unset, the request fails with
  `502 prompt_generation_failed`.

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
4. **Resolve & verify the metric.** Match the chosen `metric` name (case-insensitive,
   trimmed) against `score.summary_scores`. If no matching score exists, reject (422,
   `metric_not_available`). If the matched score's `data_type` is not NUMERIC, reject (422,
   `metric_not_numeric`) — categorical scores are deferred to Phase 2.
5. **Select weak questions.** For each `score.traces` entry, read the chosen score's value
   from its inner `scores[]` list (matched by name); skip repetitions where the score is
   missing, `unscoreable`, or non-numeric. Group by `question_id`; compute the fraction of
   scoreable repetitions whose value is below `threshold`. Keep groups where that fraction ≥
   `MIN_CONSISTENCY_RATIO`; ignore groups with no scoreable repetitions. Sort by mean
   sub-threshold severity (lowest mean first); truncate to `MAX_WEAK_QUESTIONS`. For each
   kept question, collect `{question, llm_answer, ground_truth_answer, category, mean_score}`.
6. **Select underperforming categories.** Computed generically from traces (not from the
   fixed `category_metrics.avg_cosine`/`avg_correctness`, which don't exist for arbitrary
   score names): group the scoreable traces by `category`, take the mean of the chosen
   score's values per category, keep categories whose mean is below `threshold`; sort
   ascending; truncate to `MAX_WEAK_CATEGORIES`.
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
10. **Persist the new version.** Create a `config_version` via the existing version path
    (`create_or_raise`): `version = latest active version + 1`, `config_blob` = the composed
    blob, and `commit_message` = the provenance string (truncated to 512 chars):
    `"[AI Generated] {rationale} (source_evaluation_run_id={evaluation_id}, metric={metric},
    threshold={threshold})"`. When the weak-question/category caps truncated the analysis,
    the rationale notes that. No new columns are written.
11. **Respond.** Return the new version (id, version number, `commit_message`).

> Sequence Flow Diagram: _TBD_

## Functional Requirements (Testing)

| ID | What (user-facing behavior) | Acceptance criteria | Status |
|----|-----------------------------|---------------------|--------|
| FR-1 | Generate an improved prompt from a completed evaluation | `POST /evaluations/{id}/improve-prompt` on a `completed` run returns 201 with a new `config_version` whose `version` = previous latest + 1 | Not Started |
| FR-2 | Reject improvement on a non-completed run | Run with status `pending`/`processing`/`failed` returns 409 `evaluation_not_completed` | Not Started |
| FR-3 | Reject when source config/version is unavailable | If the run's `config_version` (or parent `config`) is missing or soft-deleted, returns 409 `source_config_unavailable` | Not Started |
| FR-4 | User selects any score recorded on the run | `metric` is a free-form score name; a name matching a `summary_scores` entry (case-insensitive) is accepted | Not Started |
| FR-5 | Reject a metric absent from this run | A `metric` name with no matching score in this run returns 422 `metric_not_available` | Not Started |
| FR-5b | Reject a non-numeric metric (Phase 1) | Choosing a score whose `data_type` is CATEGORICAL returns 422 `metric_not_numeric` | Not Started |
| FR-6 | User selects the low-score threshold | `threshold` is any number (no [0, 1] bound) interpreted in the score's own units; a repetition below it is "low" | Not Started |
| FR-7 | Weak questions chosen by *consistent* low performance | A question whose sub-threshold repetition fraction ≥ `MIN_CONSISTENCY_RATIO` is selected; a question with one low repetition out of many (below ratio) is excluded | Not Started |
| FR-8 | Underperforming categories feed the analysis | Categories whose chosen-metric average < `threshold` are passed to the LLM and reflected in the analysis | Not Started |
| FR-9 | Nothing-to-improve guard | When no weak questions and no weak categories are found, returns 422 `no_weak_signals` and creates no version | Not Started |
| FR-10 | Prompt-only change | New version's `config_blob` equals the source blob except `completion.params.instructions`; model, `knowledge_base_ids`, and all other params are unchanged | Not Started |
| FR-11 | New version marked AI-generated | New `config_version.commit_message` begins with the `[AI Generated]` marker | Not Started |
| FR-12 | New version traceable to source evaluation | New `config_version.commit_message` contains `source_evaluation_run_id={the evaluation id used}` | Not Started |
| FR-13 | Rationale recorded | New version's `commit_message` contains the improvement rationale plus the `metric` and `threshold` used | Not Started |
| FR-14 | Prior iterations preserved | All pre-existing `config_version` rows for the config are unchanged and still retrievable after generation | Not Started |
| FR-15 | Caps enforced and disclosed | With > `MAX_WEAK_QUESTIONS` weak questions, only the lowest-scoring `MAX_WEAK_QUESTIONS` are used and the `commit_message` rationale notes the truncation | Not Started |
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
| metric | string | Yes | — | Name of the numeric score to judge "low" on — any score recorded on the run (e.g. `Cosine Similarity` or a custom Langfuse score). Matched case-insensitively against `summary_scores`. Must be NUMERIC in Phase 1. |
| threshold | number | Yes | — | A repetition with the chosen score below this is "low". No fixed range — interpreted in the score's own units. |

```json
{
  "metric": "Cosine Similarity",
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
    "commit_message": "[AI Generated] Tightened answer scoping and added explicit grounding instructions to address 7 consistently low questions, concentrated in the 'Eligibility' and 'Payments' categories. (source_evaluation_run_id=812, metric=Cosine Similarity, threshold=0.7)",
    "config_blob": { "...": "composed blob — prompt-only change" },
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
| 422 | metric_not_available | No score matching the given `metric` name is recorded in this run |
| 422 | metric_not_numeric | The chosen score is CATEGORICAL; only NUMERIC scores are supported in Phase 1 |
| 422 | no_weak_signals | No consistently-low questions and no underperforming categories found |
| 502 | prompt_generation_failed | The LLM call failed or returned an unusable result |

## Database Schema

**No schema changes.** No new tables, no new columns, no new enum types, and no migration.
The feature reads `evaluation_run` and adds one new `config_version` row using the existing
columns only.

### `config_version` (existing — unchanged)
Stores prompt iterations for a configuration. The AI-generated iteration is a normal new
row created through the existing version-creation path. Its AI provenance — the
`[AI Generated]` marker, the source evaluation run id, and the metric/threshold used —
plus the rationale live in the existing `commit_message` field (truncated to its 512-char
limit). Nothing about prior rows changes.

### `evaluation_run` (existing — read-only)
No changes. Read fields: `status`, `config_id`, `config_version`, `score` (JSONB:
`traces` — each with an inner `scores[]` list of `{name, value, data_type, unscoreable}` —
plus `summary_scores` and `category_metrics`), `organization_id`, `project_id`. The chosen
metric is resolved from `summary_scores` and its per-repetition values read from
`traces[].scores[]`; `per_item_scores` (cosine-specific) is not used.

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| ANTHROPIC_API_KEY | str | `""` | Platform-owned Anthropic key shared by all orgs/projects; required for the feature |
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
  consistent with human-authored versions. The AI iteration is identical in shape to a
  human-authored one — only the `commit_message` distinguishes it.
- **Provenance in `commit_message`, not dedicated columns:** Provenance ("AI Generated",
  the source run, the metric/threshold, and the rationale) is folded into the existing
  `commit_message` rather than first-class columns. This keeps the change schema-free — no
  migration, no enum, no new fields — and the AI iteration stays a plain `config_version`
  the UI already knows how to render. Trade-off: provenance is not independently queryable
  or filterable (e.g. "show all AI versions" or "versions from run 812" would require text
  search on `commit_message`); promoting it to columns is a deferred follow-up if those
  queries become needed.
- **Dynamic metric, numeric-only in Phase 1:** Scores are not a fixed set — a run carries
  one built-in cosine score plus any number of custom Langfuse scores — so `metric` is a
  free-form name resolved against `summary_scores` rather than an enum, and per-repetition
  values come from `traces[].scores[]` (by name) rather than the cosine-specific
  `per_item_scores`. Weak *categories* are likewise computed from traces, because the fixed
  `category_metrics.avg_cosine`/`avg_correctness` keys don't generalize. Categorical scores
  have no meaningful numeric threshold, so Phase 1 supports NUMERIC scores only and rejects
  categorical ones (`metric_not_numeric`); a caller-supplied "failing values" model for
  categorical scores is deferred to Phase 2. The numeric `threshold` carries no fixed range
  because custom scores use arbitrary scales.
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
