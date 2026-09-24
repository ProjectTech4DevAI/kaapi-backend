# Native LLM-as-a-Judge Correctness Score SRD

## Introduction & Purpose

This SRD defines a native, reference-based LLM-as-a-judge **correctness** score for Kaapi fast evaluations. Each evaluated row already returns a *cosine similarity* score (how close the generated answer is to the ground truth). This feature adds a second, independent **correctness** score (0 to 1\) plus a short reasoning string, produced by an LLM inside Kaapi with no manual Langfuse configuration.

Today a correctness judgment is only available by hand-configuring a model-based evaluator inside the third-party Langfuse dashboard, per project, outside the platform. That setup is easily forgotten, the judging logic lives outside Kaapi (unversioned and untailorable without engineering), and Kaapi keeps no record of how a row was judged. Early users are NGO eval teams running fast evaluations on their bots.

The feature produces, per evaluated row: a correctness score, a reasoning string, both persisted by Kaapi and written to the row's Langfuse trace alongside the existing similarity score, plus an optional per-run judge configuration the team can send with the run.

- **Phase 1 (this release):** automatic correctness judging on fast evaluations only; zero-config default (built-in prompt \+ fallback model); optional per-run judge config (ad-hoc or saved reference); both scores persisted and synced to Langfuse; reasoning per row.
- **Phase 2+ (deferred):** batch-mode judging; judge error/retry handling; confirmed performance budget and row-count guidance; rating-example entry UX and final score labels.

Intent: judging is automatic, zero-config, tailorable, explainable, and reversible, owned inside Kaapi so eval owners never open Langfuse.

## Goals

- Correctness judging runs natively inside Kaapi with zero manual Langfuse setup.
- The judge is reference-based: it compares generated answer against ground truth (and the question) and returns a 0 to 1 score plus reasoning.
- Judging is automatic: every fast run is judged, with no flag or opt-in.
- It works out of the box from a built-in default prompt and a fallback model when the run carries no judge config.
- A project team can tailor the judge per run (model, model settings, prompt template) via an ad-hoc configuration or a saved config reference, with no deploy and no extra setup step.
- Omitting the judge config reverts that run to the default prompt \+ fallback model.

## Assumptions & Constraints

- **Out of scope:** batch-mode judging (fast only); retiring Langfuse (results keep syncing to it); robust judge error/retry handling. A failed or malformed judge call must not block the similarity score or the run; the row's correctness is left unscoreable and the run still completes.
- **Trigger:** judging runs inside the existing fast-eval pipeline. No new endpoint; the only API change is one optional `judge_config` field on the existing run trigger, which tailors the judge but never toggles it.
- **Limits:** fast eval is capped at `EVAL_FAST_MAX_UNIQUE_ROWS` (10) unique rows; the judge adds one model call per evaluated row within that cap.
- **Per-row independence:** one row's judge failure must not fail sibling rows.
- **Reuse:** no new tables. Correctness rides the existing `EvaluationRun.score` record (per-trace `scores` list \+ `summary_scores`) and a new durable per-row map mirroring `per_item_scores`. The judge configuration reuses `LLMCallConfig` (the same schema the response path uses): either a reference to a saved configuration (`id` \+ `version`, persisted by the existing `config`/`config_version` flow) or an ad-hoc `blob` carrying the completion params and an optional `prompt_template`; the results read path already aggregates a `Correctness` score, so surfacing needs no change.
- **Starting provider/model:** OpenAI, matching the fast-eval response/embedding path. The fallback model is a configurable default; a run's `judge_config` may override the model and its settings.
- **Pricing:** the judge adds one LLM completion per row (paid), tracked under a `judge` stage in `EvaluationRun.cost`.

## Detailed Design (Execution Flow)

The judge slots into the fast-eval pipeline as a scoring step that runs after similarity is computed, so a judge failure can never block the cosine score. The run is marked complete with both summary scores; per-row scores are then written to Langfuse, and a failed write is recoverable from the durable maps on resync.

### Judged fast-eval run

---

**>> PLACE IMAGE HERE: `assets/flow-a.png`, judged fast-eval run.**
System-level sequence: the user and the real systems involved.

---

Each judged row's correctness score and reasoning are appended to that trace's `scores` list (reasoning carried in the score `comment`), and a `Correctness` summary score is added to `EvaluationRun.score.summary_scores` next to `Cosine Similarity`. The durable per-row correctness map is the source of truth for Langfuse resync.

### Tailor the judge (per-run config)

The run request may carry an optional `judge_config` (`LLMCallConfig`): a saved reference (`id` \+ `version`) is resolved to its stored blob at the judge step, while an ad-hoc `blob` is used directly. The blob's `completion` supplies the judge model \+ settings, and its `prompt_template` (when set) replaces the built-in judge prompt; omitting `judge_config` uses the fallback model and built-in prompt. There is no per-project judge state: each run is judged exactly as configured in its own request, and durable, versioned tailoring lives in the existing saved-config flow the reference points at.

## Functional Requirements (Testing)

| ID | What (user-facing behavior) | Acceptance criteria | Status |
| :---- | :---- | :---- | :---- |
| FR-1 | Every evaluated row gets a correctness score | After a fast run completes, each scoreable row's persisted in evaluation\_run table  score column; trace `scores` carries both a `Cosine Similarity` and a `Correctness` entry | Not Started |
| FR-2 | Correctness is 0 to 1 with reasoning | Each `Correctness` trace score has a numeric value in \[0,1\] and a non-empty reasoning `comment` | Not Started |
| FR-3 | Zero-config default works | Running the fast-eval endpoint without `judge_config` still scores every scoreable row using the fallback model \+ built-in prompt; no flag toggles judging | Not Started |
| FR-4 | Ad-hoc judge config honored | A run with `judge_config.blob` uses that blob's model, settings, and (when set) `prompt_template` for every judged row of that run | Not Started |
| FR-5 | Saved judge config honored | A run with `judge_config.id` \+ `version` resolves the stored config and judges with it; an unknown `id`/`version` fails the request with 404, not the run | Not Started |
| FR-6 | Config is per-run | `judge_config` affects only its own run; the next run without it is back on the default | Not Started |
| FR-7 | Invalid judge config rejected | A `judge_config` carrying both a saved reference and a `blob` (or neither) returns 422 before the run starts | Not Started |
| FR-8 | Two summary scores persisted | `EvaluationRun.score.summary_scores` contains both a `Cosine Similarity` and a `Correctness` entry after completion | Not Started |
| FR-9 | Both scores on the Langfuse trace | Each evaluated row's Langfuse trace shows two distinctly-named scores | Not Started |
| FR-10 | Per-row judge failure isolation | If a row's judge call fails or returns malformed output, that row's correctness is left unscoreable, its cosine score is unaffected, and the run still completes | Not Started |
| FR-11 | Judge cost tracked | After a run, `EvaluationRun.cost` includes a `judge` stage with token counts and USD | Not Started |
| FR-12 | Tenant isolation | A saved config belonging to (org A, project A) is never resolvable from any other (org, project)'s run request | Not Started |

## Endpoints

No new endpoint. The existing fast-eval run trigger (`POST /evaluations`) gains one optional field; correctness surfaces through the already-present run results.

### `POST /evaluations` (existing, one new field)

**New request field:**

| Field | Type | Required | Default | Description |
| :---- | :---- | :---- | :---- | :---- |
| judge\_config | object (`LLMCallConfig`) | No | fallback model \+ built-in prompt | Either a saved config reference (`id` \+ `version`) or an ad-hoc `blob` (completion params \+ optional `prompt_template`) — exactly one of the two |

An ad-hoc blob may set `prompt_template` to override the built-in judge prompt. Its `template` is a plain prompt string, typically graded examples (Query / Generation / Score / Reasoning) that steer the judge; it carries no interpolation placeholder, Kaapi appends the evaluated row's question, generated answer, and ground truth itself.

Ad-hoc configuration:

```json
{
  "judge_config": {
    "blob": {
      "completion": {
        "provider": "openai",
        "type": "text",
        "params": { "model": "gpt-4o", "temperature": 0.0 }
      },
      "prompt_template": { "template": "Example:\nQuery: Can eating carrots improve your vision?\nGeneration: Yes, eating carrots significantly improves your vision, especially at night. This is why people who eat lots of carrots never need glasses...\nScore: 0.3\nReasoning: The query could have been answered by simply stating that eating carrots can improve one's vision, but the generation included a lot of unasked supplementary information which makes it not very concise." }
    }
  }
}
```

Saved config reference:

```json
{
  "judge_config": { "id": "9c2e4d6f-1a2b-3c4d-5e6f-7a8b9c0d1e2f", "version": 3 }
}
```

**Response:** unchanged; both scores appear in the run results.

**Error responses (new):**

| Status | Code | Message |
| :---- | :---- | :---- |
| 404 | config\_not\_found | "No config found for the given id and version." |
| 422 | invalid\_judge\_config | "Provide either 'id' with 'version' for stored config OR 'blob' for ad-hoc config, not both." |

## Database Schema

No new tables. The judge's durable configuration lives in the existing `config`/`config_version` tables (via saved `LLMCallConfig` references); only `evaluation_run` changes, so the schema is presented as a table below (no diagram needed at this complexity).

### `evaluation_run` (existing, reused)

Correctness rides the existing score columns; one new column holds the durable correctness map.

| Column | Type | Now carries |
| :---- | :---- | :---- |
| score | JSONB | `summary_scores` gains a `Correctness` entry; each `traces[].scores` gains a `Correctness` score with value \+ reasoning `comment` |
| per\_item\_correctness | JSONB (YES, default NULL) | New column mirroring `per_item_scores`: durable `{trace_id: correctness}` map, source of truth for Langfuse resync |
| unscoreable | JSONB | Reused for rows the judge could not score, alongside existing cosine reasons |
| cost | JSONB | Gains a `judge` stage (tokens \+ USD) |

**Backfill plan:** `per_item_correctness` is nullable with default NULL; pre-feature runs need no backfill (they carry no correctness data).

## Configuration

| Setting | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| EVAL\_JUDGE\_FALLBACK\_MODEL | str | `gpt-4o-mini` | Judge model when a project has no config |
| EVAL\_JUDGE\_DEFAULT\_TEMPERATURE | float | `0.0` | Default judge temperature when not overridden |

The built-in default judge prompt and the `Correctness` score name (mirroring `COSINE_SCORE_NAME`) are owned in Kaapi code as constants.

## Design Decisions / Known Limitations

- **No new per-row table.** Correctness reuses the per-trace `scores` list and the `EvaluationRun.score` summary, matching how cosine per-row data is already persisted to S3 and synced to Langfuse; reasoning is stored in the score `comment`.
- **No judge-config table or CRUD endpoints.** The judge config is a per-run `LLMCallConfig` field on the run request; the only persistent thing is a saved config (`id` \+ `version`) in the existing `config`/`config_version` flow, which already gives versioned, reusable tailoring. A per-project binding table (plus GET/POST/DELETE and its one-active-row and tenant-isolation surface) would duplicate that persistence for no gain. Trade-off: the config must be sent with each run rather than "set once per project".
- **Reuses `LLMCallConfig`** rather than a parallel judge-only config shape, so the judge config validates and resolves exactly like the response path: one-of saved reference or ad-hoc `blob`, with `prompt_template` as the ad-hoc prompt override.
- **`per_item_correctness` as a separate column** (vs folding into `per_item_scores`) keeps the two score families independently resyncable and mirrors the existing cosine map exactly.
- **Judge runs after similarity** so cosine is already computed and a judge failure can never block it.
- **Known limitation, error/retry (deferred):** a failed judge row is left unscoreable with no judge-specific retry; defined retry/backoff is Phase 2\.
- **Open (from PRD):** judge error/retry behavior, performance budget and supported row count, the rating-example entry format, and the final user-facing score labels.
