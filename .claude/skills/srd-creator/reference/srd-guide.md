# SRD Guide, what each section is and how to write it

An SRD (Software Requirements Document) is the testable contract between whoever
requested the feature and whoever builds it, written before implementation: a
reviewer can take the Functional Requirements table and check each row against the
running system.

This guide describes each section, derived from the Kaapi Evaluation, Fast Evaluation,
and STT Evaluation SRDs. Use `srd-template.md` as the fill-in skeleton.

**The SRD is derived solely from a PRD**, no PRD, no SRD. Map PRD sections onto the
template; record anything the PRD leaves open as an open question under *Design
Decisions / Known Limitations*, never as invented detail.

**No redundancy.** State each fact exactly once, in its home section; later sections
reference it instead of repeating it. Goals are outcomes, not echoes of the
Introduction. Assumptions are boundaries, not re-listed goals. FR acceptance criteria
are checkable conditions, not restatements of the behavior column.

---

## 1. Introduction & Purpose  *(required)*

The "what and why" in a few short paragraphs.
- One sentence: what capability this SRD defines and for which system.
- The problem / motivation, what's painful today, who feels it (name the early
  users if known, e.g. "early users from Glific").
- What the feature produces at minimum (the concrete outputs).
- Explicit phasing: what is Phase 1, what is deferred to Phase 2/3. This is where
  scope gets pinned down before anyone argues about it later.
- One line on intent / quality bar (e.g. "repeatable, comparable, auditable").

## 2. Resources  *(optional)*

Links to related SRDs, external API docs, research notes, design docs. **Ask the
user for the real links** (Google Docs, PRD, related SRDs), never fabricate or
guess paths/URLs. Drop the section if the user has nothing to link.

## 3. Goals  *(required)*

A short bulleted list of what success looks like. Each goal is an outcome, not a
task. Keep it to the handful that actually define done. Examples:
- "Add a `run_mode="fast"` option so users can run a text evaluation synchronously."
- "Identical scoring semantics to the batch path."
- "Failure isolation, one item's failure must not fail the whole run."

## 4. Assumptions & Constraints  *(required)*

The boundary conditions the design assumes true. This is where you fence the scope.
Cover:
- What's explicitly **out of scope** ("Voice/audio is out of scope").
- Hard limits / caps (dataset size caps, thresholds, rate limits).
- Data assumptions (how often the golden set changes, required CSV columns).
- Reuse decisions ("No new tables. Reuse `evaluation_dataset`, `evaluation_run`").
- Pricing / billing notes if external paid APIs are involved.
- Which provider / model / version you start with.

## 5. Detailed Design (Execution Flow)  *(required)*

How it runs, with brief supporting text, not a long numbered paragraph. Render each
flow's diagram to a png in `assets/`, but do **not** embed it inline: where each
diagram belongs, leave a horizontal-rule-fenced band naming the asset file the author
pastes there (no mermaid, no HTML, no emoji). **Exactly one image per SRD: the primary
execution flow.** A second flow (e.g. plain config CRUD) is prose plus the Endpoints
section, not another image, and the DB schema is always column tables, never a diagram.
Keep the text to what a diagram can't carry: failure isolation, idempotency,
resolution rules.

Stay high-level, actors and behavior ("Pipeline → Judge: question + answer",
"persist both scores"), not internal function or variable names. Pipeline *stages*
named by behavior are fine; private helpers and field names are not.

## 6. Functional Requirements (Testing)  *(required, the core)*

A table, one row per user-facing behavior. Columns:

| ID | What (user-facing behavior) | Acceptance criteria | Status |

- **ID**: `FR-1`, `FR-2`, …
- **What**: a single behavior in plain language ("Reject `run_mode="fast"` when
  dataset has >10 unique rows").
- **Acceptance criteria**: the concrete, checkable condition ("Returns 422 with
  `dataset_too_large_for_fast` error and the actual unique-row count").
- **Status**: `Not Started` / `In Progress` / `Done`.

If you cannot write a crisp acceptance criterion, the requirement is too vague;
split or sharpen it. This table is what QA and review run against.

## 7. Endpoints  *(required when the feature has an API)*

One block per endpoint. For each:
- Method + path (`POST /evaluations`).
- One-line description of what it does.
- Request: body fields (a field table with Type / Required / Default / Description
  for non-trivial bodies) and an example JSON body.
- Response: example JSON for the success case.
- Error responses table where relevant: Status / Code / When.

Always show real JSON examples, not just prose. Reuse existing endpoints where
possible and call out only the new fields ("Existing endpoint, with a new `run_mode`
field").

## 8. Database Schema / Tables  *(required when there's a data model)*

Always column tables, never a diagram or image (the one SRD image is the execution
flow). For each table, a column table:

| Column | Type | Nullable | Default | Description |

Then a **Constraints** list: primary key, unique constraints (name them, e.g.
`uq_evaluation_dataset_name_org_project`), foreign keys, indexes.

Kaapi conventions to enforce:
- `id INTEGER PK` auto-increment (or UUID where the domain calls for it).
- `organization_id` + `project_id` on every multi-tenant table.
- `inserted_at` / `updated_at TIMESTAMP NOT NULL DEFAULT now()`, **not** `created_at`.
- Filterable data as first-class columns; bag-of-attributes as `JSONB`.
- Prefer reusing existing tables; if so, state "No new tables" and list only the
  added columns / constraints, with the backfill plan for new non-null columns.

## 9. Configuration  *(optional)*

New application settings the feature introduces (env vars / settings keys), with
type and default. Drop if there are none.

## 10. Design Decisions / Known Limitations  *(optional)*

Non-obvious choices and their rationale ("Fast eval does not route through
`execute_llm_call()` because…"), plus known gaps to revisit. Captures the "why"
so the next reader doesn't re-litigate it.

---

## Section checklist

Required: Introduction & Purpose · Goals · Assumptions & Constraints ·
Detailed Design · Functional Requirements · Endpoints (if API) · DB Schema (if data).
Optional: Resources · Configuration · Design Decisions / Known Limitations.

Before output, verify the full **Quality Checklist** in `SKILL.md`, PRD-traceable,
correctly named, no repeated facts, testable FRs, Kaapi DB conventions.
