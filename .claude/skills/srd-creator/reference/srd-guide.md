# SRD Guide, what each section is and how to write it

An SRD (Software Requirements Document) is the testable contract between whoever
requested the feature and whoever builds it, written before implementation: a
reviewer can take the Functional Requirements table and check each row against the
running system.

This guide describes each section, derived from the Kaapi Evaluation, Fast Evaluation,
and STT Evaluation SRDs. Use `srd-template.md` as the fill-in skeleton.

This guide covers *section semantics only* — what belongs in each section. The
process and the Rules (no redundancy, reuse, naming, ripple) live in `SKILL.md` and
are not repeated here.

---

## 1. Introduction & Purpose  *(required)*

The "what and why" in a few short paragraphs.
- One sentence: what capability this SRD defines and for which system.
- The problem / motivation, what's painful today, who feels it (name the early
  users if known).
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
task. Keep it to the handful that actually define done. Examples of the shape:
- "Users can run <the operation> synchronously with one new request option."
- "Identical semantics to the existing path."
- "Failure isolation, one item's failure must not fail the whole run."

## 4. Assumptions & Constraints  *(required)*

The boundary conditions the design assumes true. This is where you fence the scope.
Cover:
- What's explicitly **out of scope**.
- Hard limits / caps (size caps, thresholds, rate limits).
- Data assumptions (input format, required columns, change frequency).
- Reuse decisions ("No new tables. Reuse <existing entities from the domain map>").
- Pricing / billing notes if external paid APIs are involved.

## 5. Detailed Design (Execution Flow)  *(required)*

How it runs, with brief supporting text, not a long numbered paragraph. Keep the
text to what the diagram can't carry: failure isolation, idempotency, resolution
rules. Diagram mechanics (one image, placeholder band, no HTML/emoji) are in
`SKILL.md`.

## 6. Functional Requirements (Testing)  *(required, the core)*

A table, one row per user-facing behavior. Columns:

| ID | What (user-facing behavior) | Acceptance criteria | Status |

- **ID**: `FR-1`, `FR-2`, …
- **What**: a single behavior in plain language ("Reject a request that exceeds
  the configured cap").
- **Acceptance criteria**: the concrete, checkable condition ("Returns 422 with
  the specific error code and the actual offending count").
- **Status**: `Not Started` / `In Progress` / `Done`. Lifecycle: the SRD ships with
  every row `Not Started`; the builder flips a row to `In Progress` when its
  implementation starts, and `/pr-review` (or the reviewer) flips it to `Done` only
  after verifying the acceptance criterion against the diff.

This table is what QA and review run against.

## 7. Endpoints  *(required when the feature has an API)*

One block per endpoint. For each:
- Method + path.
- One-line description of what it does.
- Request: body fields (a field table with Type / Required / Default / Description
  for non-trivial bodies) and an example JSON body.
- Response: example JSON for the success case.
- Error responses table where relevant: Status / Code / When.

Always show real JSON examples, not just prose. Reuse existing endpoints where
possible and call out only the new fields ("Existing endpoint, with one new
optional field").

## 8. Database Schema / Tables  *(required when there's a data model)*

Per table, a column table:

| Column | Type | Nullable | Default | Description |

Then a **Constraints** list: primary key, unique constraints (name them,
`uq_<table>_<cols>` pattern), foreign keys, indexes. Conventions and reuse rules are
in `SKILL.md`. One guide-level note: when reusing a table, state "No new tables" and
list only the added columns / constraints, with the backfill plan for new non-null
columns.

## 9. Configuration  *(optional)*

New application settings the feature introduces (env vars / settings keys), with
type and default. Drop if there are none.

## 10. Design Decisions / Known Limitations  *(optional)*

Non-obvious choices and their rationale ("X deliberately does not route through
the existing Y path because…"), plus known gaps to revisit. Captures the "why"
so the next reader doesn't re-litigate it.

---

## Section checklist

Required: Introduction & Purpose · Goals · Assumptions & Constraints ·
Detailed Design · Functional Requirements · Endpoints (if API) · DB Schema (if data).
Optional: Resources · Configuration · Design Decisions / Known Limitations.

Before output, verify every **Rule** in `SKILL.md`.
