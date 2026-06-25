---
name: srd-creator
description: Create a Software Requirements Document (SRD) for a Kaapi feature. Use when the user wants to write, draft, scaffold, or plan an SRD / spec / requirements doc for a new feature or capability (e.g. an evaluation pipeline, a new endpoint set, a provider integration). Produces a structured markdown SRD from the standard Kaapi template.
---

# SRD Creator

Generate a Software Requirements Document for a Kaapi feature, matching the house
style used in the existing Evaluation / Fast Evaluation / STT Evaluation SRDs.

## What an SRD is

An SRD is the contract written *before* code. It states what to build, why,
what is in/out of scope, the execution flow, the API surface, and the DB schema —
specific enough that an engineer can build it and a reviewer can test it against
the Functional Requirements table. It is not design prose; it is testable spec.

## Workflow

1. **Read the reference material first.**
   - `reference/srd-guide.md` — what each section means and what good content looks like.
   - `reference/srd-template.md` — the skeleton to fill in.
   - If unsure of house style, skim a prior SRD; the three originals live as
     `*.docx` in the repo root.

2. **Gather the inputs.** Before writing, you need: the feature name, the problem
   it solves, who the users are, what is in scope for Phase 1 vs later phases, the
   endpoints, and the data model. If the user has not supplied these, ask — do not
   invent endpoints or schema. Missing info is the #1 cause of a useless SRD.

3. **Fill the template** section by section. Drop optional sections that don't
   apply (e.g. Resources, Configuration) rather than leaving them empty.

4. **Write the Functional Requirements table** as the testable core. Every row =
   one user-facing behavior + a concrete acceptance criterion + a Status. If you
   can't write an acceptance criterion for a requirement, the requirement is too
   vague — sharpen it.

5. **Output** as `features/<feature-slug>/SRD.md`, where `<feature-slug>` is a
   short kebab-case slug for the feature (e.g. `features/llm-judge-correctness/SRD.md`).
   Create the `features/<feature-slug>/` directory if it doesn't exist. Every feature
   keeps its SRD and PRD together in this one folder — if a `PRD.md` already exists
   for the same feature (written by the `start-prd` skill), write the `SRD.md`
   alongside it in the same folder rather than creating a new location. Reuse the
   existing slug exactly; do not invent a parallel folder.

## Rules

- Match Kaapi conventions: `inserted_at`/`updated_at` timestamps (not `created_at`),
  `organization_id` + `project_id` on every multi-tenant table, snake_case columns.
- Phase the scope explicitly. State Phase 1 (build now) vs Phase 2+ (later) so
  scope creep is visible.
- Show real request/response JSON for every endpoint, not just field lists.
- DB schema as a table: Column / Type / Nullable / Default / Description, plus a
  Constraints list (unique keys, FKs, indexes).
- Don't pad. Each section earns its place; cut what doesn't apply.
