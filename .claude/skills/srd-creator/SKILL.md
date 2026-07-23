---
name: srd-creator
description: Create a Software Requirements Document (SRD) for a Kaapi feature. Use when the user wants to write, draft, scaffold, or plan an SRD / spec / requirements doc for a new feature or capability (e.g. an evaluation pipeline, a new endpoint set, a provider integration). Produces a structured markdown SRD from the standard Kaapi template.
---

# SRD Creator

Generate a Software Requirements Document for a Kaapi feature, matching the house
style of the existing Evaluation / Fast Evaluation / STT Evaluation SRDs.

An SRD is the contract written *before* code: what to build, why, what is in/out
of scope, the execution flow, the API surface, and the DB schema, specific enough
that an engineer can build it and a reviewer can test each row of the Functional
Requirements table against the running system. Testable spec, not design prose.

## A PRD is mandatory

**The SRD is derived solely from a PRD.** If the user has not supplied one (a file
path or pasted content), stop and ask. Do not invent the problem, scope, or
requirements from the conversation. No PRD → no SRD.

Division of sources: the PRD supplies the *what and why* (problem, scope, goals,
personas); the codebase (via the wiki and model files) supplies the *shapes of
what already exists* (tables, models, endpoints). Never the reverse: don't pull
scope from the codebase, don't pull schemas from the PRD or memory.

## Workflow

1. **Read the reference, then the wiki.**
   - `reference/srd-guide.md` (what each section means), `reference/srd-template.md`
     (the skeleton). Skim a prior `features/*/SRD.md` if unsure of house style.
   - Codebase knowledge comes from the wiki, not memory or ad-hoc greps: open
     `docs/wiki/INDEX.md`, load the touched domain's `docs/wiki/modules/*.md` page
     (routes, tables → model-file map, schemas, services, external boundaries).
     Follow its link into `docs/architecture/*.md` only for a design question;
     never bulk-load those.

2. **Load the Rules (below) into a `TodoWrite` list before writing.** Tick each as
   sections land; run a final pass over them before output.

3. **Blast-radius check** (after reading the PRD, before writing). Open
   `docs/wiki/domain-map.md`; name the primary entity(ies) in the map's vocabulary
   and collect the 1-hop and 2-hop `consumed by` surfaces (tables, logical and
   external consumers). For every surface the PRD does not address, ask the user:
   in scope / deferred / out of scope, never silently include or exclude. Record the
   decisions in Assumptions.
   - **Schema comes from code, never memory.** Before writing or reusing any table,
     Read its `models/*.py` file (path from the module page's tables list); column
     names, types, nullability, and JSONB shapes in the SRD must match it. Propose a
     new table only when the domain map + model files show nothing existing fits.

4. **Read the PRD and map it onto the template.** Problem → Introduction, Goals →
   Goals, Users → personas, scope → Assumptions & phasing. Anything the PRD leaves
   open goes under *Design Decisions / Known Limitations* as an open question, never
   fabricated. Drop optional sections that don't apply. Resources links are never
   guessed: ask the user for the real Google Docs / PRD / related-SRD URLs, or drop
   the section.

5. **Write the Functional Requirements table** as the testable core (per the Rules).

6. **Output** at `features/<feature-slug>/SRD.md` (kebab-case slug, e.g.
   `features/account-balances/SRD.md`). If a `PRD.md` already exists for the feature
   (from `start-prd`), write `SRD.md` alongside it and reuse that slug exactly; do
   not fork a parallel folder. The H1 is the feature's display name
   (e.g. `# Account Balances SRD`).

7. **Render the one execution-flow diagram.** Write a mermaid source, render it with
   the helper (high-res png, no global installs, uses `npx`), and leave a placeholder
   band naming the file, do not embed it inline (the author exports to Google Docs
   and pastes the image at the band):
   ```bash
   scripts/render-diagram.sh flow-a.mmd features/<feature-slug>/assets/flow-a.png
   ```
   The band is horizontal-rule fenced (Google Docs turns `---` into a real line):
   ```markdown
   ---

   **>> PLACE IMAGE HERE: `assets/flow-a.png`, <one-line flow name>.**
   System-level sequence: the user and the real systems involved.

   ---
   ```

## Rules

The completion checklist. Each is a checkable condition; verify every one before
output.

- **Single source of truth.** State each fact once, in its home section; later
  sections reference it ("as in Goals"), never restate it. Goals are outcomes, not
  echoes of the Introduction; Assumptions are boundaries, not re-listed goals; FR
  acceptance criteria are checkable conditions, not restatements of the behavior
  column. Repetition across sections is the #1 cause of bloated SRDs. No filler or
  hedging: every sentence adds new information.
- **Every claim traces to the PRD.** No fact in the SRD that the PRD didn't supply
  (scope, goals) or the code confirmed (schemas). Open items go to Design Decisions,
  never fabricated.
- **High-level only.** Behavior and interfaces, not internals. Allowed: DB schema
  (tables, columns, constraints), object/class and endpoint contracts, error codes,
  settings. Not allowed: function names, private helpers, internal variable/field
  names, stage/step internals, concurrency-pool names, the reader sees those in PR
  review. Name *what* a thing does and *which entity* holds it, not the symbol.
- **Reuse existing models, don't invent.** Order: (1) reuse as-is, (2) extend/compose
  the existing one, (3) only if nothing fits, add a new shape and say why. A reviewer
  asking "can't we reuse X?" is a failure of this rule.
  - Reuse at the **highest-level wrapper that fits**, not the innermost params model.
    When the codebase already has a wrapper that solves the whole problem (e.g. "an
    ad-hoc value OR a reference to a saved, versioned one"), spec that wrapper whole;
    never re-spec its parts as bespoke sibling fields.
  - Prefer a **per-request config field over a per-project binding table.** A binding
    table plus its CRUD duplicates a saved/versioned config's persistence and adds
    uniqueness + tenant-isolation surface for no gain. Add one only when "set once,
    applies to every future request with nothing sent" is an explicit requirement.
- **Match system naming.** snake_case columns/fields, table names, and class names
  follow the codebase and reuse existing names; grep for an existing name before
  coining a new one. The schema must not read like it came from a different system.
- **Ripple: one design story, end to end.** When anything changes (schema, field,
  endpoint, table, flow), sweep every section that touched the old design
  (Introduction, Goals, Assumptions, Detailed Design, every FR row, every endpoint's
  request/response/error, DB schema, Design Decisions). The done condition: grep the
  SRD for the superseded names/phrases, zero hits. A doc where one section speaks the
  new design and another the old is worse than either alone.
- **Kaapi DB conventions.** `inserted_at`/`updated_at` (not `created_at`),
  `organization_id` + `project_id` on every multi-tenant table, snake_case, FK
  indexes. Filterable data as first-class columns; bag-of-attributes as `JSONB`.
- **Phase the scope explicitly.** Phase 1 (build now) vs Phase 2+ (later), so scope
  creep is visible.
- **Functional Requirements are PR-testable behaviors only.** Each row = one
  user-facing behavior + a concrete, checkable acceptance criterion + a Status. If
  you can't write a checkable criterion, the requirement is too vague, sharpen it.
  Drop rows that restate the Intro/Goals or describe internal mechanics. Lean beats
  exhaustive.
- **Real request/response JSON for every endpoint**, not just field lists. Error
  responses show the actual client-facing message string (or a field-specific
  validation example), not a paraphrase of a self-explanatory status code.
- **One image: the execution flow.** Rendered to `assets/flow-a.png`, referenced by
  the placeholder band. No second image: a secondary flow (e.g. config CRUD) is prose
  plus the Endpoints section, and the **DB schema is always column tables, never a
  diagram**. The flow shows the user and the real systems that talk to each other,
  not internal pipeline steps as separate lanes. No inline `![](...)`, no mermaid
  blocks, no HTML/inline CSS, no emoji, no em dashes (the Google Docs importer strips
  the first three; use commas, periods, or parentheses for the last).
- **Don't pad.** Each section earns its place; delete inapplicable optional sections
  rather than leaving them empty.

## Before output

- Every Rule above holds (todos ticked).
- All required sections present (see the guide's section checklist).
- Grep the SRD for any superseded name/phrase from a revision: zero hits.
