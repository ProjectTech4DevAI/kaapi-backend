---
name: srd-creator
description: Create a Software Requirements Document (SRD) for a Kaapi feature. Use when the user wants to write, draft, scaffold, or plan an SRD / spec / requirements doc for a new feature or capability (e.g. an evaluation pipeline, a new endpoint set, a provider integration). Produces a structured markdown SRD from the standard Kaapi template.
---

# SRD Creator

Generate a Software Requirements Document for a Kaapi feature, matching the house
style used in the existing Evaluation / Fast Evaluation / STT Evaluation SRDs.

## What an SRD is

An SRD is the contract written *before* code: what to build, why, what is in/out
of scope, the execution flow, the API surface, and the DB schema, specific enough
that an engineer can build it and a reviewer can test it against the Functional
Requirements table. It is testable spec, not design prose.

## A PRD is mandatory

**The SRD is derived solely from a PRD.** If the user has not supplied a PRD, a
file path or pasted content, stop and ask for one. Do not invent the problem,
scope, or requirements from the conversation or assumption. No PRD → no SRD.

Division of sources: the PRD supplies the *what and why* (problem, scope, goals,
personas); the codebase (via the wiki and model files) supplies the *shapes of
what already exists* (tables, models, endpoints). Never the reverse: don't pull
scope from the codebase, don't pull schemas from the PRD or memory.

## Workflow

1. **Read the reference material first.**
   - `reference/srd-guide.md`, what each section means and what good content looks like.
   - `reference/srd-template.md`, the skeleton to fill in.
   - If unsure of house style, skim a prior SRD under `features/*/SRD.md`.
   - **Codebase knowledge comes from the wiki, not from memory or ad-hoc greps:**
     open `docs/wiki/INDEX.md` and load the touched domain's `docs/wiki/modules/*.md`
     page. It carries the domain's routes, tables → model-file map, schemas, services,
     and external boundaries. Follow its deep-dive link into `docs/architecture/*.md`
     only for a design-level question; never bulk-load those.

2. **Load the Quality Checklist into a `TodoWrite` list before writing.** Hold it
   for the whole creation so each rule is checked once and not re-litigated per
   section. Tick items as sections land; run a final pass against it before output.

3. **Blast-radius check (after reading the PRD, before writing).** Open
   `docs/wiki/domain-map.md`. Name the primary entity(ies) the feature changes in
   the map's vocabulary, collect the 1-hop and 2-hop `consumed by` surfaces
   (tables, logical consumers, and the external consumers the map lists). For
   every surface the PRD does not address, ask the user: in scope / deferred /
   out of scope —
   never silently include or exclude. Record the decisions in the SRD
   (out-of-scope bullets in Assumptions, or a small table). For module context,
   load only the relevant `docs/wiki/modules/*.md` page via `docs/wiki/INDEX.md`;
   do not bulk-load architecture docs.

   **Schema comes from code, never memory.** The module page's tables list maps
   each table to its `models/*.py` file — before writing or reusing any table in
   the SRD's DB Schema section, Read those model files. Column names, types,
   nullability, and existing JSONB shapes in the SRD must match the model file;
   before proposing a NEW table, the domain map + model files must show nothing
   existing fits.

4. **Read the PRD and map it onto the template.** Problem → Introduction, Goals →
   Goals, Users → Introduction/personas, scope → Assumptions & phasing, etc. Fill
   section by section. Anything the PRD leaves open goes under *Design Decisions /
   Known Limitations* as an open question, never fabricated. Drop optional sections
   that don't apply (Resources, Configuration) rather than leaving them empty.
   - **Resources links are never fabricated or guessed** (no inventing
     `Fast Evaluation SRD.md`-style paths or PRD URLs). Ask the user for the actual
     Google Docs / PRD / related-SRD links; if they have none, drop the section.

5. **Write the Functional Requirements table** as the testable core. Every row =
   one user-facing behavior + a concrete acceptance criterion + a Status. If you
   can't write a checkable acceptance criterion, the requirement is too vague;
   sharpen it.

6. **Output** as `features/<feature-slug>/SRD.md`, where `<feature-slug>` is a
   short kebab-case slug for the feature (e.g. `features/account-balances/SRD.md`).
   Create the `features/<feature-slug>/` directory if it doesn't exist. Every feature
   keeps its SRD and PRD together in this one folder, if a `PRD.md` already exists
   for the same feature (written by the `start-prd` skill), write the `SRD.md`
   alongside it in the same folder rather than creating a new location. Reuse the
   existing slug exactly; do not invent a parallel folder. The H1 inside the file is
   the feature's display name (e.g. `# Account Balances SRD`).

7. **Generate exactly one image: the primary execution-flow diagram. Render it into
   `assets/`, do not embed it inline, and leave a placeholder band naming the file.**
   The author exports the `.md` to Google Docs and manually pastes the image at its
   band, so the band's job is to say *which* file goes *where*.
   - **One image only.** The single SRD image is the main execution flow. Everything
     else is words or tables, no second image: a second flow (e.g. plain config CRUD)
     is prose plus the Endpoints section, and the **DB schema is always column tables,
     never a diagram**.
   - **Render it.** Write a mermaid source for the execution flow, then run the
     skill's helper script (high-res mermaid render to a crisp png; no global
     installs, it uses `npx`):
     ```bash
     scripts/render-diagram.sh flow-a.mmd features/<feature-slug>/assets/flow-a.png
     ```
   - **Placeholder band in the `.md`.** A horizontal-rule-fenced block (renders as a
     distinct band in GitHub *and* after a Google Docs markdown import, which turns
     `---` into a real line) naming the asset file:
     ```markdown
     ---

     **>> PLACE IMAGE HERE: `assets/flow-a.png`, <one-line flow name>.**
     System-level sequence: the user and the real systems involved.

     ---
     ```
   - **No HTML tags and no inline CSS** (e.g. no `<div style=...>`), the Google Docs
     markdown importer strips them, so colored callouts do not survive. Plain
     markdown only. **No emoji.** Do not use `![](...)` inline embeds (a relative
     image path does not resolve after the Google Docs import; the band is the cue).

## Quality Checklist

Load these as todos at the start; verify each before output:

- [ ] PRD supplied and read; every SRD claim traces back to the PRD.
- [ ] Wiki loaded before writing: `docs/wiki/INDEX.md` + the touched domain's
      `docs/wiki/modules/*.md` page read; the SRD's endpoint paths, table names, and
      field names cross-checked against that page (and architecture deep-dive
      consulted only if a design question came up).
- [ ] Blast radius run against `docs/wiki/domain-map.md`: 1-hop + 2-hop consumers
      collected; every surface the PRD skips confirmed with the user
      (in scope / deferred / out of scope), none silently dropped.
- [ ] DB Schema section derived from code, not memory: the touched domain's model
      files (paths from the module page's tables list) were Read before writing any
      column table; every reused table's columns match `models/*.py`.
- [ ] Output at `features/<feature-slug>/SRD.md`, alongside the feature's `PRD.md`
      (reuse the existing slug; don't create a parallel folder).
- [ ] All required sections present (Introduction, Goals, Assumptions, Detailed
      Design, Functional Requirements; Endpoints if the feature has an API; DB
      Schema if there's a data model).
- [ ] **No fact stated more than once**, each fact lives in one home section;
      elsewhere reference it, don't restate it.
- [ ] No filler, hedging, or restated bullets; every sentence adds new information.
- [ ] Phasing explicit (Phase 1 vs Phase 2+).
- [ ] Every FR row has a concrete, testable acceptance criterion + Status.
- [ ] Every endpoint shows real request/response JSON, not just field lists.
- [ ] DB schema follows Kaapi conventions (`inserted_at`/`updated_at`,
      `organization_id` + `project_id`, snake_case, FK indexes).
- [ ] High-level only, no function names, internal variables, or step internals
      (DB schema, class/endpoint contracts, error codes, settings are fine).
- [ ] Existing models reused, not duplicated, checked the codebase first; any new
      shape is justified by no existing model fitting; config-like fields reuse the
      canonical shape named in the module wiki page, never a bespoke parallel one.
- [ ] One design story end to end: after any revision, grep the SRD for the
      superseded names and phrases (fields, models, endpoints, tables) and confirm
      zero hits across every section.
- [ ] Exactly one image (the execution flow) rendered to
      `features/<feature-slug>/assets/flow-a.png`, with a horizontal-rule-fenced band
      naming it (no inline `![](...)` embed, no mermaid blocks, no HTML, no emoji).
- [ ] DB schema is column tables, not a diagram; no second image anywhere.
- [ ] Optional empty sections deleted, not left as placeholders.

## Rules

- **No redundancy.** State each fact exactly once, in its home section. Later
  sections reference it ("as in Goals"), they don't repeat it. Goals are outcomes,
  not echoes of the Introduction. Assumptions are boundaries, not re-listed goals.
  FR acceptance criteria are checkable conditions, not restatements of the behavior
  column. Repetition across sections is the #1 cause of bloated SRDs.
- **Keep it high-level. No variable/function-level detail.** An SRD describes
  behavior and interfaces, not implementation internals. Allowed: DB schema (tables,
  columns, constraints), object/class and endpoint contracts, error codes, settings.
  Not allowed: specific function names, internal variable/field names, private
  helpers, stage/step internals, concurrency-pool names, the reader sees those in
  PR review. Name *what* a thing does and *which entity* holds it, not the symbol
  that implements it.
- **Reuse existing models, don't invent new ones.** Before specifying any config,
  request body, or pydantic/SQLModel class, search the codebase for an existing one
  that fits. Default order: (1) reuse as-is, (2) extend/compose the existing one,
  (3) only if nothing fits, add a new shape, and call out explicitly why no existing
  model worked. Never introduce a parallel config that duplicates an existing one.
  A reviewer asking "can't we reuse X?" is a failure of this rule.
  - **Reuse at the highest-level wrapper that fits, not the innermost params model.**
    When the codebase already has a wrapper shape that solves the whole problem
    (e.g. "an ad-hoc value OR a reference to a saved, versioned one"), spec that
    wrapper whole; never re-spec its parts as bespoke sibling fields. The canonical
    shapes and their model files are named in the module's wiki page; pull the exact
    shape from the code, not from memory.
  - **Prefer a per-request config field over a per-project binding table.** When a
    saved/versioned config flow already provides durable, reusable configuration, a
    dedicated binding table plus its CRUD endpoints duplicates that persistence and
    adds uniqueness and tenant-isolation surface for no gain. Add a binding table
    only when "set once, applies to every future request with nothing sent" is an
    explicit product requirement.
- **Diagrams over prose for flows, but the SRD only holds a placeholder.** Both the
  execution flow and the data model want a diagram, not a long numbered paragraph.
  The skill does not draw it; it leaves an author image placeholder note (Workflow
  step 7) and keeps the surrounding prose to what a diagram can't carry (failure
  isolation, idempotency, resolution rules). Tell the author what the diagram should
  depict so they can draw it:
  - **Execution flow at system level, not internals.** The diagram should show the
    user and the real systems that talk to each other, each with arrows in and out.
    It should not turn internal pipeline steps or in-process helpers into separate
    lanes (a lane for an in-process step that is really a provider call misleads
    readers into thinking it's a separate service).
  - **Data model** should mark reused vs new entities.
- **No em dashes.** Use commas, periods, or parentheses. Keep prose clean and
  developer-readable; short sentences over long dash-joined clauses.
- **Match existing system naming.** Column names, table names, request/response field
  names, and class names must follow the codebase's conventions and reuse existing
  names where one exists (snake_case columns, the same field names as the reused
  model). The schema must not read like it came from a different system. Grep for
  an existing name before coining a new one.
- **Any design change ripples through the whole SRD.** When anything the document
  describes changes (a schema, a field, an endpoint added or dropped, a table
  removed, a flow reworked), sweep every section that touched the old design:
  Introduction, Goals, Assumptions, Detailed Design, every FR row, every endpoint's
  request/response/error examples, the DB schema, and Design Decisions. Finish by
  grepping the doc for the superseded names and phrases; zero hits is the done
  condition. A doc where one section speaks the new design and another still speaks
  the old one is worse than either version alone.
- Match Kaapi conventions: `inserted_at`/`updated_at` timestamps (not `created_at`),
  `organization_id` + `project_id` on every multi-tenant table, snake_case columns.
- Phase the scope explicitly. State Phase 1 (build now) vs Phase 2+ (later) so
  scope creep is visible.
- **Functional Requirements are PR-testable functional behaviors only.** Every FR row
  is something a developer or client can verify works when reviewing the PR before it
  ships. Drop rows that merely restate the Intro/Goals, describe internal mechanics,
  or can't be checked against the running system. Lean beats exhaustive.
- Show real request/response JSON for every endpoint, not just field lists.
- **Error responses add value, don't restate the obvious.** Standard codes (409, 422,
  404) are self-explanatory; the value is the readable client-facing message. Show the
  actual message string (or a field-specific example for validation), not a paraphrase
  of the status code.
- DB schema as a table: Column / Type / Nullable / Default / Description, plus a
  Constraints list (unique keys, FKs, indexes).
- Keep the reference style: consistent table headers, one blank line between
  sections.
- Don't pad. Each section earns its place; cut what doesn't apply.
