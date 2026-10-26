---
name: plan-creator
description: Create an implementation plan for a Kaapi feature, grounded in the codebase wiki (docs/wiki) and the blast-radius procedure (docs/wiki/domain-map.md). Use when the user wants an implementation plan, build plan, or execution plan for a feature, SRD, or task — the step between "spec agreed" and "start coding".
---

# Plan Creator

Turn a feature spec into an ordered, file-level implementation plan that an
engineer can execute without re-deriving the design. The plan is grounded in two sources, in this order:

1. **The wiki** (`docs/wiki/`) — what already exists: routes, tables, models,
   services, external boundaries. Codebase knowledge comes from here, not from
   memory or ad-hoc greps.
2. **The blast-radius procedure** (`docs/wiki/domain-map.md`) — what the change
   touches beyond the surfaces the spec names. Every plan includes a completed
   blast-radius table; a plan without one is not done.

## Input — two modes, nothing else

- **SRD mode:** the user supplies an SRD (a `features/<slug>/SRD.md` path or
  pasted content). The SRD is the spec; plan from it.
- **Standalone mode:** no SRD — the user's request itself is the spec. Plan
  directly from it, and mark every inferred requirement as an assumption in
  the plan's Open Questions section.

Do not demand an SRD, hunt for one, or block on other documents. An SRD is
optional input, never a prerequisite.

In both modes the spec supplies the *what*; the wiki and model files supply
the *shapes of what exists*. Never invent scope from the codebase, never
invent schema from the spec or memory.

## Workflow

1. **Read the input spec** end to end before opening anything else.

2. **Load the wiki, narrowly.** Open `docs/wiki/INDEX.md`, then ONLY the
   `docs/wiki/modules/*.md` page(s) for the domain(s) the feature touches.
   Follow a module page's deep-dive link into `docs/architecture/*.md` only if
   a genuine design question comes up; never bulk-load architecture docs.

3. **Run the blast-radius procedure** from `docs/wiki/domain-map.md`:
   - Name the primary entity(ies) the feature changes, in the map's vocabulary.
   - Collect 1-hop and 2-hop `consumed by` surfaces — tables, logical
     consumers, and the external consumers the map lists (Langfuse, the
     frontend console, provider batch APIs, object storage).
   - For every surface the spec does not address, ask the user:
     in scope / deferred / out of scope. Never silently include or exclude.
   - Record every decision in the plan's Blast Radius table.

4. **Verify schema against code.** The module page maps each table to its
   `models/*.py` file. Read the model files for every table the plan touches
   before writing any column, field, or relationship into the plan. Before
   planning a NEW table or config shape, confirm via `domain-map.md` + model
   files that nothing existing fits — reuse beats invention.

5. **Map the work onto Kaapi's layers.** Steps follow the dependency spine
   from `.claude/conventions/`: model → migration → crud → service → route,
   plus the Celery task if there is background work, then tests. Consult the
   relevant `.claude/conventions/{model,crud,service,route,migration,celery}.md`
   only to confirm a placement question, not wholesale.

6. **Write the plan** using the template below. Every step names concrete
   files (existing paths verified via the wiki page; new paths following the
   module's existing layout) and states what changes in each.

7. **Output** to `features/<slug>/PLAN.md`, alongside the feature's `SRD.md` /
   `PRD.md` if they exist — reuse the existing slug exactly; do not invent a
   parallel folder. If no `features/<slug>/` folder exists, create one with a
   short kebab-case slug. The plan is the deliverable; this skill does not
   execute it or hand it to any other skill or agent — an orchestrating skill
   may invoke plan-creator and consume the `PLAN.md`, but that wiring lives
   outside this skill.

## Plan template

```markdown
# <Feature Display Name> — Implementation Plan

Source spec: <path or "verbal description, see Open Questions">

## Summary

<2-4 sentences: what is being built and the shape of the change.>

## Blast Radius

Primary entities: <entities in domain-map vocabulary>

| Surface | Hop | Impact | Decision |
|---|---|---|---|
| <table / consumer> | 1 | <what changes for it> | in scope / deferred / out of scope |

<External consumers row(s) always present — Langfuse, frontend console,
provider batch APIs, object storage — even if the decision is "unaffected".>

## Steps

### 1. <Layer>: <one-line intent>
- Files: `app/...` (change) / `app/...` (new)
- <What changes, concretely: columns, function signatures, endpoint shape.>
- Depends on: <step # or "nothing">

<repeat per step, in dependency order>

## Migration

<Table/column DDL summary. Note: rev-id computed at generation time per
CLAUDE.md (highest NNN in app/alembic/versions/ + 1).>

## Tests

<Behaviors to cover, per app/tests/ layout; which HTTP boundaries get mocked.>

## Open Questions

<Unresolved decisions and inferred assumptions. Empty section is deleted.>
```

## Quality Checklist

Load these as todos at the start; verify each before output:

- [ ] Spec read first; every planned behavior traces to the spec or is flagged
      in Open Questions.
- [ ] Wiki loaded before planning: `docs/wiki/INDEX.md` + the touched module
      page(s); plan's paths, table names, and endpoint shapes cross-checked
      against them.
- [ ] Blast radius run against `docs/wiki/domain-map.md`: 1-hop + 2-hop
      consumers collected; every surface the spec skips confirmed with the
      user; decisions recorded in the Blast Radius table, none silently
      dropped; external consumers row(s) present.
- [ ] Schema derived from code: model files Read for every touched table;
      new tables/shapes justified by nothing existing fitting.
- [ ] Steps in dependency-spine order (model → migration → crud → service →
      route → Celery → tests), each naming concrete files.
- [ ] Kaapi conventions respected: `inserted_at`/`updated_at`,
      `organization_id` + `project_id` on multi-tenant tables, snake_case,
      naming matches existing code (grep before coining a new name).
- [ ] Output at `features/<slug>/PLAN.md`, alongside existing SRD/PRD.
- [ ] No fact stated twice; empty sections deleted; no filler.
- [ ] After any revision, grep the plan for superseded names (fields, tables,
      endpoints, step titles) — zero hits across every section.

## Rules

- **Wiki before greps.** Exploratory grepping is a fallback for what the wiki
  doesn't cover, not the starting point. If the wiki page is missing or stale
  for a touched module, say so — and note that the eventual PR must update it
  (wiki maintenance rule).
- **A plan step is checkable.** Each step names files and the concrete change;
  "update the service layer" is not a step. If you can't name the file, you
  haven't planned it yet.
- **Reuse existing models and shapes** at the highest-level wrapper that fits.
  A reviewer asking "can't we reuse X?" is a planning failure.
- **Plan the migration and the wiki update as steps**, not afterthoughts: a
  schema change carries an Alembic migration step; any change to a module's
  routes/tables/models/services carries a step updating that module's wiki
  page (and `domain-map.md` if entities/edges changed) in the same PR.
- **No em dashes in the output plan.** Commas, periods, or parentheses.
- **Don't design in the plan what the spec already decided** — reference the
  SRD section instead of restating it. The plan adds sequencing, file mapping,
  and blast-radius decisions, not a second copy of the spec.
