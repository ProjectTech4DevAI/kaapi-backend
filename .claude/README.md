# AI-assisted development (`.claude/`)

How features get designed and built in kaapi-backend with Claude Code. The system is **skills +
convention docs** — no specialist subagents. Skills carry the *workflow*; convention docs carry
the *house style*; a domain map carries *impact analysis*. Every stage loads only what it needs.

## The feature lifecycle

A feature moves through four stages, each driven by a skill. Each stage consumes the previous
one's output — run them in order, but skip any that don't apply (a small change can go straight to
`feature-builder`).

| Stage | Skill | Produces | Answers |
|---|---|---|---|
| 1. PRD (product spec) | `start-prd` | `features/<slug>/PRD.md` | *why* are we building this, and for whom? (no tech) |
| 2. SRD (software spec) | `srd-creator` | `features/<slug>/SRD.md` | *what* must the software do — endpoints, schema, scope, **+ blast radius** |
| 3. Execute SRD (build) | `feature-builder` | the code | model → crud → service → route, then migration / celery / tests |
| 4. Review | `/pr-review` | review notes | convention / security / correctness gate on the diff |

Everything for one feature lives together in `features/<slug>/` (PRD, SRD), reusing the same
kebab-case slug.

## How to run it

Invoke a skill by name (e.g. type `/srd-creator`), or just describe the task — the main agent
routes to the matching skill. A typical end-to-end run:

1. **Discuss the idea**, then `/start-prd` → writes `features/<slug>/PRD.md`.
2. `/srd-creator` → writes `features/<slug>/SRD.md` (the testable spec). As part of this it reads
   `docs/domain-map.md` and does **blast-radius analysis** — it stops and asks you about every
   impacted surface the spec didn't address (in scope / deferred / out of scope), so a downstream
   surface never gets silently dropped. `/grill-me` to stress-test the spec.
3. `/feature-builder` → **executes the SRD**: walks the dependency spine, loading each layer's
   convention doc before writing that layer, then the migration / Celery tasks / tests.
4. `/pr-review` → the pre-merge gate.

For a large feature you can dispatch a **general-purpose subagent per phase** (schema-spine →
migration → tests), passing forward only signatures, paths, and the next migration rev-id — the
`feature-builder` skill behaves the same inline or inside a subagent.

## Code conventions

All house style lives in **`.claude/conventions/*.md`** — one doc per layer. The
**`backend-conventions`** skill is the index: it maps each layer to its doc and tells you to load
the relevant ones *before* writing or reviewing that layer. `feature-builder` loads them while
building — **one source of truth, so the code never drifts from house style.**

| Concern | Doc | Concern | Doc |
|---|---|---|---|
| Cross-cutting (always) | `cross-cutting.md` | Migrations | `migration.md` |
| Models | `model.md` | Celery tasks | `celery.md` |
| CRUD | `crud.md` | Tests | `test.md` |
| Services | `service.md` | Error handling | `error-handling.md` |
| Routes | `route.md` | | |

The terse cross-cutting summary in `CLAUDE.md` defers to `cross-cutting.md` — edit the doc, not
just the summary.

## The domain map (`docs/domain-map.md`)

The source of truth for **blast-radius analysis**: which product surfaces consume which, so a
change to one surface doesn't silently break a downstream one. `srd-creator` traverses its
`Consumed by` edges (1-hop then 2-hop) while writing the SRD's **Impact / Blast Radius** section.
It's a dated snapshot — `srd-creator` reconciles it against live code and flags drift, but refresh
it as surfaces are added.

## Directory map

```
.claude/
├── README.md            ← this file
├── CLAUDE.md            project context loaded into every session (terse; defers to the docs)
├── skills/
│   ├── start-prd/       PRD writer
│   ├── srd-creator/     SRD writer + blast-radius analysis (+ reference/ template & guide)
│   ├── backend-conventions/  conventions index/loader
│   ├── feature-builder/ the build workflow (executes an SRD)
│   └── grill-me/        stress-test a spec/design
└── conventions/         cross-cutting + per-layer code conventions (the source of truth)
docs/
├── domain-map.md        product surfaces + consumer edges (blast radius)
└── architecture/        deep-dive architecture docs per subsystem
features/<slug>/         PRD.md · SRD.md for each feature
```

## Maintaining the system

- **Convention changed?** Edit the doc in `.claude/conventions/`. Then sync the `/pr-review`
  checklist, which mirrors these docs as a self-contained review list.
- **New product surface?** Add it to `docs/domain-map.md` (surface + its consumes/consumed-by
  edges) so future blast-radius analysis sees it.
- **New layer or skill?** Add the convention doc, register it in the `backend-conventions` index,
  and reference it from `feature-builder`.
