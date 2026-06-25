# <Feature Name> SRD

## Introduction & Purpose

<1–2 sentences: what capability this SRD defines and for which system (Kaapi).>

<The problem / motivation — what is painful today, and for whom. Name early users
if known.>

<What the feature produces, at minimum:>
- <output 1>
- <output 2>

<Phasing. State what is in scope now vs deferred.>
- **Phase 1:** <what we build now>
- **Phase 2+:** <what comes later>

<One line on intent / quality bar — e.g. repeatable, comparable, auditable.>

## Resources
<!-- optional — delete if empty -->
- <Related SRD / link>
- <External API documentation / link>
- <Research / design notes / link>

## Goals
- <Outcome 1>
- <Outcome 2>
- <Outcome 3>

## Assumptions & Constraints
- **Out of scope:** <what this SRD explicitly does not change>
- **Limits / caps:** <size caps, thresholds, rate limits>
- **Data assumptions:** <required input format, change frequency, etc.>
- **Reuse:** <existing tables/services reused; "no new tables" if applicable>
- **Pricing:** <billing notes if a paid external API is involved>
- **Starting provider/model:** <e.g. Google Gemini 2.5 Pro, gpt-4o>

## Impact / Blast Radius

Product surfaces this feature affects, traced from `docs/domain-map.md` (follow the primary
surface's `Consumed by` edges 1-hop, then 2-hop). Every impacted surface this feature touches but
does **not** change must carry a confirmed scope status — no TBDs.

| Surface | Hop | Why affected | Status | Notes |
|---------|-----|--------------|--------|-------|
| <surface> | 1 | <direct dependency / shared table / etc.> | in scope / deferred / out of scope | confirmed {date} |

- For every domain-map surface **not** listed, state explicitly why it's unaffected.
- Record any **drift** found while reconciling `docs/domain-map.md` against live code (map said X, code does Y).

## Detailed Design (Execution Flow)

<!-- Give each distinct flow its own numbered subsection. -->

### <Flow A — e.g. Upload Flow>
1. <step>
2. <step>

### <Flow B — e.g. Run Submission>
1. <step>
2. <step>

### <Flow C — e.g. Async / Polling>
1. <step — include skip/idempotency markers and retry behavior for async stages>

> Sequence Flow Diagram: <link or embed, if any>

## Functional Requirements (Testing)

| ID | What (user-facing behavior) | Acceptance criteria | Status |
|----|-----------------------------|---------------------|--------|
| FR-1 | <behavior> | <concrete, checkable condition> | Not Started |
| FR-2 | <behavior> | <concrete, checkable condition> | Not Started |
| FR-3 | <behavior> | <concrete, checkable condition> | Not Started |

## Endpoints

### `<METHOD> /<path>`
<One-line description.>

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| <field> | <type> | <Yes/No> | <default> | <description> |

```json
{
  "<field>": "<value>"
}
```

**Response:**

```json
{
  "<field>": "<value>"
}
```

**Error responses:**

| Status | Code | When |
|--------|------|------|
| 422 | <error_code> | <condition> |
| 409 | <error_code> | <condition> |

<!-- Repeat the endpoint block per endpoint. -->

## Database Schema

### `<table_name>`
<One-line purpose. All multi-tenant tables include organization_id and project_id.>

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER (PK) | NO | auto-increment | Unique identifier |
| <col> | <type> | <YES/NO> | <default> | <description> |
| organization_id | INTEGER | NO | — | Reference to the organization |
| project_id | INTEGER | NO | — | Reference to the project |
| inserted_at | TIMESTAMP | NO | now() | Created timestamp |
| updated_at | TIMESTAMP | NO | now() | Last-updated timestamp |

**Constraints:**
- `<uq_constraint_name>` — UNIQUE on (<cols>)
- FK <col> → <table>.<col>
- Index on <col>

<!-- Repeat per table. For reused tables, list only added columns + the backfill plan. -->

## Configuration
<!-- optional — delete if empty -->

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| <SETTING_NAME> | <type> | <default> | <description> |

## Design Decisions / Known Limitations
<!-- optional — delete if empty -->
- **<decision>:** <rationale>
- **Known limitation:** <gap to revisit>
