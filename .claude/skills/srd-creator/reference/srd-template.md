<!-- Filename: "<Feature Name> SRD.md" (Title Case, spaces, " SRD" suffix). H1 below = filename minus ".md". -->

# <Feature Name> SRD

## Introduction & Purpose

<1–2 sentences: what capability this SRD defines and for which system (Kaapi).>

<The problem / motivation: what is painful today, and for whom. Name early users
if known.>

<What the feature produces, at minimum:>
- <output 1>
- <output 2>

<Phasing. State what is in scope now vs deferred.>
- **Phase 1:** <what we build now>
- **Phase 2+:** <what comes later>

<One line on intent / quality bar, e.g. repeatable, comparable, auditable.>

## Resources
<!-- optional, delete if empty -->
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

## Detailed Design (Execution Flow)

<!-- Render each flow to assets/, then leave a band naming that file. No inline embed,
no mermaid, no HTML, no emoji. Keep it high-level: actors and behavior, not
function/variable names. -->

### <Flow A: e.g. Main run>

---

**>> PLACE IMAGE HERE: `assets/flow-a.png`, <Flow A, system-level: User, services, arrows in/out>.**

---

<Brief text for what the diagram can't carry: failure isolation / idempotency.>

<!-- Add a second flow ONLY if it is genuinely distinct and a diagram beats prose.
Plain CRUD / config resolution usually needs prose + the Endpoints section, no image.
-->

### <Flow B: e.g. Config / resolution, prose only if a diagram adds nothing>

<Brief prose: resolution + revert rules.>

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

<!-- Tables only. No diagram, no image here (the one SRD image is the execution flow).
The column tables below ARE the schema; mark reused vs new in each table's heading. -->

### `<table_name>`
<One-line purpose. All multi-tenant tables include organization_id and project_id.>

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | INTEGER (PK) | NO | auto-increment | Unique identifier |
| <col> | <type> | <YES/NO> | <default> | <description> |
| organization_id | INTEGER | NO | n/a | Reference to the organization |
| project_id | INTEGER | NO | n/a | Reference to the project |
| inserted_at | TIMESTAMP | NO | now() | Created timestamp |
| updated_at | TIMESTAMP | NO | now() | Last-updated timestamp |

**Constraints:**
- `<uq_constraint_name>`: UNIQUE on (<cols>)
- FK <col> → <table>.<col>
- Index on <col>

<!-- Repeat per table. For reused tables, list only added columns + the backfill plan. -->

## Configuration
<!-- optional, delete if empty -->

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| <SETTING_NAME> | <type> | <default> | <description> |

## Design Decisions / Known Limitations
<!-- optional, delete if empty -->
- **<decision>:** <rationale>
- **Known limitation:** <gap to revisit>
