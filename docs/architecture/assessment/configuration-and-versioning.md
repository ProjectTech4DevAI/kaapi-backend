# Configuration and Versioning

A **configuration** is your saved grading setup: the rubric, the model, the input
columns, and the structured result shape. You create it once, and every
assessment references it. Configurations are **versioned** so you can iterate
safely — each change adds a new version and never disturbs results that already
ran.

This guide explains each component with examples.

---

## The tag

Every configuration carries a **tag** that fixes its shape. An assessment
configuration must use:

```
tag = ASSESSMENT
```

- The tag is chosen at creation and **cannot be changed** afterwards.
- It drives validation: a config tagged `ASSESSMENT` must match the assessment
  shape below, and a wrong shape is rejected with `422`.

---

## The shape (`config_blob`)

An `ASSESSMENT` config has two parts:

```json
{
  "name": "answer-sheet grader",
  "tag": "ASSESSMENT",
  "config_blob": {
    "pre_filters": { "...": "optional checks before grading" },
    "assessment":  { "...": "the grading call (required)" }
  }
}
```

- **`assessment`** (required) — the grading call.
- **`pre_filters`** (optional) — quick checks that run before grading.

---

## Component 1 — `assessment` (required)

The grading call.

| Field | What it is |
|---|---|
| `provider` | `openai`, `google`, or `anthropic` |
| `type` | always `"text"` |
| `params.model` | the model id, e.g. `gpt-4o` |
| `params.instructions` | the **rubric** — how to grade (system prompt) |
| `params.input_schema` | **required** — the columns you will submit, and their types |
| `params.json_output_schema` | the **result shape** returned per item |

```json
"assessment": {
  "provider": "openai",
  "type": "text",
  "params": {
    "model": "gpt-4o",
    "instructions": "Grade each section out of 25 against the rubric and give brief feedback.",
    "input_schema": { "...": "see Component 2" },
    "json_output_schema": { "...": "see Component 3" }
  }
}
```

---

## Component 2 — `input_schema` (the columns)

`input_schema` declares the columns every submitted row will have. It is
**mandatory** and must list at least one column; every column must declare a
`type`.

| `type` | Value in a row |
|---|---|
| `text` | any text |
| `image` | an image URL |
| `pdf` | a PDF URL |

Attachment columns (`image` / `pdf`) take a URL — add `"format": "url"`.

```json
"input_schema": {
  "submission_id": { "type": "text" },
  "answer_sheet":  { "type": "image", "format": "url" }
}
```

Every column declared here must be present in every submitted row (see the
[API contract](api-contract.md)).

---

## Component 3 — `json_output_schema` (the result shape)

A JSON Schema `object` describing exactly what comes back per item. Whatever you
define is what the model fills in — so you get structured scores/feedback, not
free text.

```json
"json_output_schema": {
  "type": "object",
  "properties": {
    "score":    { "type": "integer" },
    "feedback": { "type": "string" }
  },
  "required": ["score", "feedback"]
}
```

---

## Component 4 — `pre_filters` (optional)

Checks that run before grading. Each pre-filter is its own small LLM call with
its own model and criteria.

| Pre-filter | Purpose | Default on a "no" |
|---|---|---|
| `topic_relevance` | Is the item on-topic / worth grading? | Skips grading for that item (`stop_on_fail: true`) |
| `duplicate_detection` (WIP) | Is the item a duplicate? | Records only, still grades (`stop_on_fail: false`) |

Each pre-filter carries:

| Field | What it is |
|---|---|
| `provider` | `openai` / `google` / `anthropic` (defaults to `openai`) |
| `params.model` | the model for this check (a cheaper one is fine) |
| `params.instructions` | **required** — the criteria for the check |
| `stop_on_fail` | `true` = a "no" skips grading for that item; `false` = record only |
| `knowledge_base_id` | *(duplicate_detection only)* the corpus to compare against |

```json
"pre_filters": {
  "topic_relevance": {
    "provider": "openai",
    "params": {
      "model": "gpt-4o-mini",
      "instructions": "Accept only a photo of a hand-drawn answer sheet; reject blank or off-topic images."
    },
    "stop_on_fail": true
  }
}
```

An item that fails a gate still appears in the results (with an empty grading
output and its pre-filter verdict), so you can see why it was skipped.

---

## Full example

A grader on OpenAI — a `topic_relevance` gate on `gpt-4o-mini`, the assessment on
`gpt-4o`:
[`config_openai.json`](https://github.com/ProjectTech4DevAI/kaapi-backend/blob/feat/doc-assessment-architecture/z_assessment_test/tap_test/config_openai.json)

---

## Versioning

Configurations are versioned. You never overwrite one — you add a version.

| Action | Endpoint |
|---|---|
| Create a configuration | `POST /configs` (with `tag = ASSESSMENT`) |
| Add a new version | `POST /configs/{config_id}/versions` |
| List all versions | `GET /configs/{config_id}/versions` |
| Get a specific version | `GET /configs/{config_id}/versions/{version_number}` |

How it works:

- **Create** a configuration once; it starts at version 1.
- **Iterate** by adding versions — change the rubric, swap the model, adjust the
  result shape. Each save is a new version; earlier versions remain intact.
- **The tag stays fixed** across versions, so every version keeps the assessment
  shape.
- **An assessment pins the exact version it ran with** (`config_id` +
  `config_version`). Editing the configuration later never changes a result that
  has already completed — reproducible by design.
- **Track** versions with the list/get endpoints and choose which
  `config_version` to submit.

You reference the chosen `config_id` and `config_version` when you submit an
assessment — see the [API contract](api-contract.md).
