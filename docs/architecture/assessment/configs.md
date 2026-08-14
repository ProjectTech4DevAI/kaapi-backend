# Writing an Assessment Config

A **config** is your saved grading setup: the rubric, the model, the columns you'll send, and the result shape you want back. You save it once and reuse it for every assessment.

A config has two parts:

- **`assessment`** (required) — the grading call itself.
- **`pre_filters`** (optional) — quick checks that run *before* grading (e.g. "is this on-topic?").

You save a config with `POST /configs` and the tag **`ASSESSMENT`**. A wrong shape is rejected with `422`.

---

## Part 1 — the `assessment` call (required)

| Field | What it is |
|---|---|
| `provider` | `openai`, `google`, or `anthropic` |
| `type` | always `"text"` |
| `params.model` | the model id, e.g. `gpt-4o` |
| `params.instructions` | your **rubric** — how to grade (the system prompt) |
| `params.input_schema` | **required** — the columns your rows will have, and each column's type |
| `params.json_output_schema` | the **result shape** — the JSON object you want back per item |

### `input_schema` — describe your columns

`input_schema` is **mandatory** and must list at least one column. Every column must declare a `type`:

| `type` | Meaning |
|---|---|
| `text` | a plain text value |
| `image` | an image URL |
| `pdf` | a PDF URL |

Attachment columns (`image` / `pdf`) take a **URL** (add `"format": "url"`). Every column you declare here must be present in every row you submit (see [running an assessment](running-an-assessment.md)).

### `json_output_schema` — the result shape

A normal JSON Schema `object`. Whatever you define here is exactly what comes back per item, filled in by the model — so you get structured scores/feedback instead of free text.

---

## Part 2 — `pre_filters` (optional)

Pre-filters run before grading. Each one is its own small LLM call with its own model and criteria.

| Pre-filter | What it checks | Default behavior on a "no" |
|---|---|---|
| `topic_relevance` | Is the item on-topic / worth grading? | **Blocks** grading for that item (`stop_on_fail: true`) |
| `duplicate_detection` 🚧 | Is the item a duplicate? | **Records only**, still grades (`stop_on_fail: false`) |

Each pre-filter carries:

| Field | What it is |
|---|---|
| `provider` | `openai` / `google` / `anthropic` (defaults to `openai`) |
| `params.model` | the model for this check (a cheaper/faster one is fine) |
| `params.instructions` | **required** — the criteria for this check |
| `stop_on_fail` | `true` = a "no" skips grading for that item; `false` = just record the verdict |
| `knowledge_base_id` | *(duplicate_detection only)* the corpus to compare against |

A failed **gate** (`stop_on_fail: true`) skips grading for that item, but the item still appears in your results with its pre-filter verdict.

---

## Full example

A grader on OpenAI: a `topic_relevance` gate on `gpt-4o-mini`, and the assessment on `gpt-4o`.

```json
{
  "name": "answer-sheet grader",
  "tag": "ASSESSMENT",
  "commit_message": "Initial version",
  "config_blob": {
    "pre_filters": {
      "topic_relevance": {
        "provider": "openai",
        "params": {
          "model": "gpt-4o-mini",
          "temperature": 0.1,
          "instructions": "Accept only a photo of a hand-drawn answer sheet. Reject blank, off-topic, or internet images."
        },
        "stop_on_fail": true
      }
    },
    "assessment": {
      "provider": "openai",
      "type": "text",
      "params": {
        "model": "gpt-4o",
        "instructions": "Grade each section out of 25 against the rubric and give brief feedback.",
        "input_schema": {
          "submission_id": { "type": "text" },
          "answer_sheet":  { "type": "image", "format": "url" }
        },
        "json_output_schema": {
          "type": "object",
          "properties": {
            "score":    { "type": "integer" },
            "feedback": { "type": "string" }
          },
          "required": ["score", "feedback"]
        }
      }
    }
  }
}
```

A full working sample lives at [`z_assessment_test/tap_test/config_openai.json`](../../../z_assessment_test/tap_test/config_openai.json).

---

## Versioning

Every save creates a **new version**. An assessment pins the exact `config_id` + `config_version` it ran with, so editing a config later never changes results that already finished. When you submit, you choose which version to run.

Next: **[Running an assessment](running-an-assessment.md)**.
