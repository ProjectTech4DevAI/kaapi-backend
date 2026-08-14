Submit an assessment against a saved LLM configuration; results are delivered by webhook.

An assessment grades one or more items with a config-defined LLM call, optionally gated by
pre-filters (topic relevance / duplicate detection). The run mode is **inferred from the input
shape** — you do not pass a mode flag.

**Key Features:**
* Method inferred from the input: a `data` list ⇒ **BATCH**, a single `query` (+ optional
  `attachments`) ⇒ **RESPONSE**.
* Pins to a saved config **version** (`config.id` + `config.version`); the config must be tagged
  `ASSESSMENT` (see the config-create docs for the `config_blob` assessment shape).
* Optional pre-filters run before the grading call and can gate it per item.
* **Webhook-only delivery** — the result is POSTed to the request's `callback_url` on completion.
  There is no status or result poll endpoint.
* `request_metadata` is echoed back unchanged in the callback for correlation.

> **RESPONSE mode is not wired yet** — a single-object input currently returns `501 Not Implemented`.
> Only BATCH (a `data` list) is processed today.

---

## Request

```json
{
  "config": { "id": "3f9b2c10-0000-4a11-8c22-1a2b3c4d5e6f", "version": 1 },
  "input": {
    "query": "Grade the answer sheet for roll number {roll_number}, student {student_name}.",
    "data": [
      {
        "roll_number": "1",
        "student_name": "Aarav Sharma",
        "answer_sheet": "https://cdn.example.com/sheets/1.pdf"
      },
      {
        "roll_number": "2",
        "student_name": "Priya Nair",
        "answer_sheet": "https://cdn.example.com/sheets/2.pdf"
      }
    ]
  },
  "callback_url": "https://your-app.example.com/webhooks/assessment",
  "request_metadata": { "batch": "class7-term1" }
}
```

**Fields:**

* `config` (required) — pins the saved config version to run.
  * `id` (UUID, required) — the config id.
  * `version` (int ≥ 1, required) — the config version. Must reference a config tagged `ASSESSMENT`.
* `input` (required) — one of two shapes; the shape selects the method:
  * **BATCH** — `{ "query": "<template>", "data": [ {<column>: <value>}, ... ] }`
    * `query` (required, non-empty) — a template with `{column}` placeholders substituted per row.
    * `data` (required, ≥ 1 row) — submission rows. Each row is a flat `column -> string` map. The
      config's `assessment.params.input_schema` is **mandatory** and defines every column and its
      `type` (`text` / `image` / `pdf`, with an attachment `format`). Every row is validated against
      it: each declared column must be present, no undeclared columns are allowed, and `image`/`pdf`
      columns must carry a URL. A row that does not match fails with `422` (see Errors).
  * **RESPONSE** — `{ "query": "<text>", "attachments": [ ... ] }` *(deferred — returns 501)*.
* `callback_url` (required) — the webhook the result is POSTed to on completion.
* `request_metadata` (optional) — arbitrary object passed through unchanged in the callback.

The two input shapes are strictly discriminated: a body carrying `data` is BATCH, one carrying
`attachments` (or query-only) is RESPONSE. Neither accepts the other's distinguishing field.

---

## Response (submit acknowledgement)

`200` — a flat ack that the assessment was accepted and queued. It does **not** carry results;
those arrive on the webhook.

```json
{
  "success": true,
  "data": {
    "assessment_id": "8a2a7bc1-359f-4b99-a94b-ecf7621a0704",
    "status": "PROCESSING",
    "message": "Your assessment is being processed",
    "inserted_at": "2026-08-07T10:15:30Z",
    "updated_at": "2026-08-07T10:15:30Z"
  },
  "error": null,
  "metadata": null
}
```

* `assessment_id` (UUID) — correlate this with the webhook payload.
* `status` — one of `PENDING`, `PROCESSING`, `COMPLETED`, `COMPLETED_WITH_ERRORS`, `FAILED`.
* `message`, `inserted_at`, `updated_at`.

---

## Webhook callback

On completion the platform POSTs this payload to `callback_url`:

```json
{
  "assessment_id": "8a2a7bc1-359f-4b99-a94b-ecf7621a0704",
  "status": "COMPLETED",
  "data": {
    "total_items": 2,
    "counts": { "assessed": 2, "filtered": 0, "errors": 0 },
    "items": [
      {
        "output": {
          "assessment": { "grade": "B", "score": 62, "feedback": "..." },
          "pre_filter": {
            "topic_relevance": { "verdict": true, "reasoning": "..." },
            "duplicate_detection": null
          }
        },
        "error": null
      }
    ]
  },
  "request_metadata": { "batch": "class7-term1" }
}
```

**Callback fields:**

* `assessment_id`, `status` — mirror the submit ack; `status` is terminal here
  (`COMPLETED`, `COMPLETED_WITH_ERRORS`, or `FAILED`).
* `request_metadata` — the object you sent, echoed unchanged.
* `data` — the result body, keyed by method:
  * **BATCH** — an object with `total_items`, `counts`, and one `items[]` entry per input row.
    * `counts` — `assessed` (graded rows), `filtered` (gated out by a pre-filter), `errors`.
    * each item:
      * `output.assessment` — a dict when the config emits structured output (`json_output_schema`),
        a raw string for free-text output, or `null` for a gated/failed row.
      * `output.pre_filter` — per-item `topic_relevance` / `duplicate_detection` verdicts, each
        `{ "verdict": bool, "reasoning": str }` or `null` if that pre-filter was not configured.
      * `error` — a per-row error string, or `null`.
  * **RESPONSE** — a single item object with the same `output` / `error` shape.

---

## Errors

* `501 Not Implemented` — RESPONSE-mode input (single object) is not wired yet; send a BATCH `data` list.
* `422 Unprocessable Entity` — the body failed validation (e.g. `config.id`/`config.version` missing,
  `callback_url` missing, or an input shape that carries both `data` and `attachments`), or a
  submission row does not match the config's `input_schema` (a missing declared column, an
  undeclared extra column, or a non-URL `image`/`pdf` value). The row index is named in the error.
