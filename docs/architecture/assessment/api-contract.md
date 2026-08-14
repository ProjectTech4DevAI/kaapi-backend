# API Contract — `POST /assessments`

Precise request and response shapes for the BATCH assessment API. For a
walkthrough with context, see the [overview](README.md).

Everything is delivered by **webhook** — there is no status or result poll
endpoint. RESPONSE-shaped input returns `501` (WIP).

**Sample input / output JSON files:**
https://drive.google.com/drive/folders/1BCaauUuXr9DaZTWI-_-x101SDT4ktwp5?usp=share_link

---

## Request

`POST /assessments`

| Field | Type | Required | Notes |
|---|---|---|---|
| `config` | object | ✅ | which saved config version to run |
| `config.id` | UUID | ✅ | config id (must be tagged `ASSESSMENT`) |
| `config.version` | int ≥ 1 | ✅ | config version to pin |
| `input` | object | ✅ | a `data` list ⇒ BATCH; a bare `query` ⇒ RESPONSE (501) |
| `input.query` | string (non-empty) | ✅ | template; `{column}` placeholders filled per row |
| `input.data` | array (≥ 1) | ✅ | rows; each row is a flat `{ column: string }` object |
| `callback_url` | URL (**HTTPS**) | ✅ | webhook the result is POSTed to |
| `request_metadata` | object | optional | echoed back unchanged in the result |

Rules:

- **Strict input** — no extra keys are allowed on `input`; a body carrying both
  `data` and `attachments` is rejected.
- **Rows match the config's `input_schema`** — every declared column present, no
  undeclared columns, `image`/`pdf` values must be URLs. Otherwise `422`.
- **`callback_url`** must be HTTPS and public (private/loopback hosts are rejected).

```json
{
  "config": { "id": "a9015dbf-…", "version": 1 },
  "input": {
    "query": "Grade {answer_sheet} for submission {submission_id}.",
    "data": [
      { "submission_id": "s1", "answer_sheet": "https://cdn.example.com/s1.jpg" }
    ]
  },
  "callback_url": "https://your-app.example.com/webhooks/assessment",
  "request_metadata": { "batch": "class7-term1" }
}
```

### Building the batch input

The `input` object is built from your configuration's `input_schema`:

1. **One object per item** goes in `input.data`. Each object's keys are the column
   names declared in the config's `input_schema`, and the values are strings.
2. **Attachment columns** (`image` / `pdf`) take a URL string; text columns take
   plain text.
3. **`input.query`** is a template. Any `{column}` placeholder is replaced with
   that row's value at grading time, so one template applies to every row.
4. **Match the schema exactly** — every declared column present, no extra columns.

Example: for `input_schema = { submission_id: text, answer_sheet: image(url) }`,
each row is `{ "submission_id": "...", "answer_sheet": "https://..." }` and the
`query` can reference `{submission_id}` and `{answer_sheet}`.

---

## Response — submit acknowledgement (`200`)

Returned immediately; contains no results. Wrapped in the standard envelope
`{ success, data, error, metadata }`.

| Field (`data`) | Type | Notes |
|---|---|---|
| `assessment_id` | UUID | correlate with the webhook |
| `status` | enum | `PROCESSING` on accept |
| `message` | string | human-readable |
| `inserted_at` / `updated_at` | timestamp | ISO-8601 |

```json
{
  "success": true,
  "data": {
    "assessment_id": "8a2a7bc1-…",
    "status": "PROCESSING",
    "message": "Your assessment is being processed",
    "inserted_at": "2026-08-12T10:15:30Z",
    "updated_at": "2026-08-12T10:15:30Z"
  },
  "error": null,
  "metadata": null
}
```

---

## Webhook — the result (POST to `callback_url`)

Delivered once, on completion.

| Field | Type | Notes |
|---|---|---|
| `assessment_id` | UUID | matches the ack |
| `status` | enum | terminal (see below) |
| `data` | object | the `AssessmentBatchResult` (BATCH) |
| `request_metadata` | object \| null | echoed from the request |

`data` (`AssessmentBatchResult`):

| Field | Type | Notes |
|---|---|---|
| `total_items` | int | number of input rows |
| `counts.assessed` | int | rows graded |
| `counts.filtered` | int | rows gated out by a pre-filter |
| `counts.errors` | int | rows with an error |
| `items` | array | one `AssessmentResult` per input row, in order |

`items[]` (`AssessmentResult`):

| Field | Type | Notes |
|---|---|---|
| `output.assessment` | object \| string \| null | your `json_output_schema` filled in; string for free-text; `null` if gated out / failed |
| `output.pre_filter.topic_relevance` | `{verdict: bool, reasoning: string}` \| null | null if not configured |
| `output.pre_filter.duplicate_detection` | `{verdict: bool, reasoning: string}` \| null | null if not configured |
| `error` | string \| null | per-row error |

```json
{
  "assessment_id": "8a2a7bc1-…",
  "status": "COMPLETED",
  "data": {
    "total_items": 2,
    "counts": { "assessed": 1, "filtered": 1, "errors": 0 },
    "items": [
      {
        "output": {
          "assessment": { "score": 20, "feedback": "…" },
          "pre_filter": { "topic_relevance": { "verdict": true, "reasoning": "…" } }
        },
        "error": null
      },
      {
        "output": {
          "assessment": null,
          "pre_filter": { "topic_relevance": { "verdict": false, "reasoning": "off-topic" } }
        },
        "error": null
      }
    ]
  },
  "request_metadata": { "batch": "class7-term1" }
}
```

---

## Status values

| Status | Meaning |
|---|---|
| `PENDING` | accepted, not started |
| `PROCESSING` | grading in progress (the ack status) |
| `COMPLETED` | all rows graded, no errors |
| `COMPLETED_WITH_ERRORS` | finished, some rows errored |
| `FAILED` | the run failed |

`status` lives on the envelope only — it is never duplicated inside `data`.

## Error codes (at submit)

| Code | When |
|---|---|
| `422` | invalid body, or a row doesn't match `input_schema`, or a non-HTTPS/private `callback_url` |
| `404` | config id not found |
| `501` | RESPONSE-shaped input (single `query`) — WIP |
| `503` | failed to dispatch for processing (retry) |
