# AI Assessments — Getting Started

**An assessment uses an LLM to grade your items against a rubric and gives you back a structured result** (scores, reasoning, feedback) for every item — not free text, but a fixed JSON shape you choose.

You give Kaapi two things:

1. A **config** — your rubric (the grading instructions), the model to use, and the exact result shape you want back.
2. Your **items** — the rows you want graded (text and/or image/PDF URLs).

Kaapi grades every item and delivers the results to your **webhook**.

---

## The whole flow in three steps

| Step | You do | Kaapi does |
|---|---|---|
| **1. Create a config** | Save an `ASSESSMENT` config once (`POST /configs`) | Stores it, versioned |
| **2. Submit items** | `POST /assessments` with your rows + a `callback_url` | Returns an `assessment_id`, starts grading in the background |
| **3. Get results** | Wait for the webhook | POSTs the finished results to your `callback_url` |

You never poll or wait on the request — submitting returns immediately, and the results arrive later at your webhook.

![BATCH assessment flow](assets/batch-flow.png)

**BATCH is fully batched.** Both stages run as provider **batch jobs** — the
pre-filters run as a batch, and the assessment runs as a batch. Results are
delivered to your **webhook** when everything completes (no polling).

---

## Two methods (Kaapi picks for you)

You never set a "mode". Kaapi looks at your input and decides:

| Method | When | Input shape | Status |
|---|---|---|---|
| **BATCH** | Many items at once | `data` is a list of rows | ✅ Available |
| **RESPONSE** | A single item, fast | a single item's `attachments` (no `data`) | 🚧 WIP (returns `501` today) |

This guide covers **BATCH**, the method that is live.

---

## Supported models

Pick the provider per config (and per pre-filter):

| Provider | Value in config | Status |
|---|---|---|
| OpenAI | `openai` | ✅ |
| Google (AI Studio / Gemini) | `google` | ✅ |
| Anthropic (Claude) | `anthropic` | ✅ |
| Google Cloud / Vertex | — | 🚧 WIP |

---

## Where to go next

1. **[Configuration and versioning](configuration-and-versioning.md)** — build your rubric, choose the model, define the result shape, and manage versions.
2. **[API contract](api-contract.md)** — request/response fields, types, status values, and error codes.
