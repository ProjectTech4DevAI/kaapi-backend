# Standalone Guardrails API

Send a piece of text to Kaapi, choose which safety checks to run on it, and get back a cleaned-up version of the text (or a "blocked" verdict).

Use it when you run your own LLM workflow and only want Kaapi's guardrails. To call an LLM *with* guardrails, use `/api/v1/llm/call` with `input_guardrails` / `output_guardrails` in the config instead. Both accept the same validator shapes.

- **Endpoint:** `POST /api/v1/guardrails`
- **Auth:** `X-API-KEY: ApiKey <your key>` (project-scoped)
- **Mode:** asynchronous. You get a `job_id` immediately, and the result arrives on your `callback_url` or via `GET /api/v1/guardrails/{job_id}`.

A Postman collection with every example below is in [`standalone-guardrails.postman_collection.json`](standalone-guardrails.postman_collection.json).

---

## Quick start

**1. Submit text with one validator**

```bash
curl -X POST "$KAAPI_URL/api/v1/guardrails" \
  -H "X-API-KEY: $KAAPI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hi, I am Ravi. Mail me at ravi.k@example.com or call 9876543210.",
    "config": [{ "type": "pii_remover" }]
  }'
```

```json
{
  "success": true,
  "data": {
    "job_id": "fae56084-637e-4540-8ba0-836a1dbc81c3",
    "status": "PENDING",
    "message": "Guardrails are being applied; poll GET /guardrails/{job_id} for the result."
  }
}
```

**2. Fetch the result**

```bash
curl "$KAAPI_URL/api/v1/guardrails/fae56084-637e-4540-8ba0-836a1dbc81c3" \
  -H "X-API-KEY: $KAAPI_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "job_id": "fae56084-637e-4540-8ba0-836a1dbc81c3",
    "status": "SUCCESS",
    "guardrails_response": {
      "response": {
        "response_id": "230230e1-c2b6-489c-88cd-7d7e55dc486d",
        "output": {
          "type": "text",
          "content": { "format": "text", "value": "Hi, I am Ravi. Mail me at <EMAIL_ADDRESS> or call <PHONE_NUMBER>." }
        }
      },
      "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0 },
      "provider_raw_response": null
    },
    "error_message": null,
    "warnings": []
  }
}
```

The cleaned text is always at `data.guardrails_response.response.output.content.value`.

---

## Request body

| Field | Type | Required | What it does |
|---|---|---|---|
| `text` | string | yes | The text to check. Can be a user message or an LLM reply. Must not be empty. |
| `config` | array | yes | The validators to run, at least one. See [Choosing validators](#choosing-validators). |
| `callback_url` | string | no | Public `https://` URL that receives the result. Leave it out to poll instead. |
| `request_metadata` | object | no | Any JSON you want handed back in the callback's `metadata`, e.g. your conversation ID. Not returned when polling. |

---

## Choosing validators

Each entry in `config` is **one** of two shapes. Kaapi tells them apart by whether `validator_config_id` is present.

### A. Inline spec: describe the validator in the request

```json
{ "type": "pii_remover" }
{ "type": "pii_remover", "entity_types": ["EMAIL_ADDRESS"], "on_fail": "exception" }
```

`type` is the only required field. Every other setting has a sensible default. Good for quick starts and for settings that vary per request.

### B. Stored config: point at a validator you saved earlier

```json
{ "validator_config_id": "b7881a0e-698c-478c-b4c2-80a438b3d607" }
```

Create stored configs with `POST /api/v1/guardrails/validators/configs` and list them with `GET /api/v1/guardrails/validators/configs`. Good when several apps or teams should share one tuned setup.

### Shared options

| Option | Values | Default | Meaning |
|---|---|---|---|
| `on_fail` | `fix`, `exception`, `rephrase` | `fix` (`rephrase` for `topic_relevance_llm`) | What to do when the check trips. `fix`: return a cleaned version of the text. `exception`: block the text, so the job ends `FAILED` with a reason. `rephrase`: replace the text with a fallback message. |
| `is_enabled` | `true` / `false` | `true` | `false` keeps the entry in your payload but skips it. |

You can mix both shapes and list several validators in one request. They run as a chain, each one receiving the previous one's output.

---

## The validators

These five types are offered to tenants: usable inline and listed by `GET /api/v1/guardrails`. Seven more exist but need a stored config; see [Other validator types](#other-validator-types-stored-config-only). Results below are real outputs from a local run.

### `pii_remover`: mask personal data

| Param | Default |
|---|---|
| `entity_types` | `CREDIT_CARD`, `EMAIL_ADDRESS`, `IBAN_CODE`, `IP_ADDRESS`, `MEDICAL_LICENSE`, `PHONE_NUMBER`, `URL`, `IN_AADHAAR`, `IN_PAN`, `IN_PASSPORT`, `IN_VEHICLE_REGISTRATION`, `IN_VOTER` |

`PERSON`, `LOCATION` and `NRP` are also accepted but off by default.

| Request `config` | Output |
|---|---|
| `[{"type": "pii_remover"}]` | `Hi, I'm Ravi. Mail me at <EMAIL_ADDRESS> or call <PHONE_NUMBER>. My PAN is <IN_PAN>.` |
| `[{"type": "pii_remover", "entity_types": ["EMAIL_ADDRESS"]}]` | `Hi, I'm Ravi. Mail me at <EMAIL_ADDRESS> or call 9876543210. My PAN is ABCDE1234F.` |
| `[{"type": "pii_remover", "on_fail": "exception"}]` | Job `FAILED`, `error_message: "PII detected in the text."` |

### `uli_slur_match`: redact slurs (English + Hindi/Hinglish)

| Param | Default | Values |
|---|---|---|
| `languages` | `["en", "hi"]` | language codes |
| `severity` | `all` | `low`, `medium`, `high`, `all` |

| Request `config` | Output |
|---|---|
| `[{"type": "uli_slur_match"}]` | `tu ek dum [REDACTED_SLUR] hai, bakwaas mat kar.` |
| `[{"type": "uli_slur_match", "on_fail": "exception"}]` | Job `FAILED`, `error_message: "Mentioned toxic words: <word>"` |

> **Known issue:** this validator currently lowercases the whole text, including other validators' placeholders when chained (`<EMAIL_ADDRESS>` becomes `<email_address>`).

### `content_safety`: catch toxic or abusive messages

A multilingual toxicity classifier. If the text is flagged, the whole message is replaced with a fallback message.

| Param | Default |
|---|---|
| `threshold` | `0.5` (higher means less sensitive) |
| `slur_severity` | `all` |
| `fallback_message` | `Please rephrase the query without unsafe content.` |

| Request `config` | Input → Output |
|---|---|
| `[{"type": "content_safety"}]` | `You are a worthless idiot…` → `Please rephrase the query without unsafe content.` |
| `[{"type": "content_safety", "on_fail": "rephrase", "fallback_message": "Please keep it respectful."}]` | → `Please keep it respectful.` |
| `[{"type": "content_safety"}]` | `What are the opening hours of the health centre?` → unchanged |

### `gender_assumption_bias`: neutralise gendered assumptions

| Param | Default | Values |
|---|---|---|
| `categories` | `["all"]` | `generic`, `healthcare`, `education`, `all` |

| Request `config` | Output |
|---|---|
| `[{"type": "gender_assumption_bias"}]` | `Every nurse should remember that they must be gentle, and every surgeon knows they must stay calm.` |

### `topic_relevance_llm`: keep conversations on topic

Uses an LLM to judge whether the text fits your allowed topic. Off-topic text is replaced with a fallback message.

| Param | Default | Meaning |
|---|---|---|
| `configuration` | none | Plain-language description of the allowed scope. |
| `topic_relevance_config_id` | none | A stored topic config, as an alternative to `configuration`. |
| `threshold` | `2` | Strictness of the relevance score. |
| `fallback_message` | none | Text returned when off-topic. |

```json
{
  "type": "topic_relevance_llm",
  "configuration": "Questions about maternal health, pregnancy, nutrition during pregnancy and newborn care.",
  "fallback_message": "I can only help with maternal and newborn health questions."
}
```

> Not verified in the local run: the guardrails service needs a working OpenAI key for this validator.

---

## Other validator types (stored config only)

The guardrails service has 7 more types. They aren't offered to tenants yet, so they are **not** accepted inline and don't appear in `GET /api/v1/guardrails`. They do run if you create a stored config and reference it by `validator_config_id`:

```bash
# 1. Create the config (settings go next to `type`; note `on_fail_action` here, not `on_fail`)
curl -X POST "$KAAPI_URL/api/v1/guardrails/validators/configs" \
  -H "X-API-KEY: $KAAPI_API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "Support ban list", "type": "ban_list", "stage": "input", "on_fail_action": "fix", "banned_words": ["scam", "fraud"]}'

# 2. Use it
curl -X POST "$KAAPI_URL/api/v1/guardrails" \
  -H "X-API-KEY: $KAAPI_API_KEY" -H "Content-Type: application/json" \
  -d '{"text": "This offer is a scam and pure fraud, avoid it.", "config": [{"validator_config_id": "<id from step 1>"}]}'
```

Config names must be unique within a project.

| Type | What it does | Required settings | Local test result |
|---|---|---|---|
| `ban_list` | Redacts words you list | `banned_words` **or** `ban_list_id` (from `POST /guardrails/ban_lists`) | ✅ Works. `This offer is a scam and pure fraud` → `This offer is a s and pure f`. Each banned word is cut to its first letter rather than removed. `exception` → `FAILED: Output contains banned words`. |
| `nsfw_text` | Flags sexual/explicit content (local English model) | none (`threshold` 0.8, `validation_method` `sentence`) | ✅ Works. Explicit text → empty string; clean text unchanged. `exception` → `FAILED: The following sentences in your response were found to be NSFW`. |
| `profanity_free` | Flags English profanity | none | ⚠️ Flagged **every** text in the local install, including `This is quite ridiculous, honestly.` Not usable until that's investigated. |
| `llamaguard_7b` | Safe/unsafe policy classifier (remote Guardrails Hub) | none (`policies` optional) | ❌ Not installed in the local guardrails service (`guardrails_grhub_llamaguard_7b` missing). Job `FAILED: An unexpected error occurred.` |
| `topic_relevance` | Older LLM topic check | `topic_relevance_config_id` (from `POST /guardrails/llm_prompt_configs`) | ⏸ Needs OpenAI. Runs, but the local key lacks access. |
| `llm_critic` | LLM scores text against your metrics | `metrics`, `max_score`, `llm_callable`: all three | ⏸ Needs OpenAI. Settings accepted once `metrics` used the right shape (below). |
| `answer_relevance_custom_llm` | Does an answer address the question? | `prompt_template` with `{query}` and `{answer}` (or `custom_prompt_id`) | ⏸ Needs OpenAI. Send `text` as a JSON string: `"{\"query\": \"...\", \"answer\": \"...\"}"`. |

Things to watch with these types:

- **Empty text on a "pass":** `nsfw_text` and `profanity_free` can't partially fix text. With `on_fail_action: fix`, a hit returns an **empty string** with `status: SUCCESS` and no warning. The LLM-backed types do the same when their LLM call fails. Treat an empty result as blocked, or use `on_fail_action: exception`.
- **`llm_critic` metrics shape:** each metric must be an object with a `threshold`, not a plain string. A plain string crashes the job:

  ```json
  "metrics": { "helpfulness": { "description": "Does the text give a useful, specific answer?", "threshold": 3 } }
  ```

- **LLM failures:** `llm_critic` and `llamaguard_7b` fail the job with a generic `An unexpected error occurred.`. Check the guardrails service logs for the cause.

---

## Getting the result

**Poll:** `GET /api/v1/guardrails/{job_id}` until `status` is `SUCCESS` or `FAILED`. Most jobs finish in under a second; `content_safety` and `topic_relevance_llm` can take a few seconds.

**Callback:** pass `callback_url` and Kaapi POSTs the result to it. The URL must be public `https://`. Localhost and private IPs are refused. The body is the same envelope as `guardrails_response` above, plus your `request_metadata` and the `warnings` list under `metadata`:

```json
{
  "success": true,
  "data": { "response": { "response_id": "…", "output": { "type": "text", "content": { "format": "text", "value": "<cleaned text>" } } }, "usage": { … }, "provider_raw_response": null },
  "error": null,
  "metadata": { "conversation_id": "conv-42", "warnings": [] }
}
```

When the text is blocked, the callback has `success: false`, `data: null` and the reason in `error`. Callbacks are signed the same way as `/llm/call` webhooks when a webhook secret is configured.

### Job outcomes

| `status` | When | Where to look |
|---|---|---|
| `SUCCESS` | The text passed, or was cleaned. | `guardrails_response…value` holds the text to use. |
| `SUCCESS` + `warnings` | Nothing ran, or the guardrails service was unreachable. The **original text** is returned. | `warnings` |
| `FAILED` | A validator with `on_fail: "exception"` tripped, or the guardrails service rejected the config. | `error_message` |

### Warnings you may see

| Warning | Cause |
|---|---|
| `Request contained N duplicate validator entries; duplicates were ignored…` | The same stored ID or identical inline spec was listed twice. Each runs once. |
| `Request contained no usable validators after deduplication; original text was returned unchanged.` | Every entry had `is_enabled: false`. |
| `Validators were submitted but none resolved against the guardrails service; original text was returned unchanged.` | The `validator_config_id`s don't exist, or don't belong to your project. |
| `Guardrails service was unavailable; original text was returned unchanged.` | The guardrails service couldn't be reached, so Kaapi fails open. |

> Treat a warning that says "original text was returned unchanged" as **not checked**, not as safe.

---

## Errors on submit

These fail straight away, before a job is created.

| HTTP | Example request | Response `errors` |
|---|---|---|
| 422 | `config: [{"type": "not_a_validator"}]` | `config.0.inline.type: Input should be 'content_safety', 'gender_assumption_bias', 'pii_remover', 'topic_relevance_llm' or 'uli_slur_match'` |
| 422 | `config: []` | `config: List should have at least 1 item after validation, not 0` |
| 422 | `text: ""` | `text: String should have at least 1 character` |
| 422 | `config: [{"validator_config_id": "not-a-uuid"}]` | `config.0.ref.validator_config_id: Input should be a valid UUID…` |
| 401 | missing `X-API-KEY` | `Invalid Authorization format` |

Inline **settings** (anything other than `type`) are checked by the guardrails service when the job runs, not on submit. A bad value such as `"on_fail": "explode"` is accepted (200), and the job then ends `FAILED` with `Guardrails service rejected the request (HTTP 422)`.

---

## Good to know

- **Order:** stored configs run before inline specs, whatever order you list them in. Within each group, your order is kept.
- **Input stage:** every check runs as an input-stage validation. Pairing a prompt with an LLM reply isn't supported on this endpoint.
- **Storage:** the submitted text is stored on the job for traceability, because this endpoint exists to inspect unsafe content.
- **Old payloads:** older `{validator_config_id, type, tag}` entries still work. `type` and `tag` are ignored.
