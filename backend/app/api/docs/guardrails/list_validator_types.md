Lists every validator type the guardrails service can run, with the JSON Schema of the tuning keys each one accepts.

Start here when you are setting guardrails up. This endpoint is the authority on what **your** deployment supports: if a type is listed you can configure it, and if it is missing that deployment does not have it yet — whatever other documentation says.

<br>

## Setting one up

Two steps.

**Step 1 — save a reusable config** with `POST /guardrails/validators/configs`. The body is the five shared fields, plus the tuning keys of the type you picked:

```json
{
  "name": "PII Redaction Input",
  "type": "pii_remover",
  "stage": "input",
  "on_fail_action": "fix",
  "is_enabled": true,
  "entity_types": ["EMAIL_ADDRESS", "PHONE_NUMBER", "IN_AADHAAR"]
}
```

| Shared field | Required | What it means |
|---|---|---|
| `name` | yes | Your label for this setup, 5–225 characters, unique in the project. The same type can be saved many times under different names. |
| `type` | yes | One of the types listed below. |
| `stage` | yes | `input` checks the user's message, `output` checks the model's reply. Advisory only — a run checks whatever text you hand it. |
| `on_fail_action` | no, defaults to `fix` | `fix` repairs the text where it can, `rephrase` returns a stand-in reply instead, `exception` fails the request outright. |
| `is_enabled` | no, defaults to `true` | Whether this config is allowed to run. |

The response carries an `id`. That id is what you use everywhere else.

**Step 2 — run it** with `POST /guardrails`:

```json
{
  "text": "Call me on 9876543210",
  "config": [{"validator_config_id": "<id from step 1>"}]
}
```

Guardrails compose: list several ids and they run in that order over the same message.

<br>

## One naming trap

The schemas below describe each validator *as it runs*, so they spell the failure action `on_fail`. When you *save* a config in step 1, the same setting is called `on_fail_action`. Same three values, two names, depending on which endpoint you are talking to.

<br>

## What this endpoint returns

The one guardrails route that is **not** wrapped in the usual `{"success": ..., "data": ...}` envelope. The body is a bare object:

```json
{
  "validators": [
    {
      "type": "pii_remover",
      "config": {"type": "object", "properties": {"entity_types": {"...": "..."}}}
    }
  ]
}
```

Each `config` is generated from the service's own model, so it is always current — and a little noisy, carrying the `title`, `$defs` and `anyOf` wrappers that JSON Schema needs for optional fields. The reference below is that same information, trimmed and explained.

<br>

## Errors

| Code | Meaning |
|---|---|
| `401` / `403` | Your API key is missing, invalid, or not allowed to use guardrails. |
| `502` | The guardrails service is unreachable, or returned something that was not JSON. |
