Apply guardrails to a piece of text and deliver the sanitised result via a webhook callback.

This endpoint exists for callers who manage their own LLM workflow but want to
reuse Kaapi's guardrails service. It is symmetric for input and output
guardrails: send the text that needs sanitisation in `text` along with one or
more `validator_config_id`s, and receive the sanitised text on your
`callback_url`.

### Flow

1. Caller POSTs `{text, config, callback_url}` to `/api/v1/guardrails`.
2. Kaapi creates a job (`job_type=LLM_GUARDRAILS`), returns `job_id` with HTTP 200
   immediately.
3. A Celery worker resolves the validators, calls the guardrails service, and
   POSTs the sanitised text (or a hard-block error) to `callback_url`.
4. The full upstream guardrails response and the original request body are
   persisted on `job.meta` for traceability and can be inspected via
   `GET /api/v1/guardrails/{job_id}`.

### Webhook payload

The webhook receives a standard `APIResponse` envelope:

```json
{
  "success": true,
  "data": {
    "response": {
      "response_id": "<guardrails-response-id-or-null>",
      "output": {
        "type": "text",
        "content": { "format": "text", "value": "<sanitised text>" }
      }
    },
    "usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0,
      "reasoning_tokens": 0
    },
    "provider_raw_response": null
  },
  "error": null,
  "metadata": { "<your request_metadata>": "...", "warnings": [] }
}
```

If the guardrails service hard-blocks the text, `success` is `false`, `error`
carries the upstream message, and `data` is `null`. If the guardrails service
is unreachable the job still succeeds but the webhook carries the original
text unchanged and `metadata.warnings` carries a human-readable note (e.g.
`"Guardrails service was unavailable; original text was returned unchanged."`).
Other warnings may surface for duplicate validator IDs, an empty validator
list, or a missing sanitised text in the upstream response.

### Notes

- `config[].type` and `config[].tag` are caller-side
  bookkeeping. They are not interpreted by the server but are useful for your
  own correlation (echoed back via `request_metadata` if you include them
  there).
- For output-guardrail flows that need the original prompt paired with the
  LLM output, this endpoint v1 sends only `text`; pairing is not exposed.
- The same webhook signing scheme as `/llm/call` is used when a webhook secret
  is configured.
