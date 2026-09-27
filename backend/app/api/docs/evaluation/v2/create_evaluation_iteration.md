Kick off a self-driving eval -> improve-prompt -> eval loop.

Each round runs a v2 judged evaluation against `dataset_id` with the current
config version, then — unless the round's stop score already clears the
ceiling or `max_rounds` is reached — hands the config to the prompt-improvement
step to produce the next version and repeats. The loop runs entirely in the
background; this endpoint only validates the request, creates the tracking
row, and dispatches round 1.

The stop score is the mean of **Adherence to Ground Truth** and **Adherence to
Prompt** from the round's judge summary (Adherence to Knowledge Base is
reported but never gates). The loop stops on whichever comes first:

- the stop score reaches the configured ceiling (`ceiling_reached`)
- `max_rounds` is exhausted (`max_rounds_reached`)
- a round fails to produce a usable score (`round_failed`)

`dataset_id` and `config_id`/`config_version` are validated the same way as
`POST /api/v2/evaluations` (dataset must exist and be accessible, config must
resolve to a text OpenAI config within the fast-eval row limit) — the loop
always runs judged.

## Rounds cap (optional)

`max_rounds` bounds how many eval/improve rounds the loop may run. Omit it to
use `EVAL_ITERATION_MAX_ROUNDS_DEFAULT`; a value above
`EVAL_ITERATION_MAX_ROUNDS_HARD_CAP` is rejected with a 422.

## Completion webhook (required)

`callback_url` (HTTPS only) receives the round-by-round report once the loop
stops, for any reason. Same delivery semantics as the v2 evaluation callback —
best-effort, at-least-once, signed with `X-Webhook-Signature` /
`X-Webhook-Timestamp` when a `webhook_secret` credential is configured for the
project. The URL is rejected with `422 invalid_callback_url` if it is not a
public HTTPS endpoint (SSRF guard).

## Example

```json
{
  "dataset_id": 123,
  "experiment_name": "iterate-smoke-1",
  "config_id": "f54f0d67-4817-4103-9fdf-b74b3d46733e",
  "config_version": 1,
  "max_rounds": 5,
  "callback_url": "https://example.com/webhooks/eval-iteration-complete"
}
```

## Error responses

| Status | When |
| --- | --- |
| 404 | `dataset_id` does not exist or is not accessible to this organization/project |
| 400 | The config fails to resolve for the given `config_id`/`config_version` |
| 422 | `invalid_callback_url` — not a public HTTPS endpoint (SSRF guard) |
| 422 | The config is not a text OpenAI config, or the dataset exceeds the fast-eval row limit |
| 500 | `evaluation_iteration_enqueue_failed` — the tracking row was created but round 1 could not be queued |
