List every validator type the guardrails service supports, together with the JSON Schema of the tuning fields each one accepts.

Use this to discover what to put in the body of `POST /guardrails/validators/configs`: the `config` schema of an entry tells you exactly which extra keys that validator type understands.

### Response shape

This is the one guardrails route that is **not** wrapped in the standard `APIResponse` envelope. The body is a bare object with one entry per supported validator type (abridged — `config` is a full JSON Schema generated from the service's own model, so it is always current):

```json
{
  "validators": [
    {"type": "pii_remover", "config": { "...JSON Schema..." }},
    {"type": "uli_slur_match", "config": { "...JSON Schema..." }}
  ]
}
```

### Validator types

`uli_slur_match`, `pii_remover`, `gender_assumption_bias`, `ban_list`, `topic_relevance`, `topic_relevance_llm`, `llm_critic`, `llamaguard_7b`, `profanity_free`, `nsfw_text`, `answer_relevance_custom_llm`.

Every validator additionally accepts `on_fail` (`exception` | `fix` | `rephrase`, default `fix`) and `stage` (`input` | `output`).

### Errors

- `502` — the guardrails service is unreachable or returned a non-JSON body.
