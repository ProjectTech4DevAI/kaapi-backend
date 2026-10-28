Update a stored LLM prompt config. Fields you omit are left unchanged.

### Request

```json
{"llm_prompt": "Pregnancy care: Updated scope definition"}
```

`name`, `description`, `prompt_schema_version`, `llm_prompt` and `is_active` can be patched.

### Notes

- `validator_name` is immutable — it cannot be patched. Create a new config instead.
- `is_active` is settable only here, not on create.
- Editing `llm_prompt` on an `answer_relevance_custom_llm` config still requires both `{query}` and `{answer}` placeholders.

### Errors

- `400` — the update collides with an existing config.
- `404` — no such config, or it belongs to another project.
- `422` — the body failed validation, the placeholder rule was broken, or `prompt_config_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
