Partially update a model configuration. Only fields sent are updated; omitted fields stay unchanged.

**Updatable fields:** `completion_type`, `config`, `input_modalities`, `output_modalities`, `pricing`, `is_active`

Arrays and objects are **replaced** (no deep merge). `provider` and `model_name` cannot be changed here.

### Example

```json
{
  "completion_type": ["text", "stt"],
  "pricing": {
    "response": { "input_token_cost": 0.5, "output_token_cost": 3.0 }
  }
}
```

### Errors

- `404` — Model not found.
- `422` — Validation error.
