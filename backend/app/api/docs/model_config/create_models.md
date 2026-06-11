Create one or more model configurations.

Accepts a single object or an array. Response is always an array.

**Required:** `provider`, `model_name`, `completion_type`, `config`
**Optional:** `input_modalities`, `output_modalities`, `pricing`, `is_active`

`(provider, model_name)` must be unique.

### Example (single)

```json
{
  "provider": "google",
  "model_name": "gemini-2.5-flash",
  "completion_type": ["text", "stt"],
  "config": { "temperature": { "type": "float", "default": 1.0, "min": 0.0, "max": 2.0 } },
  "input_modalities": ["TEXT", "AUDIO"],
  "output_modalities": ["TEXT"],
  "pricing": {
    "response": { "input_token_cost": 0.3, "output_token_cost": 2.5 }
  },
  "is_active": true
}
```

### Example (multiple)

```json
[
  {
    "provider": "sarvamai",
    "model_name": "saaras:v3",
    "completion_type": ["stt"],
    "config": {},
    "input_modalities": ["AUDIO"],
    "output_modalities": ["TEXT"]
  },
  {
    "provider": "elevenlabs",
    "model_name": "scribe_v2",
    "completion_type": ["stt"],
    "config": {},
    "input_modalities": ["AUDIO"],
    "output_modalities": ["TEXT"]
  }
]
```

### Errors

- `422` — Validation error.
- DB integrity error on duplicate `(provider, model_name)`.
