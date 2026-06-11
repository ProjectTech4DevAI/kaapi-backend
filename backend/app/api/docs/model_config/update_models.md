Update multiple model configurations in one request.

Each item must include `provider` + `model_name` to identify the target. Other fields are optional and follow the same rules as the single PATCH endpoint (replace semantics, no deep merge).

Atomic — if any target is missing, no updates are applied.

### Example

```json
[
  { "provider": "google", "model_name": "gemini-2.5-flash", "completion_type": ["text", "stt"] },
  { "provider": "sarvamai", "model_name": "saaras:v3", "is_active": false }
]
```

### Errors

- `404` — One or more targets not found.
- `422` — Validation error.
