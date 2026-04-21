## Endpoint

**GET** `/api/v1/models/{provider}/{model_name}`

Retrieve a specific model configuration by provider and model name.

Returns model details including supported config parameters, input/output modalities, pricing, and active status.

### Path Parameters

- **`provider`** (required) — Provider name (e.g. `openai`, `google`)
- **`model_name`** (required) — Model name (e.g. `gpt-4o`, `gpt-4o-mini`)

### Example Response

```json
{
  "success": true,
  "data": {
    "id": 2,
    "provider": "openai",
    "model_name": "gpt-4o",
    "config": {
      "temperature": {
        "type": "float",
        "default": 1.0,
        "min": 0.0,
        "max": 2.0,
        "description": "Controls randomness. Lower = more deterministic."
      },
      "top_p": {
        "type": "float",
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "description": "Nucleus sampling. Use either this or temperature, not both."
      },
      "max_output_tokens": {
        "type": "int",
        "default": 2048,
        "min": 1,
        "max": 32768,
        "description": "Max tokens in the response."
      }
    },
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"],
    "pricing": {
      "response": {
        "input_token_cost": 2.5,
        "output_token_cost": 10
      },
      "batch": {
        "input_token_cost": 1.25,
        "output_token_cost": 5
      }
    },
    "is_active": true,
    "inserted_at": "2026-03-12T00:00:00",
    "updated_at": "2026-03-12T00:00:00"
  }
}
```

### Error Response

- `404 Not Found` — Model not found for the given `provider` and `model_name`.
