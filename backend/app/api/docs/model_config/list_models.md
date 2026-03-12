## Endpoint

**GET** `/api/v1/models`

Retrieve a list of all active model configurations.

Returns model details including provider, model name, supported config parameters, input/output modalities, and default assignment.

Optionally filter by provider (e.g. openai, google).

### Query Parameters

- **`provider`** (optional) — Filter by provider name (e.g. `openai`, `google`)
- **`skip`** (optional, default 0) — Number of records to skip for pagination
- **`limit`** (optional, default 100) — Maximum number of records to return

### Example Response

```json
{
  "success": true,
  "data": {
    "data": [
      {
        "id": 1,
        "provider": "openai",
        "model_name": "gpt-4o-mini",
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
        "default_for": null,
        "is_active": true,
        "inserted_at": "2026-03-12T00:00:00",
        "updated_at": "2026-03-12T00:00:00"
      }
    ],
    "count": 1
  }
}
```
