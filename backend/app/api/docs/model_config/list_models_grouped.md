## Endpoint

**GET** `/api/v1/models/grouped`

Retrieve active models grouped by provider.

Supports pagination of model rows before grouping:
- `skip` (default `0`)
- `limit` (default `100`, max `100`)

Returns a dictionary where each key is a provider present in the paginated slice, and each value is a list of active model configurations for that provider.
Includes `metadata.has_more` when additional model rows exist.

### Example Response

```json
{
  "success": true,
  "metadata": {
    "has_more": true
  },
  "data": {
    "openai": [
      {
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
          }
        },
        "input_modalities": ["TEXT", "IMAGE"],
        "output_modalities": ["TEXT"],
        "pricing": {
          "response": {
            "input_token_cost": 2.5,
            "output_token_cost": 10
          }
        },
        "is_active": true,
        "inserted_at": "2026-03-12T00:00:00",
        "updated_at": "2026-03-12T00:00:00"
      }
    ]
  }
}
```
