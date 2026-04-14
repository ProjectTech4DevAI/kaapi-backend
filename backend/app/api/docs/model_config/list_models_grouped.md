## Endpoint

**GET** `/api/v1/models/grouped`

Retrieve all active models grouped by provider.

Returns a dictionary where each key is a provider and each value is a list of active model configurations for that provider.

### Example Response

```json
{
  "success": true,
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
    ],
    "google": []
  }
}
```
