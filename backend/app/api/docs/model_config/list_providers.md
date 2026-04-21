## Endpoint

**GET** `/api/v1/models/providers`

Retrieve the list of providers that currently have active models.

Returns provider names sorted in ascending order.

### Example Response

```json
{
  "success": true,
  "data": ["google", "openai"]
}
```
