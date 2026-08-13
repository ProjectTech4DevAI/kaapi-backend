Create a new version for an existing configuration.

To create a new version, provide the `config_id` in the URL path and the new
configuration parameters in the request body. The system will automatically
create a new version under the same configuration with an incremented version number.
Version numbers are automatically incremented sequentially (1, 2, 3, etc.)
and cannot be manually set or skipped.

## Examples

Send only the fields you want to change. The `config_blob` shape follows the parent
config: the `completion` shape for a `default` config, the `assessment` shape for an
`ASSESSMENT` config.

**When the parent config is `default` (completion shape):**
```json
{
  "config_blob": {
    "completion": {
      "params": { "temperature": 0.5 }
    }
  },
  "commit_message": "Lower temperature"
}
```

**When the parent config is `ASSESSMENT` (assessment shape):**
```json
{
  "config_blob": {
    "pre_filters": {
      "topic_relevance": {
        "params": { "model": "gpt-4o", "instructions": "Is this a Class 7 answer sheet?" }
      }
    },
    "assessment": {
      "params": { "model": "gpt-4o" }
    }
  },
  "commit_message": "Switch grading model"
}
```

## Important
- This endpoint accepts partial updates using dict[str, Any] for config_blob.
- Only the fields that need to be updated should be provided.
- The `type` field is inherited from the existing configuration and cannot be changed. Provider and model can change between versions.
