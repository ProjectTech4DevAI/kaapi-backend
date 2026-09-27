Create a new version for an existing configuration.

To create a new version, provide the `config_id` in the URL path and the new
configuration parameters in the request body. The system will automatically
create a new version under the same configuration with an incremented version number.
Version numbers are automatically incremented sequentially (1, 2, 3, etc.)
and cannot be manually set or skipped.

The `config_blob` shape follows the parent config: the `completion` shape for a
`default` config, the `assessment` shape for an `ASSESSMENT` config. How the body is
applied differs between the two, so read the matching section below.

## default configs — partial update

Send only the fields you want to change. They are merged onto the latest version, so
anything you omit is carried forward.

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

## ASSESSMENT configs — full blob

Send the **whole** `config_blob` every time. It replaces the previous version rather than
merging onto it, so a column dropped from `input_schema`, a removed `json_output_schema`
field, or an omitted `pre_filters` block is genuinely gone in the new version. A partial
body is rejected with `422`, because `input_schema` and `assessment` are mandatory.

```json
{
  "config_blob": {
    "input_schema": {
      "rubric": { "type": "text" },
      "answer": { "type": "text", "strict": true }
    },
    "pre_filters": {
      "topic_relevance": {
        "provider": "openai",
        "params": { "model": "gpt-4o", "instructions": "Is this a Class 7 answer sheet?" },
        "stop_on_fail": true
      }
    },
    "assessment": {
      "provider": "openai",
      "type": "text",
      "params": {
        "model": "gpt-4o",
        "instructions": "You are an AI Assessment Evaluator ...",
        "submission": "Grade this answer against the rubric: {rubric}\n\nAnswer: {answer}"
      }
    }
  },
  "commit_message": "Drop the unused columns"
}
```

## Important
- Every field inside `config_blob` can change between versions, including provider and model.
- `tag` belongs to the parent configuration and is never part of a version body.
- `type` is inherited from the existing configuration and cannot be changed.
- A run pins the `config_id` and `version` it was submitted with, so a new version never
  alters a run that is already in flight or finished.
