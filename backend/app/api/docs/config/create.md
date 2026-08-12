Create a new LLM configuration with an initial version.

Configurations allow you to store and manage reusable LLM parameters
(such as temperature, max_tokens, model selection, etc.) with version control.

**The `tag` field** classifies the config and selects which `config_blob` shape is valid.
It is a body field, defaults to `"default"`, and is fixed once set (new versions inherit it
and it cannot be changed later). Available values:

* `"default"` — general LLM configs, using the `completion` shape. Consumed by LLM call /
  chain / response, evaluations, and prompt improvement. Omit `tag` to get this.
* `"ASSESSMENT"` — grading / assessment configs, using the `assessment` shape. Consumed
  only by the assessment submit pipeline.

**Key Features:**
* Automatically creates an initial version (v1) with the provided configuration
* Enforces unique configuration names per project
* Stores provider-specific parameters as flexible JSON (config_blob)
* Supports optional commit messages for tracking changes
* Kaapi providers (`openai`, `google`, `sarvamai`) — params are validated and mapped internally
* Native providers (`openai-native`, `google-native`, `sarvamai-native`) — params are passed through as-is
* Supports three completion types: `"text"`, `"stt"` and `"tts"`
* Supports both `input_guardrails` and `output_guardrails`

## `tag: "default"` — config blob examples (completion shape)

**Example for the config blob: OpenAI Responses API with File Search -**

```json
"config_blob": {
  "completion": {
    "provider": "openai",
    "type": "text",
    "params": {
      "model": "gpt-4o-mini",
      "instructions": "You are a helpful assistant for farming communities...",
      "temperature": 1,
      "knowledge_base_ids": [
        "vs_692d71f3f5708191b1c46525f3c1e196"
      ]
    }
  }
}
```

**Example for the config blob: Google gemini STT -**
```json
"config_blob":{
  "completion": {
    "provider": "google",
    "type": "stt",
    "params": {
      "model": "gemini-3-flash-preview",
      "instructions": "You are a helpful assistant ...",
      "input_language": "english",
      "output_language": "hindi",
      "temperature": 1
    }
  }
}
```

**Example for the config blob: Google gemini TTS -**
```json
"config_blob":{
  "completion": {
    "provider": "google",
    "type": "tts",
    "params": {
      "model": "gemini-2.5-pro-preview-tts",
      "voice": "Kore",
      "language": "hindi",
      "response_format": "mp3"
    }
  }
}
```

## `tag: "ASSESSMENT"` — assessment config

When `tag` is `"ASSESSMENT"`, `config_blob` uses the assessment shape instead of the
`completion` shape above. Its full schema is validated on the server but is intentionally
not rendered in the OpenAPI spec (so the default `config_blob` schema stays stable) — the
shape is documented here:

* `assessment` (required) — the grading call. `provider` is `openai` | `google` |
  `anthropic`, `type` is `"text"`. `params` carries the `model`, the `instructions`
  (system prompt), an optional `json_output_schema` (structured-output JSON schema), and a
  **mandatory, non-empty** `input_schema` mapping each column name to `{ type, format }`
  (`type` is **required**: `text` | `image` | `pdf`; `format`: `url` for attachment
  columns). Every declared column must be present in every submission row (see the submit
  docs for per-row validation).
* `pre_filters` (optional) — `topic_relevance` and/or `duplicate_detection`. Each runs its
  own llm call, so it carries `provider` (default `openai`) + its own `params`
  (a `TextLLMParams` object: `model`, `temperature`, …). Its criteria live in
  `params.instructions` (a **mandatory** field, exactly like the assessment call);
  `params.model` defaults to `gpt-5.6-luna` when omitted. `duplicate_detection` also takes
  an optional `knowledge_base_id`. Each pre-filter carries `stop_on_fail` (`true` = a
  failing verdict stops the chain and skips the assessment for that row; `false` = the
  verdict is just recorded).

```json
"config_blob": {
  "pre_filters": {
    "topic_relevance": {
      "provider": "openai",
      "params": {
        "model": "gpt-4o",
        "temperature": 0.1,
        "instructions": "Is this a Class 7 answer sheet?"
      },
      "stop_on_fail": true
    }
  },
  "assessment": {
    "provider": "openai",
    "type": "text",
    "params": {
      "model": "gpt-4o",
      "instructions": "You are an AI Assessment Evaluator ...",
      "input_schema": {
        "gcs_url": { "type": "image", "format": "url" },
        "rubric":  { "type": "text" }
      },
      "json_output_schema": {
        "type": "object",
        "properties": { "grade": { "type": "string" }, "feedback": { "type": "string" } },
        "required": ["grade", "feedback"]
      }
    }
  }
}
```

The configuration name must be unique within your project. Once created,
you can create additional versions to track parameter changes while
maintaining the configuration history.

**Provider–type support:**
* `openai` / `openai-native` — `"text"`
* `google` / `google-native` — `"text"`, `"stt"`, `"tts"`
* `sarvamai` / `sarvamai-native` — `"stt"`, `"tts"`
