Create a new LLM configuration with an initial version.

Configurations allow you to store and manage reusable LLM parameters
(such as temperature, max_tokens, model selection, etc.) with version control.

**Key Features:**
* Automatically creates an initial version (v1) with the provided configuration
* Enforces unique configuration names per project
* Stores provider-specific parameters as flexible JSON (config_blob)
* Supports optional commit messages for tracking changes
* Kaapi providers (openai, google) — params are validated and mapped internally
* Native providers (openai-native, google-native) — params are passed through as-is
* Supports three completion types: `"text"`, `"stt"` and `"tts"`
* Supports both `input_guardrails` and `output_guardrails`

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

The configuration name must be unique within your project. Once created,
you can create additional versions to track parameter changes while
maintaining the configuration history. Type `"text"` is applicable for OpenAI models.
Type `"text"`, `"stt"` and `"tts"` are applicable for Google Gemini models.
