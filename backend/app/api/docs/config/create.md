Create a new LLM configuration with an initial version.

Configurations allow you to store and manage reusable LLM parameters
(such as temperature, max_tokens, model selection, etc.) with version control.

**Key Features:**
* Automatically creates an initial version (v1) with the provided configuration
* Enforces unique configuration names per project
* Stores provider-specific parameters as flexible JSON (config_blob)
* Supports optional commit messages for tracking changes
* Provider-agnostic storage - params are passed through to the provider as-is
* Supports three types of config blob types; "text", "stt" and "tts".


**Example for the config blob: OpenAI Responses API with File Search -**

```json
"config_blob": {
    "completion": {
      "provider": "openai",
      "type":"text",
      "params": {
        "model": "gpt-4o-mini",
        "instructions": "You are a helpful assistant for farming communities...",
        "temperature": 1,
        "tools": [
          {
            "type": "file_search",
            "vector_store_ids": ["vs_692d71f3f5708191b1c46525f3c1e196"],
            "max_num_results": 20
          }]}}}
```

The configuration name must be unique within your project. Once created,
you can create additional versions to track parameter changes while
maintaining the configuration history. Type 'text' is applicable only for OpenAI models.
Type "stt" and "tts" are only applicable for Gemini models.
