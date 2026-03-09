Make an LLM API call using either a stored configuration or an ad-hoc configuration.

This endpoint initiates an asynchronous LLM call job. The request is queued
for processing, and results are delivered via the callback URL when complete.

### Key Parameters

**`query`** (required) - Query parameters for this LLM call:

- **`input`** (required) — User input in one of three forms:

  1. **Plain string** — automatically treated as text input
     ```json
     "input": "Hello"
     ```

  2. **Structured input object** — with `type` and `content`
     ```json
     "input": {"type": "text", "content": {"format": "text", "value": "Hello"}}
     ```

  +  3. **List of structured inputs** — for multimodal use cases (supports `image`, `pdf`, and `text`;  `audio` is only supported as a single structured input)
     ```json
     "input": [
       {"type": "text", "content": {"format": "text", "value": "Describe this image"}},
       {"type": "image", "content": {"format": "base64", "value": "..."}}
     ]
     ```

  **Supported input types:** `text`, `audio`, `image`, `pdf`

  **Content format by type:**
  - `text` — format: `"text"`
  - `audio` — format: `"base64"`, optional `mime_type` (e.g. `audio/wav`)
  - `image` — format: `"base64"` or `"url"`, optional `mime_type` (default: `image/png`)
  - `pdf` — format: `"base64"` or `"url"`, optional `mime_type` (default: `application/pdf`)

  For `image` and `pdf`, `content` can be a single object or a list of objects.

- **`conversation`** (optional) — Conversation configuration
  - `id` (string): Existing conversation ID to continue
  - `auto_create` (boolean, default false): Create a new conversation if no ID provided
  - **Note**: Cannot specify both `id` and `auto_create=true`

**`config`** (required) - Configuration for the LLM call (just choose one mode):

- **Mode 1: Stored Configuration**
  - `id` (UUID): Configuration ID
  - `version` (integer >= 1): Version number
  - **Both required together**
  - **Note**: When using stored configuration, do not include the `blob` field in the request body

- **Mode 2: Ad-hoc Configuration**
  - `blob` (object): Complete configuration object
    - `completion` (required, object): Completion configuration
      - `provider` (required, string): Provider type — `"openai"` or `"google"` (Kaapi abstraction), or `"openai-native"` or `"google-native"` (pass-through)
      - `type` (required, string): Completion type — `"text"`, `"stt"`, `"tts"`
      - `params` (required, object): Parameters structure depends on provider and type (see schema for detailed structure)
    - `input_guardrails` (optional, list)
    - `output_guardrails` (optional, list)
  - **Note**
    - When using ad-hoc configuration, do not include `id` and `version` fields
    - When using the Kaapi abstraction, parameters that are not supported by the selected provider or model are automatically suppressed. If any parameters are ignored, a list of warnings is included in the metadata.warnings. For example, the GPT-5 model does not support the temperature parameter, so Kaapi will neither throw an error nor pass this parameter to the model; instead, it will return a warning in the metadata.warnings response.
  - **Recommendation**: Use stored configs (Mode 1) for production; use ad-hoc configs only for testing/validation
  - **Schema**: Check the API schema or examples below for the complete parameter structure for each provider type

**`callback_url`** (optional, HTTPS URL):
- Webhook endpoint to receive the response
- Must be a valid HTTPS URL
- If not provided, response is only accessible through job status

**`include_provider_raw_response`** (optional, boolean, default false):
- When true, includes the unmodified raw response from the LLM provider

**`request_metadata`** (optional, object):
- Custom JSON metadata
- Passed through unchanged in the response

### Note
- `warnings` list is automatically added in response metadata when using Kaapi configs if any parameters are suppressed or adjusted (e.g., temperature on reasoning models)

---
