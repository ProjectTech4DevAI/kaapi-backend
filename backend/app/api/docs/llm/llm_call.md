Make an LLM API call using either a stored configuration or an ad-hoc configuration.

This endpoint initiates an asynchronous LLM call job. The request is queued
for processing, and results are delivered via the callback URL when complete.

### Key Parameters

**`query`** (required) - Query parameters for this LLM call:
- `input` (required, string, min 1 char): User question/prompt/query
- `conversation` (optional, object): Conversation configuration
  - `id` (optional, string): Existing conversation ID to continue
  - `auto_create` (optional, boolean, default false): Create new conversation if no ID provided
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
      - `provider` (required, string): Provider type - either `"openai"` (Kaapi abstraction) or `"openai-native"` (pass-through)
      - `params` (required, object): Parameters structure depends on provider type (see schema for detailed structure)
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
