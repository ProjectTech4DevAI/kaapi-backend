Retrieve the status and results of an LLM call job by job ID.

This endpoint allows you to poll for the status and results of an asynchronous LLM call job that was previously initiated via the POST `/llm/call` endpoint.

### Path Parameters

**`job_id`** (required, UUID) - The unique identifier of the job returned when the LLM call was created.

### Response

The endpoint returns an `LLMJobPublic` object containing:

- **`job_id`** (UUID) - The unique identifier of the job
- **`status`** (string) - Current status of the job. Possible values:
  - `PENDING` - Job has been created and is waiting to be processed
  - `PROCESSING` - Job is currently being processed
  - `SUCCESS` - Job completed successfully
  - `FAILED` - Job failed during processing
- **`llm_response`** (object | null) - The complete LLM response when status is `SUCCESS`, containing:
  - `response` - Normalized LLM response with provider_response_id, conversation_id, provider, model, and output
  - `usage` - Token usage information (input_tokens, output_tokens, total_tokens)
- **`error_message`** (string | null) - Error details if the job failed, otherwise null
- **`job_inserted_at`** (datetime) - Timestamp when the job was created
- **`job_updated_at`** (datetime) - Timestamp when the job was last updated

### Usage

1. Create an LLM call using POST `/llm/call` to receive a `job_id`
2. Use this endpoint to poll for the job status
3. When the status is `SUCCESS`, the `llm_response` field will contain the complete LLM response
4. When the status is `FAILED`, check the `error_message` field for details

### Polling Strategy

- Poll this endpoint periodically until `status` is either `SUCCESS` or `FAILED`
- Use exponential backoff (e.g., 1s, 2s, 4s, 8s) to reduce server load
- Stop polling when status is terminal (`SUCCESS` or `FAILED`)

### Notes

- This endpoint returns both the job status AND the actual LLM response when complete
- LLM responses are also delivered asynchronously via the callback URL (if provided)
- Jobs can be queried at any time after creation
- The endpoint returns a 404 error if the job_id does not exist
