Retrieve the status and results of an LLM call job by job ID.

This endpoint allows you to poll for the status and results of an asynchronous LLM call job that was previously initiated via the POST `/llm/call` endpoint.


### Notes

- This endpoint returns both the job status AND the actual LLM response when complete
- LLM responses are also delivered asynchronously via the callback URL (if provided)
- Jobs can be queried at any time after creation
