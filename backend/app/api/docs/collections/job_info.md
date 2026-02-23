Retrieve information about a collection job by the collection job ID. This endpoint provides detailed status and metadata for a specific collection job in Kaapi. It is especially useful for:

* Fetching the collection job object, including the collection job ID, the current status, and the associated collection details.

* If the job has finished, has been successful and it was a job of creation of collection then this endpoint will fetch the associated collection details.

* If the delete-collection job succeeds, the status is set to "successful" and the `collection` key contains the ID of the collection that has been deleted.

**Response Fields for Successful Creation Jobs:**

**Note:** While the API schema shows both `llm_service_id`/`llm_service_name` AND `knowledge_base_id`/`knowledge_base_provider`, the actual collection object in the response will only include the fields relevant to what was created:

- **If an Assistant was created** (with model + instructions): The response will only include `llm_service_id` and `llm_service_name`
- **If only a Vector Store was created** (without model/instructions): The response will only include `knowledge_base_id` and `knowledge_base_provider`
