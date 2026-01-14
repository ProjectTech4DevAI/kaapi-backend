Retrieve information about a collection job by the collection job ID. This endpoint provides detailed status and metadata for a specific collection job in Kaapi. It is especially useful for:

* Fetching the collection job object, including the collection job ID, the current status, and the associated collection details.

* If the job has finished, has been successful and it was a job of creation of collection then this endpoint will fetch the associated collection details from the collection table, including:
  - `llm_service_id` and `llm_service_name`.
  - Collection metadata such as ID, project, organization, and timestamps.

* If the delete-collection job succeeds, the status is set to “successful” and the `collection` key contains the ID of the collection that has been deleted.
