Retrieve detailed information about a specific collection by its collection id. This endpoint returns the collection object including its project, organization,
timestamps, and associated LLM service details (`llm_service_id` and `llm_service_name`).

Additionally, if the `include_docs` flag in the request body is true then you will get a list of document IDs associated with a given collection as well. Note that, documents returned are not only stored by Kaapi, but also by Vector store provider.
