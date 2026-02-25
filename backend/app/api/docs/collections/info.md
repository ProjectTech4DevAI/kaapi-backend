Retrieve detailed information about a specific collection by its collection id. This endpoint returns the collection object including its project, organization, timestamps, and service-specific details.

**Response Fields:**

**Note:** While the API schema shows both `llm_service_id`/`llm_service_name` AND `knowledge_base_id`/`knowledge_base_provider`, the actual response will only include the fields relevant to what was created:

- **If an Assistant was created** (with model + instructions): The response will only include `llm_service_id` and `llm_service_name`
- **If only a Vector Store was created** (without model/instructions): The response will only include `knowledge_base_id` and `knowledge_base_provider`

**Including Documents:**

If the `include_docs` flag in the request body is true then you will get a list of document IDs associated with a given collection as well. Note that, documents returned are not only stored by Kaapi, but also by Vector store provider.

Additionally, if you set the `include_url` parameter to true, a signed URL will be included in the response, which is a clickable link to access the retrieved document. If you don't set it to true, the URL will not be included in the response.
