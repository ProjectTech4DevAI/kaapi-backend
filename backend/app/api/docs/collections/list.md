List all _active_ collections that have been created and are not deleted.

**Response Fields:**

**Note:** While the API schema shows both `llm_service_id`/`llm_service_name` AND `knowledge_base_id`/`knowledge_base_provider`, each collection in the response will only include the fields relevant to what was created:

- **If an Assistant was created** (with model + instructions): The response will only include `llm_service_id` and `llm_service_name` (e.g., `llm_service_name: "gpt-4o"` and the assistant ID)
- **If only a Vector Store was created** (without model/instructions): The response will only include `knowledge_base_id` and `knowledge_base_provider` (e.g., `knowledge_base_provider: "openai vector store"` and the vector store ID)
