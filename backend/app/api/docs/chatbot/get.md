Get one chatbot of the caller's project by `chatbot_id`.

Returns 404 if the chatbot doesn't exist in this project or was deleted. The chatbot's flow is fetched separately with `GET /chatbots/{chatbot_id}/flow`.
