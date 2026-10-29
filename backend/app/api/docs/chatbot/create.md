Create a chatbot in the caller's project and return it (`201`).

Only `name` is required (1–255 characters; display only). A new chatbot has no flow: save one with `PUT /chatbots/{chatbot_id}/flow` and publish it with `POST /chatbots/{chatbot_id}/flow/publish` before starting sessions.
