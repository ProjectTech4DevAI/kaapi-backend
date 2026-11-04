Create a chatbot in the caller's project and return it (`201`).

Only `name` is required (1–255 characters; display only). A new chatbot has no flow: its first `PUT /chatbots/{chatbot_id}/flow` goes live at once, after which sessions can start. Later saves are drafts that go live with `POST /chatbots/{chatbot_id}/flow/publish`.
