Soft-delete a chatbot; returns `{"message": "Chatbot deleted successfully"}`. It disappears from list/get and its flow can no longer be edited, published, or used for new sessions.

In the same transaction every `active` session of the chatbot becomes `inactive` (those still `in_progress` become `abandoned`; `complete` ones stay `complete`), so further messages to them return 409. Stored sessions and transcripts are kept. Returns 404 if the chatbot doesn't exist in this project or was already deleted.
