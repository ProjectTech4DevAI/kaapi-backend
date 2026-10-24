CONCURRENT_SESSION_CONFLICT_MESSAGE = (
    "A concurrent session was just created for this contact and chatbot; retry"
)


class ChatbotInputError(ValueError):
    """The turn request can't produce a valid model call (caller's fault)."""


class ChatbotRefGenerationError(Exception):
    """No unused chatbot_ref could be generated for the project."""


class ChatbotProviderError(Exception):
    """The chat-model provider failed or was unreachable."""


class ChatbotSessionConflictError(Exception):
    """Another active session for the same contact + chatbot was committed concurrently."""
