class ChatbotInputError(ValueError):
    """The turn request can't produce a valid model call (caller's fault)."""


class ChatbotProviderError(Exception):
    """The chat-model provider failed or was unreachable."""
