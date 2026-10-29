CONCURRENT_SESSION_CONFLICT_MESSAGE = (
    "A concurrent session was just created for this contact and chatbot; retry"
)
CONCURRENT_FLOW_CONFLICT_MESSAGE = (
    "The chatbot flow was changed concurrently; reload the flow and retry"
)


class ChatbotInputError(ValueError):
    """The turn request can't produce a valid model call (caller's fault)."""


class ChatbotProviderError(Exception):
    """The chat-model provider failed or was unreachable."""


class ChatbotSessionConflictError(Exception):
    """Another active session for the same contact + chatbot was committed concurrently."""


class ChatbotNotFoundError(Exception):
    """The chatbot doesn't exist in the caller's project or was deleted."""


class ChatbotFlowNotFoundError(Exception):
    """The chatbot has no flow version (or not the requested one)."""


class ChatbotFlowConflictError(Exception):
    """The flow isn't in the state the operation needs (no draft, nothing live, or a concurrent write)."""


class ChatbotFlowValidationError(ValueError):
    """The flow fails publish-time validation (caller's fault)."""
