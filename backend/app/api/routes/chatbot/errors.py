from fastapi import HTTPException

from app.services.chatbot.exceptions import (
    ChatbotFlowConflictError,
    ChatbotFlowNotFoundError,
    ChatbotFlowValidationError,
    ChatbotNotFoundError,
    ChatbotSessionConflictError,
)

_STATUS_BY_CHATBOT_ERROR: dict[type[Exception], int] = {
    ChatbotNotFoundError: 404,
    ChatbotFlowNotFoundError: 404,
    ChatbotFlowConflictError: 409,
    ChatbotSessionConflictError: 409,
    ChatbotFlowValidationError: 422,
}
CHATBOT_SERVICE_ERRORS: tuple[type[Exception], ...] = tuple(_STATUS_BY_CHATBOT_ERROR)


def to_http_exception(error: Exception) -> HTTPException:
    """Map a chatbot service error to the HTTP status every chatbot route uses for it."""
    for error_type, status_code in _STATUS_BY_CHATBOT_ERROR.items():
        if isinstance(error, error_type):
            return HTTPException(status_code=status_code, detail=str(error))
    raise TypeError(
        f"[to_http_exception] Unmapped chatbot error: {type(error).__name__}"
    )
