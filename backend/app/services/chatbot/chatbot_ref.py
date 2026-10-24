import logging
import secrets

from sqlmodel import Session

from app.crud.chatbot import chatbot_ref_exists
from app.services.chatbot.exceptions import ChatbotRefGenerationError

logger = logging.getLogger(__name__)

CHATBOT_REF_PREFIX = "LLMResponseWithPrompt-"
CHATBOT_REF_SUFFIX_LENGTH = 5
HEX_BASE = 16
CHATBOT_REF_MAX_ATTEMPTS = 5


def _random_chatbot_ref() -> str:
    suffix_value = secrets.randbelow(HEX_BASE**CHATBOT_REF_SUFFIX_LENGTH)
    return f"{CHATBOT_REF_PREFIX}{suffix_value:0{CHATBOT_REF_SUFFIX_LENGTH}x}"


def generate_chatbot_ref(session: Session, *, project_id: int) -> str:
    """Return a chatbot_ref not yet used by any session in the project."""
    for _attempt in range(CHATBOT_REF_MAX_ATTEMPTS):
        candidate = _random_chatbot_ref()
        if not chatbot_ref_exists(
            session, project_id=project_id, chatbot_ref=candidate
        ):
            return candidate

    error_message = (
        f"[KAAPI] Could not generate an unused chatbot_ref after "
        f"{CHATBOT_REF_MAX_ATTEMPTS} attempts; retry, or pass an existing chatbot_ref. "
        f"If this persists, contact Kaapi. (code: ChatbotRefGenerationError)"
    )
    logger.error(f"[generate_chatbot_ref] {error_message} | project_id: {project_id}")
    raise ChatbotRefGenerationError(error_message)
