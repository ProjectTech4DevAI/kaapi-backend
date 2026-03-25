import logging
from datetime import datetime, timezone

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def raise_from_unknown(error: Exception, status_code=500):
    logger.warning(
        'Unexpected exception "{}": {}'.format(
            type(error).__name__,
            error,
        )
    )
    raise HTTPException(status_code=status_code, detail=str(error))


def configure_openai(credentials: dict) -> tuple["OpenAI", bool]:
    """
    Configure OpenAI client with the provided credentials.

    Args:
        credentials: Dictionary containing OpenAI credentials (api_key)

    Returns:
        Tuple of (OpenAI client instance, success boolean)
    """
    from openai import OpenAI

    if not credentials or "api_key" not in credentials:
        return None, False

    try:
        # Configure OpenAI client
        client = OpenAI(api_key=credentials["api_key"])
        return client, True
    except Exception as e:
        logger.error(f"Failed to configure OpenAI client: {str(e)}")
        return None, False
