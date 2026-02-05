from typing import Any
from uuid import UUID
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def call_guardrails(
    input_text: str, guardrail_config: list[dict], job_id: UUID
) -> dict[str, Any]:
    """
    Call the Kaapi guardrails service to validate and process input text.

    Args:
        input_text: Text to validate and process.
        guardrail_config: List of validator configurations to apply.
        job_id: Unique identifier for the request.

    Returns:
        JSON response from the guardrails service with validation results.
    """
    payload = {
        "request_id": str(job_id),
        "input": input_text,
        "validators": guardrail_config,
    }

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                settings.KAAPI_GUARDRAILS_URL,
                json=payload,
                headers=headers,
            )

            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(
            f"[call_guardrails] Service unavailable. Bypassing guardrails. job_id={job_id}. error={e}"
        )

        return {
            "success": False,
            "bypassed": True,
            "data": {
                "safe_text": input_text,
                "rephrase_needed": False,
            },
        }
