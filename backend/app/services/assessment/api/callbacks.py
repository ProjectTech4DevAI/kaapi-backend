"""Webhook delivery for the BATCH API-client path.

POSTs the ``AssessmentCallback`` envelope to the client's callback_url on completion via
the shared SSRF-guarded ``send_callback`` (HMAC-signed with the project webhook secret) —
the same transport the response path uses.
"""

import logging
from typing import Any

from app.models.assessment import (
    Assessment,
    AssessmentBatchResult,
    AssessmentCallback,
    AssessmentStatus,
)
from app.utils import get_webhook_secret, send_callback

logger = logging.getLogger(__name__)


def deliver(
    *,
    assessment: Assessment,
    result: AssessmentBatchResult,
    callback_url: str,
    request_metadata: dict[str, Any] | None,
) -> bool:
    """POST the assessment result to ``callback_url`` (HMAC-signed). Returns whether it was sent.

    ``request_metadata`` is echoed back unchanged for client-side correlation.
    """
    callback = AssessmentCallback(
        assessment_id=assessment.id,
        status=assessment.status,
        data=result,
        request_metadata=request_metadata,
    )
    webhook_secret = get_webhook_secret(
        assessment.project_id, assessment.organization_id
    )
    sent = send_callback(
        callback_url,
        {
            "success": assessment.status != AssessmentStatus.FAILED,
            "data": callback.model_dump(mode="json"),
            "error": None,
            "metadata": None,
        },
        webhook_secret=webhook_secret,
    )
    logger.info(
        "[deliver] Callback %s | assessment_id=%s | status=%s",
        "sent" if sent else "failed",
        assessment.id,
        assessment.status,
    )
    return sent
