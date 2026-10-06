"""Webhook delivery for the BATCH API-client path.

POSTs the ``AssessmentCallback`` envelope to the client's callback_url on completion via
the shared SSRF-guarded ``send_callback`` (HMAC-signed with the project webhook secret) —
the same transport the response path uses. One inline attempt, no retry.
"""

import logging

from pydantic import JsonValue
from sqlmodel import Session

from app.models.assessment import (
    Assessment,
    AssessmentBatchResult,
    AssessmentCallback,
    AssessmentStatus,
)
from app.services.assessment.api.result_files import presign_result_files
from app.utils import get_webhook_secret, send_callback

logger = logging.getLogger(__name__)


def deliver(
    *,
    session: Session,
    assessment: Assessment,
    result: AssessmentBatchResult,
    callback_url: str,
    request_metadata: dict[str, JsonValue] | None,
    failure_message: str | None,
) -> bool:
    """POST the assessment result to ``callback_url`` (HMAC-signed). Returns whether it was sent.

    ``request_metadata`` is the client's own echo; the envelope ``metadata`` carries the
    presigned result-file urls, and ``failure_message`` becomes the envelope ``error``.
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

    try:
        files = presign_result_files(session=session, assessment=assessment)
    except Exception:
        # A metadata bug must never cost the client its result.
        logger.error(
            "[deliver] Callback metadata failed, delivering without it | assessment_id=%s",
            assessment.id,
            exc_info=True,
        )
        files = None

    sent = send_callback(
        callback_url,
        {
            "success": assessment.status != AssessmentStatus.FAILED,
            "data": callback.model_dump(mode="json"),
            "error": failure_message,
            "metadata": {"files": files.model_dump()} if files else None,
        },
        webhook_secret=webhook_secret,
    )
    logger.info(
        "[deliver] Callback %s | assessment_id=%s | status=%s | files=%s",
        "sent" if sent else "failed",
        assessment.id,
        assessment.status,
        files,
    )
    return sent
