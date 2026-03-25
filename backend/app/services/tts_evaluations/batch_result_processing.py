"""Celery task function for TTS evaluation result processing.

Processes completed Gemini TTS batch results: downloads JSONL,
extracts audio, converts PCM to WAV, uploads to S3, updates DB.
"""

import base64
import logging
import uuid
from typing import Any

from sqlmodel import Session, select

from app.core.batch import BATCH_KEY, GeminiBatchProvider, GeminiClient
from app.core.cloud.storage import get_cloud_storage
from app.core.db import engine
from app.core.storage_utils import upload_to_object_store
from app.core.util import now
from app.crud.tts_evaluations.result import count_results_by_status
from app.crud.tts_evaluations.run import update_tts_run
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult

logger = logging.getLogger(__name__)



def _extract_audio_from_response(response: dict[str, Any]) -> str | None:
    """Extract base64-encoded audio data from a Gemini TTS response.

    Gemini TTS returns audio as base64-encoded PCM data in the
    inlineData field of the response parts. Handles both camelCase
    (REST API) and snake_case (Python SDK / batch JSONL) field names.

    Args:
        response: Gemini response dictionary

    Returns:
        Base64 encoded audio string, or None if not found
    """
    # Navigate: candidates -> content -> parts -> inlineData/inline_data -> data
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            # Handle both camelCase (inlineData) and snake_case (inline_data)
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            if inline_data.get("data"):
                return inline_data["data"]

    part_keys = [
        list(p.keys())
        for c in response.get("candidates", [])
        for p in c.get("content", {}).get("parts", [])
    ]
    logger.warning(
        f"[_extract_audio_from_response] No audio data found | "
        f"response_keys={list(response.keys())}, parts={part_keys}"
    )
    return None
