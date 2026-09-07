"""Shared OpenAI retry policy for the synchronous evaluation stages.

Responses, embeddings and the judge all issue single-row OpenAI calls from a
worker thread pool, so they share one transient-error policy rather than each
declaring its own.
"""

import logging
from collections.abc import Callable
from typing import Any

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 30.0

RETRYABLE_OPENAI_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def retry_openai_call(logger: logging.Logger) -> Callable[..., Any]:
    """Tenacity decorator retrying transient OpenAI errors with jittered backoff.

    `reraise=True` so call-site handlers see the original `OpenAIError` rather
    than tenacity's `RetryError`.
    """
    return retry(
        retry=retry_if_exception_type(RETRYABLE_OPENAI_ERRORS),
        wait=wait_random_exponential(
            multiplier=RETRY_BASE_DELAY_SECONDS, max=RETRY_MAX_DELAY_SECONDS
        ),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
