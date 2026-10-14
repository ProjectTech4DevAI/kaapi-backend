"""Retry policies for synchronous evaluation stages.

Embeddings/judge retry on exceptions; generation retries on `retryable`, since
`execute_llm_call` returns failures instead of raising.
"""

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import openai
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)

from app.services.llm.chain.types import BlockResult

RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 30.0

RETRYABLE_OPENAI_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

P = ParamSpec("P")
R = TypeVar("R")


def retry_openai_call(
    logger: logging.Logger,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
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


def _is_retryable_llm_result(result: BlockResult) -> bool:
    """True only for failures `execute_llm_call` flagged retryable."""
    return result.error is not None and result.retryable


def _last_llm_result(retry_state: RetryCallState) -> BlockResult:
    """Return the final result on exhaustion; raising would kill the whole chunk."""
    outcome = retry_state.outcome
    if outcome is None:  # unreachable: tenacity records an attempt before calling back
        raise RuntimeError("retry_error_callback fired with no attempt outcome")
    return outcome.result()


def retry_llm_call(
    logger: logging.Logger,
) -> Callable[[Callable[P, BlockResult]], Callable[P, BlockResult]]:
    """Retry `execute_llm_call` results flagged `retryable`; same limits as OpenAI."""
    return retry(
        retry=retry_if_result(_is_retryable_llm_result),
        wait=wait_random_exponential(
            multiplier=RETRY_BASE_DELAY_SECONDS, max=RETRY_MAX_DELAY_SECONDS
        ),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.INFO),
        retry_error_callback=_last_llm_result,
    )
