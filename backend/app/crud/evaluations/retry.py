"""Shared retry policies for the synchronous evaluation stages.

Embeddings and the judge issue single-row OpenAI calls from a worker thread pool
and share one transient-error policy rather than each declaring its own. Generation
goes through `execute_llm_call`, which reports failure by return value instead of
raising, so it gets a result-based policy over the same limits — keyed off the
`retryable` flag the call sets, since the exception type is gone by then.
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
    """True only for a failure `execute_llm_call` marked as worth re-running.

    The classification lives on `BlockResult.retryable` because `error` is one
    string for every failure kind — a config that doesn't resolve, a rejected model,
    a revoked key and a guardrails auth fail-closed all look identical here, and
    none of them can succeed on a second attempt. Guardrail verdicts are content
    decisions and never set the flag either.
    """
    return result.error is not None and result.retryable


def _last_llm_result(retry_state: RetryCallState) -> BlockResult:
    """The final attempt's `BlockResult`, returned instead of raising `RetryError`.

    `reraise` does not apply to `retry_if_result`, so without this an exhausted
    retry raises into the caller's thread pool and kills the whole chunk — which the
    cron healer then re-enqueues, re-charging the provider for every row in it.
    """
    outcome = retry_state.outcome
    if outcome is None:  # unreachable: tenacity records an attempt before calling back
        raise RuntimeError("retry_error_callback fired with no attempt outcome")
    return outcome.result()


def retry_llm_call(
    logger: logging.Logger,
) -> Callable[[Callable[P, BlockResult]], Callable[P, BlockResult]]:
    """Tenacity decorator retrying an `execute_llm_call` result flagged `retryable`,
    with the same limits as `retry_openai_call`.

    A fail-closed guardrails auth error still fails the row — deliberate: broken
    guardrail credentials must not read as a silently unscoreable run. It is not
    retried, though. Retrying could never clear it, and on the *output* side each
    attempt would re-charge the provider, since the completion is generated before
    output guardrails run.
    """
    return retry(
        retry=retry_if_result(_is_retryable_llm_result),
        wait=wait_random_exponential(
            multiplier=RETRY_BASE_DELAY_SECONDS, max=RETRY_MAX_DELAY_SECONDS
        ),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.INFO),
        retry_error_callback=_last_llm_result,
    )
