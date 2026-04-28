"""Live evaluation: per-row Responses + Embeddings execution.

Runs inside a single Celery task per dataset item. Returns a dict matching
the shape `parse_evaluation_output()` produces in batch mode, plus an `error`
field and the two embedding vectors used for cosine similarity. The aggregator
partitions on `error` and reuses the rest of the existing post-processing path.
"""

import logging
from typing import Any

import openai
from openai import OpenAI
from sqlmodel import Session

from app.core.db import engine
from app.crud.evaluations.batch import build_response_body
from app.crud.evaluations.core import resolve_evaluation_config
from app.crud.evaluations.embeddings import EMBEDDING_MODEL
from app.models.llm.request import STTLLMParams, TextLLMParams, TTSLLMParams
from app.utils import get_openai_client, handle_openai_error

logger = logging.getLogger(__name__)

# Errors worth re-raising so Celery's autoretry can back off and try again.
# Anything else (auth, validation, model-not-found, etc.) is a permanent
# failure for this row — captured in the result dict so the aggregator can
# count it without crashing the chord.
RETRYABLE_OPENAI_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def _resolve_text_params(
    session: Session, eval_run_id: int, project_id: int, config_id, config_version: int
) -> TextLLMParams:
    config, error = resolve_evaluation_config(
        session=session,
        config_id=config_id,
        config_version=config_version,
        project_id=project_id,
    )
    if error or config is None:
        raise ValueError(
            f"[execute_eval_row] Config resolution failed | eval_run_id={eval_run_id} | "
            f"error={error}"
        )

    param_models = {
        "text": TextLLMParams,
        "stt": STTLLMParams,
        "tts": TTSLLMParams,
    }
    model_class = param_models.get(config.completion.type)
    if model_class is not TextLLMParams:
        raise ValueError(
            f"[execute_eval_row] Live mode only supports text evaluations | "
            f"eval_run_id={eval_run_id} | type={config.completion.type}"
        )
    return TextLLMParams.model_validate(config.completion.params)


def _build_error_result(item: dict[str, Any], error_message: str) -> dict[str, Any]:
    """Return a result dict marked as failed. Aggregator partitions on `error`."""
    question = item.get("input", {}).get("question", "")
    ground_truth = item.get("expected_output", {}).get("answer", "")
    question_id = item.get("metadata", {}).get("question_id")
    return {
        "item_id": item["id"],
        "question": question,
        "generated_output": "",
        "ground_truth": ground_truth,
        "response_id": None,
        "usage": None,
        "question_id": question_id,
        "error": error_message,
        "output_embedding": None,
        "ground_truth_embedding": None,
        "embedding_input_tokens": 0,
    }


def execute_eval_row(
    *,
    eval_run_id: int,
    item: dict[str, Any],
    organization_id: int,
    project_id: int,
    config_id: Any,
    config_version: int,
) -> dict[str, Any]:
    """Run the Responses + Embeddings pair for one dataset item.

    Resolves the config and OpenAI client per-call (so each Celery task is
    self-contained and reflects current credentials). Returns a result dict
    on both success and permanent failure; raises on retryable transient
    errors so the task wrapper can back off.
    """
    log_prefix = f"[execute_eval_row][eval={eval_run_id}][item={item.get('id')}]"

    question = item.get("input", {}).get("question", "")
    ground_truth = item.get("expected_output", {}).get("answer", "")
    question_id = item.get("metadata", {}).get("question_id")

    if not question:
        logger.warning(f"{log_prefix} Skipping item - no question found")
        return _build_error_result(item, "Item has no question")

    with Session(engine) as session:
        try:
            text_params = _resolve_text_params(
                session=session,
                eval_run_id=eval_run_id,
                project_id=project_id,
                config_id=config_id,
                config_version=config_version,
            )
            client: OpenAI = get_openai_client(
                session=session,
                org_id=organization_id,
                project_id=project_id,
            )
        except Exception as e:
            logger.error(f"{log_prefix} Setup failed | {e}", exc_info=True)
            return _build_error_result(item, f"Setup failed: {e}")

    body = build_response_body(question=question, config=text_params)

    try:
        response = client.responses.create(**body)
    except RETRYABLE_OPENAI_ERRORS:
        # Let Celery autoretry handle these — re-raise.
        raise
    except openai.OpenAIError as e:
        error_message = handle_openai_error(e)
        logger.error(f"{log_prefix} Responses API permanent error | {error_message}")
        return _build_error_result(item, error_message)

    generated_output = response.output_text or ""
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    if not generated_output or not ground_truth:
        # No way to compute cosine similarity without both texts. Keep the
        # response result for trace creation, but skip embeddings.
        logger.info(f"{log_prefix} Skipping embeddings - empty output or ground_truth")
        return {
            "item_id": item["id"],
            "question": question,
            "generated_output": generated_output,
            "ground_truth": ground_truth,
            "response_id": response.id,
            "usage": usage,
            "question_id": question_id,
            "error": None,
            "output_embedding": None,
            "ground_truth_embedding": None,
            "embedding_input_tokens": 0,
        }

    try:
        embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[generated_output, ground_truth],
            encoding_format="float",
        )
    except RETRYABLE_OPENAI_ERRORS:
        raise
    except openai.OpenAIError as e:
        # Keep the response result; the aggregator skips embeddings for this row.
        logger.error(
            f"{log_prefix} Embeddings API error, continuing without similarity | "
            f"{handle_openai_error(e)}"
        )
        return {
            "item_id": item["id"],
            "question": question,
            "generated_output": generated_output,
            "ground_truth": ground_truth,
            "response_id": response.id,
            "usage": usage,
            "question_id": question_id,
            "error": None,
            "output_embedding": None,
            "ground_truth_embedding": None,
            "embedding_input_tokens": 0,
        }

    output_embedding = None
    ground_truth_embedding = None
    for emb in embedding_response.data:
        if emb.index == 0:
            output_embedding = emb.embedding
        elif emb.index == 1:
            ground_truth_embedding = emb.embedding

    embedding_input_tokens = (
        embedding_response.usage.prompt_tokens
        if embedding_response.usage is not None
        else 0
    )

    return {
        "item_id": item["id"],
        "question": question,
        "generated_output": generated_output,
        "ground_truth": ground_truth,
        "response_id": response.id,
        "usage": usage,
        "question_id": question_id,
        "error": None,
        "output_embedding": output_embedding,
        "ground_truth_embedding": ground_truth_embedding,
        "embedding_input_tokens": embedding_input_tokens,
    }
