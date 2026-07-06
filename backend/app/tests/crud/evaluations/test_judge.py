"""Unit tests for the native LLM-as-a-judge correctness scorer (`crud/evaluations/judge.py`).

Covers FR-2 (score in [0,1] + non-empty reasoning parsing), FR-3 (zero-config
fallback model + built-in prompt), FR-4 (ad-hoc blob model/settings/template),
FR-5/FR-12 (stored-ref resolution + tenant isolation at resolve_judge_blob),
FR-7 (LLMCallConfig one-of validator), and FR-10 (per-row failure: judge_row
raises on transient-exhausted OpenAI errors or malformed output).

The only mocked boundary is OpenAI (`openai_client.responses.create`); the DB is
real via the transactional `db` fixture, since build_judge_params/resolve_judge_blob
run the real mapper and the real scoped config resolution.
"""

from types import SimpleNamespace
from uuid import uuid4

import openai
import pytest
from pydantic import ValidationError
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.judge import (
    JudgeResult,
    _parse_judge_output,
    build_judge_params,
    judge_row,
    resolve_judge_blob,
)
from app.crud.evaluations.score import DEFAULT_JUDGE_PROMPT, JUDGE_OUTPUT_INSTRUCTION
from app.models.llm.request import (
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
    PromptTemplate,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_config, create_test_project


def _judge_response(text: str, *, input_tokens: int = 10, output_tokens: int = 5):
    """Mimic the OpenAI Responses SDK return shape the judge reads."""
    return SimpleNamespace(
        output_text=text,
        output=[],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _blob(
    *, model: str = "gpt-4o", temperature: float = 0.0, template: str | None = None
):
    return ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={"model": model, "temperature": temperature},
        ),
        prompt_template=PromptTemplate(template=template) if template else None,
    )


class TestParseJudgeOutput:
    """FR-2: the parser enforces score in [0,1] and a non-empty reasoning string."""

    def test_valid_json_returns_score_and_reasoning(self) -> None:
        score, reasoning = _parse_judge_output(
            '{"score": 0.8, "reasoning": "Mostly correct, minor omission."}'
        )
        assert score == 0.8
        assert reasoning == "Mostly correct, minor omission."

    def test_boundary_scores_zero_and_one_accepted(self) -> None:
        assert _parse_judge_output('{"score": 0, "reasoning": "wrong"}')[0] == 0.0
        assert _parse_judge_output('{"score": 1, "reasoning": "perfect"}')[0] == 1.0

    def test_prose_wrapped_json_is_extracted(self) -> None:
        score, reasoning = _parse_judge_output(
            'Here is my judgment: {"score": 0.5, "reasoning": "partial"} done.'
        )
        assert score == 0.5
        assert reasoning == "partial"

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="out of"):
            _parse_judge_output('{"score": 1.4, "reasoning": "ok"}')

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="out of"):
            _parse_judge_output('{"score": -0.2, "reasoning": "ok"}')

    def test_empty_reasoning_raises(self) -> None:
        with pytest.raises(ValueError, match="reasoning"):
            _parse_judge_output('{"score": 0.7, "reasoning": "   "}')

    def test_missing_score_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            _parse_judge_output('{"reasoning": "no score here"}')

    def test_non_numeric_score_raises(self) -> None:
        with pytest.raises(ValueError, match="not a number"):
            _parse_judge_output('{"score": "high", "reasoning": "ok"}')

    def test_non_json_text_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            _parse_judge_output("the answer is completely correct")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _parse_judge_output("   ")


class TestResolveJudgeBlob:
    """FR-4/FR-5/FR-12: how a run's judge_config maps to a ConfigBlob."""

    def test_none_config_returns_none_for_zero_config_default(
        self, db: Session
    ) -> None:
        blob = resolve_judge_blob(session=db, judge_config=None, project_id=1)
        assert blob is None

    def test_adhoc_blob_is_used_directly(self, db: Session) -> None:
        adhoc = _blob(model="gpt-4o", template="grade strictly")
        judge_config = LLMCallConfig(blob=adhoc)

        blob = resolve_judge_blob(session=db, judge_config=judge_config, project_id=1)

        assert blob is adhoc

    def test_stored_reference_resolves_to_its_blob(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        config = create_test_config(
            db, project_id=user_api_key.project_id, use_kaapi_schema=True
        )
        judge_config = LLMCallConfig(id=config.id, version=1)

        blob = resolve_judge_blob(
            session=db,
            judge_config=judge_config,
            project_id=user_api_key.project_id,
        )

        assert blob is not None
        assert blob.completion.params["model"] == "gpt-4o"

    def test_stored_reference_from_other_project_is_not_resolvable(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """FR-12: a config saved in another project never resolves for this run."""
        other_project = create_test_project(db)
        foreign_config = create_test_config(
            db, project_id=other_project.id, use_kaapi_schema=True
        )
        judge_config = LLMCallConfig(id=foreign_config.id, version=1)

        with pytest.raises(ValueError, match="Failed to resolve"):
            resolve_judge_blob(
                session=db,
                judge_config=judge_config,
                project_id=user_api_key.project_id,
            )

    def test_unknown_stored_reference_raises(self, db: Session) -> None:
        judge_config = LLMCallConfig(id=uuid4(), version=1)
        with pytest.raises(ValueError, match="Failed to resolve"):
            resolve_judge_blob(session=db, judge_config=judge_config, project_id=1)


class TestBuildJudgeParams:
    """FR-3/FR-4: model + settings + system prompt selection."""

    def test_zero_config_uses_fallback_model_and_default_prompt(
        self, db: Session
    ) -> None:
        base_params, prompt = build_judge_params(session=db, blob=None)

        assert base_params["model"] == settings.EVAL_JUDGE_FALLBACK_MODEL
        assert base_params["instructions"] == DEFAULT_JUDGE_PROMPT
        assert prompt == DEFAULT_JUDGE_PROMPT

    def test_blob_supplies_model_and_temperature(self, db: Session) -> None:
        base_params, prompt = build_judge_params(
            session=db, blob=_blob(model="gpt-4o", temperature=0.5)
        )

        assert base_params["model"] == "gpt-4o"
        assert base_params["temperature"] == 0.5
        # No template on the blob → built-in prompt.
        assert prompt == DEFAULT_JUDGE_PROMPT

    def test_blob_prompt_template_overrides_default_prompt(self, db: Session) -> None:
        base_params, prompt = build_judge_params(
            session=db, blob=_blob(template="Example: Query ... Score: 0.3")
        )

        assert prompt == "Example: Query ... Score: 0.3"
        assert base_params["instructions"] == "Example: Query ... Score: 0.3"

    def test_bot_instructions_and_kb_are_stripped_from_judge(self, db: Session) -> None:
        """A bot's own instructions/KB must never leak into the grader."""
        blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                type="text",
                params={
                    "model": "gpt-4o",
                    "instructions": "You are the NGO helpdesk bot.",
                    "knowledge_base_ids": ["vs_123"],
                },
            )
        )

        base_params, prompt = build_judge_params(session=db, blob=blob)

        assert prompt == DEFAULT_JUDGE_PROMPT
        assert base_params["instructions"] == DEFAULT_JUDGE_PROMPT
        assert "tools" not in base_params


class TestJudgeRow:
    """FR-2/FR-10: one judge completion per row; raises so the caller can isolate."""

    def test_happy_path_returns_score_reasoning_usage(self) -> None:
        client = _make_client(
            _judge_response(
                '{"score": 0.9, "reasoning": "Accurate and complete."}',
                input_tokens=12,
                output_tokens=8,
            )
        )

        result = judge_row(
            openai_client=client,
            base_params={"model": "gpt-4o-mini", "instructions": DEFAULT_JUDGE_PROMPT},
            question="What is the capital of France?",
            generated_answer="Paris",
            ground_truth="Paris",
        )

        assert isinstance(result, JudgeResult)
        assert result.score == 0.9
        assert result.reasoning == "Accurate and complete."
        assert result.usage == {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        }

    def test_row_qa_and_output_contract_are_appended_to_input(self) -> None:
        client = _make_client(_judge_response('{"score": 0.5, "reasoning": "partial"}'))

        judge_row(
            openai_client=client,
            base_params={"model": "gpt-4o-mini"},
            question="Q-TEXT",
            generated_answer="A-TEXT",
            ground_truth="GT-TEXT",
        )

        sent_input = client.responses.create.call_args.kwargs["input"]
        assert "Q-TEXT" in sent_input
        assert "A-TEXT" in sent_input
        assert "GT-TEXT" in sent_input
        assert JUDGE_OUTPUT_INSTRUCTION in sent_input

    def test_malformed_output_raises(self) -> None:
        client = _make_client(_judge_response("not json at all"))

        with pytest.raises(ValueError):
            judge_row(
                openai_client=client,
                base_params={"model": "gpt-4o-mini"},
                question="Q",
                generated_answer="A",
                ground_truth="GT",
            )

    def test_openai_error_propagates(self) -> None:
        from unittest.mock import MagicMock

        # Real int status_code so the handler's `status >= 500` branch works.
        response = MagicMock()
        response.status_code = 401
        client = _make_client(
            openai.AuthenticationError(message="bad key", response=response, body=None)
        )

        with pytest.raises(openai.OpenAIError):
            judge_row(
                openai_client=client,
                base_params={"model": "gpt-4o-mini"},
                question="Q",
                generated_answer="A",
                ground_truth="GT",
            )


class TestLLMCallConfigValidator:
    """FR-7: exactly one of (id+version) XOR blob is valid."""

    def test_both_stored_ref_and_blob_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not both"):
            LLMCallConfig(id=uuid4(), version=1, blob=_blob())

    def test_neither_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMCallConfig()

    def test_id_without_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMCallConfig(id=uuid4())

    def test_blob_only_is_valid(self) -> None:
        cfg = LLMCallConfig(blob=_blob())
        assert cfg.blob is not None
        assert not cfg.is_stored_config

    def test_stored_ref_only_is_valid(self) -> None:
        cfg = LLMCallConfig(id=uuid4(), version=3)
        assert cfg.is_stored_config


def _make_client(create_return_or_raise):
    """Build a fake OpenAI client whose responses.create returns or raises."""
    from unittest.mock import MagicMock

    client = MagicMock()
    if isinstance(create_return_or_raise, Exception):
        client.responses.create.side_effect = create_return_or_raise
    else:
        client.responses.create.return_value = create_return_or_raise
    return client
