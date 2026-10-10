import uuid
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.jobs import JobCrud
from app.crud.llm import save_rephrase_guardrail_call
from app.models import Job, JobType
from app.models.llm.request import (
    ConfigBlob,
    InlineValidator,
    LLMCallConfig,
    LlmCall,
    NativeCompletionConfig,
    QueryParams,
    Validator,
)
from app.services.llm.guardrails import (
    GuardrailsOutcome,
    apply_guardrails,
    list_validators_config,
    run_guardrails_validation,
    summarize_validator_results,
)
from app.tests.utils.utils import get_project


TEST_JOB_ID = uuid.uuid4()
TEST_TEXT = "hello world"
TEST_CONFIG = [{"type": "pii_remover"}]
TEST_PROJECT_ID = 1
TEST_ORGANIZATION_ID = 1


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_success(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"success": True}
    mock_response.raise_for_status.return_value = None

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = run_guardrails_validation(
        TEST_TEXT,
        TEST_CONFIG,
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
    )

    assert result == {"success": True}
    mock_client.post.assert_called_once()

    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["input"] == TEST_TEXT
    assert kwargs["json"]["validators"] == TEST_CONFIG
    assert kwargs["json"]["request_id"] == str(TEST_JOB_ID)
    assert "project_id" not in kwargs["json"]
    assert "organization_id" not in kwargs["json"]
    assert kwargs["headers"]["X-PROJECT-ID"] == str(TEST_PROJECT_ID)
    assert kwargs["headers"]["X-ORGANIZATION-ID"] == str(TEST_ORGANIZATION_ID)
    assert kwargs["params"]["suppress_pass_logs"] == "false"
    assert kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert kwargs["headers"]["Content-Type"] == "application/json"


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_http_error_bypasses(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad", request=MagicMock(), response=mock_response
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = run_guardrails_validation(
        TEST_TEXT,
        TEST_CONFIG,
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
    )

    assert result["success"] is False
    assert result["bypassed"] is True
    assert result["data"]["safe_text"] == TEST_TEXT


@pytest.mark.parametrize("status_code", [401, 403, 422])
@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_auth_error_fails_closed(
    mock_client_cls, status_code
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "unauthorized", request=MagicMock(), response=mock_response
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = run_guardrails_validation(
        TEST_TEXT,
        TEST_CONFIG,
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
    )

    assert result["success"] is False
    assert result.get("bypassed") is False
    assert "rejected the request" in result["error"]


@patch("app.services.llm.guardrails.httpx.Client")
def test_list_validators_config_auth_error_raises(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "forbidden", request=MagicMock(), response=mock_response
    )

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    with pytest.raises(ValueError, match=r"rejected \(HTTP 403\)"):
        list_validators_config(
            input_validator_configs=[Validator(validator_config_id=uuid.uuid4())],
            output_validator_configs=[],
            organization_id=1,
            project_id=1,
        )


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_uses_settings(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"ok": True}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    run_guardrails_validation(
        TEST_TEXT,
        TEST_CONFIG,
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
    )

    _, kwargs = mock_client.post.call_args
    assert (
        kwargs["headers"]["Authorization"] == f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}"
    )


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_serializes_validator_models(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"success": True}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    vid = uuid.uuid4()
    run_guardrails_validation(
        TEST_TEXT,
        [Validator(validator_config_id=vid)],
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
    )

    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["validators"] == [{"validator_config_id": str(vid)}]


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_includes_output_in_payload(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"success": True}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    run_guardrails_validation(
        TEST_TEXT,
        TEST_CONFIG,
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
        output_text="some llm response",
    )

    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["input"] == TEST_TEXT
    assert kwargs["json"]["output"] == "some llm response"


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_omits_output_from_payload_by_default(
    mock_client_cls,
) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"success": True}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    run_guardrails_validation(
        TEST_TEXT,
        TEST_CONFIG,
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
    )

    _, kwargs = mock_client.post.call_args
    assert "output" not in kwargs["json"]


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_allows_disable_suppress_pass_logs(
    mock_client_cls,
) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"success": True}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    run_guardrails_validation(
        TEST_TEXT,
        TEST_CONFIG,
        TEST_JOB_ID,
        TEST_PROJECT_ID,
        TEST_ORGANIZATION_ID,
        suppress_pass_logs=False,
    )

    _, kwargs = mock_client.post.call_args
    assert kwargs["params"]["suppress_pass_logs"] == "false"


@patch("app.services.llm.guardrails.httpx.Client")
def test_list_validators_config_fetches_input_and_output_by_refs(
    mock_client_cls,
) -> None:
    input_validator_configs = [Validator(validator_config_id=uuid.uuid4())]
    output_validator_configs = [Validator(validator_config_id=uuid.uuid4())]

    input_response = MagicMock()
    input_response.raise_for_status.return_value = None
    input_response.json.return_value = {
        "success": True,
        "data": [{"type": "uli_slur_match", "config": {"severity": "high"}}],
    }
    output_response = MagicMock()
    output_response.raise_for_status.return_value = None
    output_response.json.return_value = {
        "success": True,
        "data": [{"type": "gender_assumption_bias"}],
    }

    mock_client = MagicMock()
    mock_client.get.side_effect = [input_response, output_response]
    mock_client_cls.return_value.__enter__.return_value = mock_client

    input_guardrails, output_guardrails = list_validators_config(
        input_validator_configs=input_validator_configs,
        output_validator_configs=output_validator_configs,
        organization_id=1,
        project_id=1,
    )

    assert input_guardrails == [
        {"type": "uli_slur_match", "config": {"severity": "high"}}
    ]
    assert output_guardrails == [{"type": "gender_assumption_bias"}]
    assert mock_client.get.call_count == 2

    first_call_kwargs = mock_client.get.call_args_list[0].kwargs
    second_call_kwargs = mock_client.get.call_args_list[1].kwargs
    assert first_call_kwargs["params"]["ids"] == [
        str(v.validator_config_id) for v in input_validator_configs
    ]
    assert second_call_kwargs["params"]["ids"] == [
        str(v.validator_config_id) for v in output_validator_configs
    ]
    assert first_call_kwargs["headers"]["X-ORGANIZATION-ID"] == "1"
    assert first_call_kwargs["headers"]["X-PROJECT-ID"] == "1"


@patch("app.services.llm.guardrails.httpx.Client")
def test_list_validators_config_empty_short_circuits_without_http(
    mock_client_cls,
) -> None:
    input_guardrails, output_guardrails = list_validators_config(
        input_validator_configs=[],
        output_validator_configs=[],
        organization_id=1,
        project_id=1,
    )

    assert input_guardrails == []
    assert output_guardrails == []
    mock_client_cls.assert_not_called()


@patch("app.services.llm.guardrails.httpx.Client")
def test_list_validators_config_omits_none_tenant(mock_client_cls) -> None:
    input_validator_configs = [Validator(validator_config_id=uuid.uuid4())]

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"success": True, "data": []}

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    list_validators_config(
        input_validator_configs=input_validator_configs,
        output_validator_configs=[],
        organization_id=None,
        project_id=None,
    )

    _, kwargs = mock_client.get.call_args
    assert kwargs["params"]["ids"] == [
        str(v.validator_config_id) for v in input_validator_configs
    ]
    assert "organization_id" not in kwargs["params"]
    assert "project_id" not in kwargs["params"]
    assert "X-ORGANIZATION-ID" not in kwargs["headers"]
    assert "X-PROJECT-ID" not in kwargs["headers"]


@patch("app.services.llm.guardrails.httpx.Client")
def test_list_validators_config_network_error_fails_open(mock_client_cls) -> None:
    input_validator_configs = [Validator(validator_config_id=uuid.uuid4())]

    mock_client = MagicMock()
    mock_client.get.side_effect = httpx.ConnectError("Network is unreachable")
    mock_client_cls.return_value.__enter__.return_value = mock_client

    input_guardrails, output_guardrails = list_validators_config(
        input_validator_configs=input_validator_configs,
        output_validator_configs=[],
        organization_id=1,
        project_id=1,
    )

    assert input_guardrails == []
    assert output_guardrails == []


def test_summarize_validator_results_extracts_per_validator_fields() -> None:
    outcome = GuardrailsOutcome(
        safe_text="My credit card is [REDACTED]",
        error=None,
        bypassed=False,
        rephrase_needed=False,
        raw={
            "success": True,
            "data": {
                "safe_text": "My credit card is [REDACTED]",
                "validator_results": [
                    {
                        "name": "PIIRemover",
                        "type": "pii_remover",
                        "stage": "input",
                        "order": 1,
                        "outcome": "FAIL",
                        "error": "PII detected in the text.",
                        "input_text": "My credit card is 4111 1111 1111 1111",
                        "output_text": "My credit card is [REDACTED]",
                    }
                ],
            },
        },
    )

    summaries = summarize_validator_results(outcome)

    assert summaries == [
        {
            "name": "PIIRemover",
            "outcome": "FAIL",
            "error": "PII detected in the text.",
            "input_text": "My credit card is 4111 1111 1111 1111",
            "output_text": "My credit card is [REDACTED]",
        }
    ]


def test_summarize_validator_results_empty_when_raw_has_no_validator_results() -> None:
    outcome = GuardrailsOutcome(
        safe_text="hello",
        error=None,
        bypassed=True,
        rephrase_needed=False,
        raw={},
    )

    assert summarize_validator_results(outcome) == []


_PASS_RESPONSE = {"success": True, "data": {"safe_text": "clean"}}


def _ref() -> tuple[Validator, dict[str, Any]]:
    config_id = uuid4()
    return Validator(validator_config_id=config_id), {
        "id": str(config_id),
        "type": f"fetched_{config_id.hex[:8]}",
    }


def _apply(validators: list, output_text: str | None = None) -> GuardrailsOutcome:
    return apply_guardrails(
        text=TEST_TEXT,
        validators=validators,
        job_id=TEST_JOB_ID,
        project_id=TEST_PROJECT_ID,
        organization_id=TEST_ORGANIZATION_ID,
        output_text=output_text,
    )


def _assert_noop(outcome: GuardrailsOutcome) -> None:
    assert outcome.safe_text == TEST_TEXT
    assert outcome.error is None
    assert outcome.bypassed is False
    assert outcome.raw == {}
    assert outcome.applied is False


@patch("app.services.llm.guardrails.run_guardrails_validation")
@patch("app.services.llm.guardrails.list_validators_config")
class TestApplyGuardrails:
    def test_disabled_entries_are_neither_fetched_nor_posted(
        self, mock_fetch, mock_run
    ) -> None:
        enabled_ref, enabled_row = _ref()
        disabled_ref, _ = _ref()
        disabled_ref.is_enabled = False
        mock_fetch.return_value = ([enabled_row], [])
        mock_run.return_value = _PASS_RESPONSE

        _apply(
            [
                disabled_ref,
                enabled_ref,
                InlineValidator(type="off_inline", is_enabled=False),
            ]
        )

        fetched_ids = [
            v.validator_config_id
            for v in mock_fetch.call_args.kwargs["input_validator_configs"]
        ]
        assert fetched_ids == [enabled_ref.validator_config_id]
        assert mock_run.call_args.args[1] == [enabled_row]

    @pytest.mark.parametrize(
        "validators",
        [
            [],
            None,
            [
                Validator(validator_config_id=uuid4(), is_enabled=False),
                InlineValidator(type="pii_remover", is_enabled=False),
            ],
        ],
        ids=["empty", "none", "all_disabled"],
    )
    def test_nothing_enabled_is_noop_without_http(
        self, mock_fetch, mock_run, validators
    ) -> None:
        outcome = _apply(validators)

        _assert_noop(outcome)
        mock_fetch.assert_not_called()
        mock_run.assert_not_called()

    def test_inline_only_skips_fetch_and_posts(self, mock_fetch, mock_run) -> None:
        mock_run.return_value = _PASS_RESPONSE

        outcome = _apply([InlineValidator(type="pii_remover")])

        mock_fetch.assert_not_called()
        assert mock_run.call_args.args[1] == [{"type": "pii_remover", "stage": "input"}]
        assert outcome.safe_text == "clean"

    def test_posts_in_blob_order_not_fetch_order(self, mock_fetch, mock_run) -> None:
        ref_x, row_x = _ref()
        ref_y, row_y = _ref()
        mock_fetch.return_value = ([row_y, row_x], [])
        mock_run.return_value = _PASS_RESPONSE

        _apply([InlineValidator(type="a"), ref_x, InlineValidator(type="b"), ref_y])

        assert mock_run.call_args.args[1] == [
            {"type": "a", "stage": "input"},
            row_x,
            {"type": "b", "stage": "input"},
            row_y,
        ]

    def test_inline_spec_posted_shape(self, mock_fetch, mock_run) -> None:
        mock_run.return_value = _PASS_RESPONSE

        _apply(
            [
                InlineValidator(type="ban_list", stage="output", banned_words=["x"]),
                InlineValidator(type="pii_remover", on_fail="exception"),
            ]
        )

        assert mock_run.call_args.args[1] == [
            {"type": "ban_list", "stage": "input", "banned_words": ["x"]},
            {"type": "pii_remover", "on_fail": "exception", "stage": "input"},
        ]

    def test_output_path_sets_output_stage(self, mock_fetch, mock_run) -> None:
        ref, row = _ref()
        mock_fetch.return_value = ([], [row])
        mock_run.return_value = _PASS_RESPONSE

        _apply([ref, InlineValidator(type="pii_remover")], output_text="llm out")

        assert mock_fetch.call_args.kwargs["input_validator_configs"] is None
        assert mock_run.call_args.args[1] == [
            row,
            {"type": "pii_remover", "stage": "output"},
        ]
        assert mock_run.call_args.kwargs["output_text"] == "llm out"

    def test_ref_missing_upstream_is_dropped(self, mock_fetch, mock_run) -> None:
        found_ref, found_row = _ref()
        missing_ref, _ = _ref()
        mock_fetch.return_value = ([found_row], [])
        mock_run.return_value = _PASS_RESPONSE

        _apply([missing_ref, found_ref])

        assert mock_run.call_args.args[1] == [found_row]

    def test_only_ref_missing_upstream_is_noop(self, mock_fetch, mock_run) -> None:
        ref, _ = _ref()
        mock_fetch.return_value = ([], [])

        outcome = _apply([ref])

        _assert_noop(outcome)
        mock_run.assert_not_called()

    def test_fetch_failing_open_still_posts_inline_specs(
        self, mock_fetch, mock_run
    ) -> None:
        ref, _ = _ref()
        mock_fetch.return_value = ([], [])  # list_validators_config's fail-open shape
        mock_run.return_value = _PASS_RESPONSE

        _apply([ref, InlineValidator(type="pii_remover")])

        assert mock_run.call_args.args[1] == [{"type": "pii_remover", "stage": "input"}]


_SAFE_TEXT = "Please rephrase: content not allowed."
_CONFIG_BLOB = ConfigBlob(
    completion=NativeCompletionConfig(
        provider="openai-native",
        params={"model": "gpt-4o"},
        type="text",
    ),
    input_guardrails=[Validator(validator_config_id=uuid4())],
)
_CONFIG = LLMCallConfig(blob=_CONFIG_BLOB)


class TestSaveRephraseGuardrailCall:
    @pytest.fixture
    def job(self, db: Session) -> Job:
        j = JobCrud(session=db).create(
            job_type=JobType.LLM_API, trace_id="rephrase-test"
        )
        db.commit()
        return j

    def _call(self, db: Session, job: Job, **overrides: Any) -> UUID | None:
        project = get_project(db)
        kwargs = dict(
            session=db,
            query=QueryParams(input="original unsafe input"),
            config=_CONFIG,
            request_metadata=None,
            config_blob=_CONFIG_BLOB,
            guardrail_direct_response=_SAFE_TEXT,
            job_id=job.id,
            project_id=project.id,
            organization_id=project.organization_id,
            chain_id=None,
        )
        kwargs.update(overrides)
        return save_rephrase_guardrail_call(**kwargs)

    def test_success_returns_uuid(self, db: Session, job: Job) -> None:
        result = self._call(db, job)
        assert isinstance(result, UUID)

    def test_success_saves_original_input_and_job_id(
        self, db: Session, job: Job
    ) -> None:
        llm_call_id = self._call(db, job)
        llm_call = db.exec(select(LlmCall).where(LlmCall.id == llm_call_id)).first()
        assert llm_call is not None
        assert llm_call.input == "original unsafe input"
        assert llm_call.job_id == job.id

    def test_success_saves_safe_text_as_content(self, db: Session, job: Job) -> None:
        llm_call_id = self._call(db, job)
        llm_call = db.exec(select(LlmCall).where(LlmCall.id == llm_call_id)).first()
        assert llm_call.content == {
            "type": "text",
            "content": {"format": "text", "value": _SAFE_TEXT},
        }

    def test_success_saves_zero_usage(self, db: Session, job: Job) -> None:
        llm_call_id = self._call(db, job)
        llm_call = db.exec(select(LlmCall).where(LlmCall.id == llm_call_id)).first()
        assert llm_call.usage == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def test_create_llm_call_error_returns_none(self, db: Session, job: Job) -> None:
        with patch(
            "app.crud.llm.create_llm_call",
            side_effect=Exception("DB insert failed"),
        ):
            result = self._call(db, job)
        assert result is None

    def test_update_llm_call_response_error_returns_none(
        self, db: Session, job: Job
    ) -> None:
        with patch(
            "app.crud.llm.update_llm_call_response",
            side_effect=Exception("DB update failed"),
        ):
            result = self._call(db, job)
        assert result is None

    def test_chain_id_forwarded_to_create_llm_call(self, db: Session, job: Job) -> None:
        chain_id = uuid4()
        with patch("app.crud.llm.create_llm_call") as mock_create:
            mock_create.return_value = MagicMock(id=uuid4())
            with patch("app.crud.llm.update_llm_call_response"):
                self._call(db, job, chain_id=chain_id)
        _, kwargs = mock_create.call_args
        assert kwargs["chain_id"] == chain_id

    def test_request_metadata_forwarded_to_llm_call_request(
        self, db: Session, job: Job
    ) -> None:
        metadata = {"request_id": "abc", "user": "test"}
        with patch("app.crud.llm.create_llm_call") as mock_create:
            mock_create.return_value = MagicMock(id=uuid4())
            with patch("app.crud.llm.update_llm_call_response"):
                self._call(db, job, request_metadata=metadata)
        _, kwargs = mock_create.call_args
        assert kwargs["request"].request_metadata == metadata
