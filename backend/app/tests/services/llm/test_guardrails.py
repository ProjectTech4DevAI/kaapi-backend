import uuid
from unittest.mock import MagicMock, patch

import httpx

from app.core.config import settings
from app.models.llm.request import Validator
from app.services.llm.guardrails import (
    list_validators_config,
    run_guardrails_validation,
)


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
    assert kwargs["json"]["project_id"] == TEST_PROJECT_ID
    assert kwargs["json"]["organization_id"] == TEST_ORGANIZATION_ID
    assert kwargs["params"]["suppress_pass_logs"] == "true"
    assert kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert kwargs["headers"]["Content-Type"] == "application/json"


@patch("app.services.llm.guardrails.httpx.Client")
def test_run_guardrails_validation_http_error_bypasses(mock_client_cls) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad", request=None, response=None
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
def test_list_validators_config_omits_none_query_params(mock_client_cls) -> None:
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
