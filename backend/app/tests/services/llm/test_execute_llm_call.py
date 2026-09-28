"""Direct tests for `execute_llm_call`, which `execute_job` (test_jobs.py) cannot
reach: `record_call` is keyword-only on the function and `guardrail_outcome` /
`llm_call_id` live on `BlockResult`, not on the job callback payload."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from sqlmodel import Session, func, select

from app.models.llm import (
    LLMCallResponse,
    LLMResponse,
    QueryParams,
    TextContent,
    TextOutput,
    Usage,
)
from app.models.llm.request import ConfigBlob, LLMCallConfig, LlmCall
from app.services.llm.chain.types import BlockResult
from app.services.llm.jobs import execute_llm_call
from app.tests.utils.utils import get_project

VALIDATOR_CONFIG_ID = "00000000-0000-0000-0000-000000000001"
PROXY_URL = "https://api.tap.example/v1/predictions"

TEXT_COMPLETION = {
    "type": "text",
    "provider": "openai-native",
    "params": {"model": "gpt-4o"},
}
PROXY_COMPLETION = {
    "type": "proxy",
    "provider": None,
    "params": {"client_llm_url": PROXY_URL},
}
# Kaapi-shaped (not `openai-native`), so it goes through
# `transform_kaapi_config_to_native`; the KB makes the mapper emit a file_search tool.
KAAPI_KB_COMPLETION = {
    "type": "text",
    "provider": "openai",
    "params": {"model": "gpt-4o", "knowledge_base_ids": ["vs_abc123"]},
}
PROXY_PAYLOAD = {
    "id": "resp_abc",
    "model": "gpt-5",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Proxy answer."}],
        }
    ],
    "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
}


def build_blob(
    completion: dict[str, Any],
    *,
    input_guardrails: bool = False,
    output_guardrails: bool = False,
) -> ConfigBlob:
    return ConfigBlob.model_validate(
        {
            "completion": completion,
            "input_guardrails": [{"validator_config_id": VALIDATOR_CONFIG_ID}]
            if input_guardrails
            else [],
            "output_guardrails": [{"validator_config_id": VALIDATOR_CONFIG_ID}]
            if output_guardrails
            else [],
        }
    )


def llm_call_count(db: Session) -> int:
    return db.exec(select(func.count()).select_from(LlmCall)).one()


@contextmanager
def guardrails_http(*verdicts: dict[str, Any] | Exception) -> Iterator[MagicMock]:
    """Patch the guardrails HTTP boundary: every GET resolves one validator
    config, and POSTs return `verdicts` in call order."""
    config_response = MagicMock()
    config_response.raise_for_status.return_value = None
    config_response.json.return_value = {
        "success": True,
        "data": [{"type": "uli_slur_match"}],
    }

    responses = []
    for verdict in verdicts:
        response = MagicMock()
        if isinstance(verdict, Exception):
            response.raise_for_status.side_effect = verdict
        else:
            response.raise_for_status.return_value = None
            response.json.return_value = verdict
        responses.append(response)

    client = MagicMock()
    client.get.return_value = config_response
    client.post.side_effect = responses

    with patch("app.services.llm.guardrails.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = client
        yield client


def http_status_error(status_code: int) -> httpx.HTTPStatusError:
    response = MagicMock()
    response.status_code = status_code
    return httpx.HTTPStatusError("rejected", request=MagicMock(), response=response)


@pytest.fixture
def provider(db: Session):
    with (
        patch("app.services.llm.jobs.Session") as session_cls,
        patch("app.services.llm.jobs.get_llm_provider") as get_provider,
    ):
        session_cls.return_value.__enter__.return_value = db
        session_cls.return_value.__exit__.return_value = None
        instance = MagicMock()
        get_provider.return_value = instance
        yield instance


@pytest.fixture
def metrics():
    with (
        patch("app.services.llm.jobs.record_llm_call_started") as started,
        patch("app.services.llm.jobs.record_llm_call_finished") as finished,
    ):
        yield started, finished


@pytest.fixture
def provider_response() -> LLMCallResponse:
    return LLMCallResponse(
        response=LLMResponse(
            provider_response_id="resp-123",
            conversation_id=None,
            model="gpt-4o",
            provider="openai",
            output=TextOutput(content=TextContent(value="Provider answer.")),
        ),
        usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        provider_raw_response=None,
    )


@pytest.fixture
def proxy_http():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = PROXY_PAYLOAD

    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with (
        patch(
            "app.services.llm.jobs.get_provider_credential",
            return_value={"api_key": "tap-token"},
        ),
        patch("app.services.llm.jobs.httpx.Client", return_value=client),
    ):
        yield client


def call(
    db: Session,
    blob: ConfigBlob,
    *,
    text: str = "what are my land rights",
    **kw: Any,
) -> BlockResult:
    project = get_project(db)
    return execute_llm_call(
        config=LLMCallConfig(blob=blob),
        query=QueryParams(input=text),
        job_id=uuid4(),
        project_id=project.id,
        organization_id=project.organization_id,
        request_metadata=None,
        langfuse_credentials=None,
        **kw,
    )


class TestRecordCallFalse:
    def test_provider_path_writes_no_llm_call_and_no_metrics(
        self, db: Session, provider, metrics, provider_response
    ):
        started, finished = metrics
        provider.execute.return_value = (provider_response, None)
        before = llm_call_count(db)

        result = call(db, build_blob(TEXT_COMPLETION), record_call=False)

        assert result.error is None
        assert result.response.response.output.content.value == "Provider answer."
        assert result.usage.total_tokens == 30
        assert result.llm_call_id is None
        assert llm_call_count(db) == before
        started.assert_not_called()
        finished.assert_not_called()

    def test_proxy_path_writes_no_llm_call_and_no_metrics(
        self, db: Session, provider, metrics, proxy_http
    ):
        started, finished = metrics
        before = llm_call_count(db)

        with patch(
            "app.services.llm.jobs.update_llm_call_response"
        ) as update_llm_call_response:
            result = call(db, build_blob(PROXY_COMPLETION), record_call=False)

        assert result.error is None
        assert result.response.response.output.content.value == "Proxy answer."
        assert result.usage.total_tokens == 18
        assert result.llm_call_id is None
        assert llm_call_count(db) == before
        # Without the `if llm_call_id:` guard this runs with a None id, and the
        # surrounding try/except swallows the failure — so the mock is the only
        # place the guard is observable.
        update_llm_call_response.assert_not_called()
        started.assert_not_called()
        finished.assert_not_called()

    def test_rephrase_path_writes_no_llm_call(self, db: Session, provider):
        rephrase_text = "Please rephrase without unsafe content."
        before = llm_call_count(db)
        with guardrails_http(
            {
                "success": True,
                "bypassed": False,
                "data": {"safe_text": rephrase_text, "rephrase_needed": True},
            }
        ):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, input_guardrails=True),
                record_call=False,
            )

        assert result.error is None
        assert result.response.response.output.content.value == rephrase_text
        assert result.guardrail_outcome == "rephrased"
        assert result.llm_call_id is None
        assert llm_call_count(db) == before
        provider.execute.assert_not_called()


class TestRecordCallDefault:
    """`record_call` defaults to True — the audit row and the metrics must still
    fire, otherwise the eval flag has leaked into production traffic."""

    @pytest.fixture
    def llm_call_crud(self):
        with (
            patch("app.services.llm.jobs.create_llm_call") as create_llm_call,
            patch("app.services.llm.jobs.update_llm_call_response"),
        ):
            create_llm_call.return_value = MagicMock(id=uuid4())
            yield create_llm_call

    def test_provider_path_records_call_and_metrics(
        self, db: Session, provider, metrics, provider_response, llm_call_crud
    ):
        started, finished = metrics
        provider.execute.return_value = (provider_response, None)

        result = call(db, build_blob(TEXT_COMPLETION))

        assert result.error is None
        assert result.llm_call_id == llm_call_crud.return_value.id
        started.assert_called_once()
        finished.assert_called_once()

    def test_proxy_path_records_call_and_metrics(
        self, db: Session, provider, metrics, proxy_http, llm_call_crud
    ):
        started, finished = metrics

        result = call(db, build_blob(PROXY_COMPLETION))

        assert result.error is None
        assert result.llm_call_id == llm_call_crud.return_value.id
        started.assert_called_once()
        finished.assert_called_once()


class TestGuardrailOutcome:
    def test_input_hard_block(self, db: Session, provider):
        with guardrails_http({"success": False, "error": "Unsafe content detected"}):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, input_guardrails=True),
                record_call=False,
            )

        assert result.error == "Unsafe content detected"
        assert result.guardrail_outcome == "blocked"
        provider.execute.assert_not_called()

    def test_input_stripped_to_empty_is_a_block(self, db: Session, provider):
        with guardrails_http(
            {
                "success": True,
                "bypassed": False,
                "data": {"safe_text": "   ", "rephrase_needed": False},
            }
        ):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, input_guardrails=True),
                record_call=False,
            )

        assert (
            result.error
            == "Input guardrails rejected the request and left no usable content."
        )
        assert result.guardrail_outcome == "blocked"
        provider.execute.assert_not_called()

    def test_output_hard_block_on_provider_path_keeps_usage(
        self, db: Session, provider, provider_response
    ):
        provider.execute.return_value = (provider_response, None)
        with guardrails_http({"success": False, "error": "Output blocked"}):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, output_guardrails=True),
                record_call=False,
            )

        assert result.error == "Output blocked"
        assert result.guardrail_outcome == "blocked"
        assert result.usage.total_tokens == 30

    def test_output_hard_block_on_proxy_path_keeps_usage(
        self, db: Session, provider, proxy_http
    ):
        with (
            patch(
                "app.services.llm.guardrails.list_validators_config",
                return_value=([], [{"type": "pii_remover"}]),
            ),
            patch(
                "app.services.llm.guardrails.run_guardrails_validation",
                return_value={
                    "success": False,
                    "bypassed": False,
                    "error": "Output blocked",
                },
            ),
        ):
            result = call(
                db,
                build_blob(PROXY_COMPLETION, output_guardrails=True),
                record_call=False,
            )

        assert result.error == "Output blocked"
        assert result.guardrail_outcome == "blocked"
        assert result.usage.total_tokens == 18

    def test_provider_error_has_no_guardrail_outcome(self, db: Session, provider):
        provider.execute.return_value = (None, "API rate limit exceeded")

        result = call(db, build_blob(TEXT_COMPLETION), record_call=False)

        assert result.error == "API rate limit exceeded"
        assert result.guardrail_outcome is None

    @pytest.mark.parametrize("status_code", [401, 403, 422])
    def test_guardrails_auth_failure_has_no_guardrail_outcome(
        self, db: Session, provider, status_code: int
    ):
        with guardrails_http(http_status_error(status_code)):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, input_guardrails=True),
                record_call=False,
            )

        assert result.error == (
            f"Guardrails service rejected the request (HTTP {status_code})"
        )
        assert result.guardrail_outcome is None
        provider.execute.assert_not_called()


class TestRetryableFlag:
    """`BlockResult.retryable` is what `retry_llm_call` keys off, and `error` alone
    cannot tell a rate limit from a revoked key. These pin the flag where it is set,
    not where it is read.
    """

    def test_provider_failure_is_retryable(self, db: Session, provider):
        provider.execute.return_value = (None, "API rate limit exceeded")

        result = call(db, build_blob(TEXT_COMPLETION), record_call=False)

        assert result.retryable is True

    def test_proxy_transport_failure_is_retryable(
        self, db: Session, provider, proxy_http
    ):
        proxy_http.post.side_effect = httpx.ConnectError("connection refused")

        result = call(db, build_blob(PROXY_COMPLETION), record_call=False)

        assert result.error.startswith("Proxy call failed:")
        assert result.retryable is True

    def test_guardrail_block_is_not_retryable(self, db: Session, provider):
        with guardrails_http({"success": False, "error": "Unsafe content detected"}):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, input_guardrails=True),
                record_call=False,
            )

        assert result.guardrail_outcome == "blocked"
        assert result.retryable is False

    @pytest.mark.parametrize("status_code", [401, 403, 422])
    def test_input_guardrails_auth_failure_is_not_retryable(
        self, db: Session, provider, status_code: int
    ):
        with guardrails_http(http_status_error(status_code)):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, input_guardrails=True),
                record_call=False,
            )

        assert result.retryable is False
        provider.execute.assert_not_called()

    def test_output_guardrails_auth_failure_is_not_retryable(
        self, db: Session, provider, provider_response
    ):
        """The expensive direction: the completion is generated *before* output
        guardrails run, so each retry would re-charge the provider for a failure
        that broken credentials guarantee will repeat."""
        provider.execute.return_value = (provider_response, None)
        with guardrails_http(http_status_error(401)):
            result = call(
                db,
                build_blob(TEXT_COMPLETION, output_guardrails=True),
                record_call=False,
            )

        assert result.error == "Guardrails service rejected the request (HTTP 401)"
        assert result.guardrail_outcome is None
        assert result.retryable is False
        assert result.usage.total_tokens == 30

    def test_unresolvable_stored_config_is_not_retryable(self, db: Session, provider):
        project = get_project(db)

        result = execute_llm_call(
            config=LLMCallConfig(id=uuid4(), version=1),
            query=QueryParams(input="what are my land rights"),
            job_id=uuid4(),
            project_id=project.id,
            organization_id=project.organization_id,
            request_metadata=None,
            langfuse_credentials=None,
            record_call=False,
        )

        assert result.error is not None
        assert result.retryable is False
        provider.execute.assert_not_called()


class TestFileSearchResultsAreOptIn:
    """`include=["file_search_call.results"]` makes OpenAI return the retrieved chunk
    text, which the `knowledge_base` metric scores. It is pure extra payload for a
    caller that didn't ask for the raw provider response, hence the opt-in.
    """

    @staticmethod
    def _params_sent_to_provider(provider) -> dict[str, Any]:
        completion_config = provider.execute.call_args.args[0]
        return completion_config.params

    def test_raw_response_requested_asks_for_the_hits(
        self, db: Session, provider, provider_response
    ):
        provider.execute.return_value = (provider_response, None)

        call(
            db,
            build_blob(KAAPI_KB_COMPLETION),
            record_call=False,
            include_provider_raw_response=True,
        )

        params = self._params_sent_to_provider(provider)
        assert params["tools"][0]["type"] == "file_search"
        assert params["include"] == ["file_search_call.results"]

    def test_default_call_does_not_ask_for_the_hits(
        self, db: Session, provider, provider_response
    ):
        provider.execute.return_value = (provider_response, None)

        call(db, build_blob(KAAPI_KB_COMPLETION), record_call=False)

        params = self._params_sent_to_provider(provider)
        assert params["tools"][0]["type"] == "file_search"
        assert "include" not in params
