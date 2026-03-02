from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest

from app.models.llm.request import (
    LLMChainRequest,
    LLMCallConfig,
    ConfigBlob,
    NativeCompletionConfig,
    QueryParams,
    ChainStatus,
)
from app.models.llm.request import ChainBlock as ChainBlockModel
from app.models.llm.response import (
    LLMCallResponse,
    LLMResponse,
    Usage,
    TextOutput,
    TextContent,
)
from app.models import JobStatus
from app.services.llm.chain.chain import ChainBlock, ChainContext, LLMChain
from app.services.llm.chain.executor import ChainExecutor
from app.services.llm.chain.types import BlockResult


@pytest.fixture
def context():
    return ChainContext(
        job_id=uuid4(),
        chain_id=uuid4(),
        project_id=1,
        organization_id=1,
        callback_url="https://example.com/callback",
        total_blocks=1,
    )


@pytest.fixture
def request_obj():
    return LLMChainRequest(
        query=QueryParams(input="hello"),
        blocks=[
            ChainBlockModel(
                config=LLMCallConfig(
                    blob=ConfigBlob(
                        completion=NativeCompletionConfig(
                            provider="openai-native",
                            type="text",
                            params={"model": "gpt-4"},
                        )
                    )
                )
            )
        ],
        callback_url="https://example.com/callback",
    )


@pytest.fixture
def text_response():
    return LLMCallResponse(
        response=LLMResponse(
            provider_response_id="resp-1",
            conversation_id=None,
            model="gpt-4",
            provider="openai",
            output=TextOutput(content=TextContent(value="Response text")),
        ),
        usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        provider_raw_response=None,
    )


@pytest.fixture
def success_result(text_response):
    return BlockResult(
        response=text_response,
        llm_call_id=uuid4(),
        usage=text_response.usage,
    )


@pytest.fixture
def failure_result():
    return BlockResult(error="Provider failed")


class TestChainExecutor:
    def _make_executor(self, context, request_obj, chain_result):
        mock_chain = MagicMock(spec=LLMChain)
        mock_chain.execute.return_value = chain_result
        return ChainExecutor(chain=mock_chain, context=context, request=request_obj)

    def test_run_success_with_callback(self, context, request_obj, success_result):
        executor = self._make_executor(context, request_obj, success_result)

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.send_callback") as mock_callback,
            patch(
                "app.services.llm.chain.executor.update_llm_chain_status"
            ) as mock_chain_status,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()

            result = executor.run()

            assert result["success"] is True
            mock_callback.assert_called_once()
            # Verify chain status updated to COMPLETED
            completed_call = [
                c
                for c in mock_chain_status.call_args_list
                if c[1].get("status") == ChainStatus.COMPLETED
            ]
            assert len(completed_call) == 1

    def test_run_success_without_callback(self, context, request_obj, success_result):
        request_obj.callback_url = None
        context.callback_url = None
        executor = self._make_executor(context, request_obj, success_result)

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.send_callback") as mock_callback,
            patch("app.services.llm.chain.executor.update_llm_chain_status"),
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()

            result = executor.run()

            assert result["success"] is True
            mock_callback.assert_not_called()

    def test_run_failure_updates_status(self, context, request_obj, failure_result):
        executor = self._make_executor(context, request_obj, failure_result)

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.send_callback"),
            patch(
                "app.services.llm.chain.executor.update_llm_chain_status"
            ) as mock_chain_status,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()

            result = executor.run()

            assert result["success"] is False
            assert result["error"] == "Provider failed"
            # Verify chain status updated to FAILED
            failed_call = [
                c
                for c in mock_chain_status.call_args_list
                if c[1].get("status") == ChainStatus.FAILED
            ]
            assert len(failed_call) == 1

    def test_run_failure_sends_callback(self, context, request_obj, failure_result):
        executor = self._make_executor(context, request_obj, failure_result)

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.send_callback") as mock_callback,
            patch("app.services.llm.chain.executor.update_llm_chain_status"),
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()

            result = executor.run()

            mock_callback.assert_called_once()

    def test_run_unexpected_exception_handled(self, context, request_obj):
        mock_chain = MagicMock(spec=LLMChain)
        mock_chain.execute.side_effect = RuntimeError("Something broke")
        executor = ChainExecutor(chain=mock_chain, context=context, request=request_obj)

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.send_callback"),
            patch("app.services.llm.chain.executor.update_llm_chain_status"),
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()

            result = executor.run()

            assert result["success"] is False
            assert "Unexpected error occurred" in result["error"]

    def test_setup_updates_job_and_chain_status(
        self, context, request_obj, success_result
    ):
        executor = self._make_executor(context, request_obj, success_result)

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.send_callback"),
            patch(
                "app.services.llm.chain.executor.update_llm_chain_status"
            ) as mock_chain_status,
            patch("app.services.llm.chain.executor.JobCrud") as mock_job_crud,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()

            executor.run()

            # _setup should set chain to RUNNING
            running_calls = [
                c
                for c in mock_chain_status.call_args_list
                if c[1].get("status") == ChainStatus.RUNNING
            ]
            assert len(running_calls) == 1
