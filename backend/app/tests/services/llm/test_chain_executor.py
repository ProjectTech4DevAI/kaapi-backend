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


class TestOnBlockCompleted:
    def _make_executor(self, context, request_obj):
        mock_chain = MagicMock(spec=LLMChain)
        return ChainExecutor(chain=mock_chain, context=context, request=request_obj)

    def test_aggregates_usage(self, context, request_obj):
        executor = self._make_executor(context, request_obj)
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        result = BlockResult(
            response=MagicMock(), llm_call_id=uuid4(), usage=usage, error=None
        )

        with patch("app.services.llm.chain.executor.Session"):
            executor._on_block_completed(0, result)

        assert context.aggregated_usage.input_tokens == 10
        assert context.aggregated_usage.output_tokens == 20
        assert context.aggregated_usage.total_tokens == 30

    def test_aggregates_usage_across_blocks(self, context, request_obj):
        executor = self._make_executor(context, request_obj)
        result1 = BlockResult(
            response=MagicMock(),
            llm_call_id=uuid4(),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            error=None,
        )
        result2 = BlockResult(
            response=MagicMock(),
            llm_call_id=uuid4(),
            usage=Usage(input_tokens=5, output_tokens=15, total_tokens=20),
            error=None,
        )

        with patch("app.services.llm.chain.executor.Session"):
            executor._on_block_completed(0, result1)
            executor._on_block_completed(1, result2)

        assert context.aggregated_usage.input_tokens == 15
        assert context.aggregated_usage.total_tokens == 50

    def test_updates_db_on_success(self, context, request_obj):
        executor = self._make_executor(context, request_obj)
        llm_call_id = uuid4()
        result = BlockResult(
            response=MagicMock(), llm_call_id=llm_call_id, usage=MagicMock(), error=None
        )

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch(
                "app.services.llm.chain.executor.update_llm_chain_block_completed"
            ) as mock_update,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            executor._on_block_completed(0, result)

            mock_update.assert_called_once_with(
                mock_session.return_value.__enter__.return_value,
                chain_id=context.chain_id,
                llm_call_id=llm_call_id,
            )

    def test_skips_db_update_on_error(self, context, request_obj):
        executor = self._make_executor(context, request_obj)
        result = BlockResult(error="Block failed", usage=MagicMock())

        with patch(
            "app.services.llm.chain.executor.update_llm_chain_block_completed"
        ) as mock_update:
            executor._on_block_completed(0, result)
            mock_update.assert_not_called()

    def test_sends_intermediate_callback(self, context, request_obj, text_response):
        context.total_blocks = 3
        context.intermediate_callback_flags = [True, True, False]
        executor = self._make_executor(context, request_obj)
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.update_llm_chain_block_completed"),
            patch("app.services.llm.chain.executor.send_callback") as mock_callback,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            executor._on_block_completed(0, result)

            mock_callback.assert_called_once()

    def test_skips_intermediate_callback_for_last_block(
        self, context, request_obj, text_response
    ):
        context.total_blocks = 3
        context.intermediate_callback_flags = [True, True, False]
        executor = self._make_executor(context, request_obj)
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.update_llm_chain_block_completed"),
            patch("app.services.llm.chain.executor.send_callback") as mock_callback,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            executor._on_block_completed(2, result)

            mock_callback.assert_not_called()

    def test_skips_intermediate_callback_when_flag_false(
        self, context, request_obj, text_response
    ):
        context.total_blocks = 3
        context.intermediate_callback_flags = [False, True, False]
        executor = self._make_executor(context, request_obj)
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.update_llm_chain_block_completed"),
            patch("app.services.llm.chain.executor.send_callback") as mock_callback,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            executor._on_block_completed(0, result)

            mock_callback.assert_not_called()

    def test_intermediate_callback_exception_is_swallowed(
        self, context, request_obj, text_response
    ):
        context.total_blocks = 3
        context.intermediate_callback_flags = [True, True, False]
        executor = self._make_executor(context, request_obj)
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.executor.Session") as mock_session,
            patch("app.services.llm.chain.executor.update_llm_chain_block_completed"),
            patch(
                "app.services.llm.chain.executor.send_callback",
                side_effect=Exception("Connection error"),
            ),
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            # Should not raise
            executor._on_block_completed(0, result)
