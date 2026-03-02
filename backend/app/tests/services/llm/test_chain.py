from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest

from app.models.llm.request import (
    LLMCallConfig,
    ConfigBlob,
    NativeCompletionConfig,
    QueryParams,
    TextInput,
    TextContent,
    AudioInput,
)
from app.models.llm.response import (
    LLMCallResponse,
    LLMResponse,
    Usage,
    TextOutput,
    TextContent as ResponseTextContent,
    AudioOutput,
    AudioContent,
)
from app.services.llm.chain.chain import (
    ChainBlock,
    ChainContext,
    LLMChain,
    result_to_query,
)
from app.services.llm.chain.types import BlockResult


@pytest.fixture
def context():
    return ChainContext(
        job_id=uuid4(),
        chain_id=uuid4(),
        project_id=1,
        organization_id=1,
        callback_url="https://example.com/callback",
        total_blocks=3,
        intermediate_callback_flags=[True, True, False],
    )


@pytest.fixture
def text_response():
    return LLMCallResponse(
        response=LLMResponse(
            provider_response_id="resp-1",
            conversation_id=None,
            model="gpt-4",
            provider="openai",
            output=TextOutput(content=ResponseTextContent(value="Hello world")),
        ),
        usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        provider_raw_response=None,
    )


@pytest.fixture
def audio_response():
    return LLMCallResponse(
        response=LLMResponse(
            provider_response_id="resp-2",
            conversation_id=None,
            model="gemini",
            provider="google",
            output=AudioOutput(
                content=AudioContent(
                    format="base64",
                    value="audio-data-base64",
                    mime_type="audio/wav",
                )
            ),
        ),
        usage=Usage(input_tokens=5, output_tokens=15, total_tokens=20),
        provider_raw_response=None,
    )


def make_config():
    return LLMCallConfig(
        blob=ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                type="text",
                params={"model": "gpt-4"},
            )
        )
    )


class TestResultToQuery:
    def test_text_output_to_query(self, text_response):
        result = BlockResult(response=text_response, usage=text_response.usage)

        query = result_to_query(result)

        assert isinstance(query.input, TextInput)
        assert query.input.content.value == "Hello world"

    def test_audio_output_to_query(self, audio_response):
        result = BlockResult(response=audio_response, usage=audio_response.usage)

        query = result_to_query(result)

        assert isinstance(query.input, AudioInput)
        assert query.input.content.value == "audio-data-base64"

    def test_unsupported_output_type_raises(self):
        mock_response = MagicMock()
        mock_response.response.output.type = "unknown"
        mock_response.response.output.__class__ = type("Unknown", (), {})
        result = BlockResult(response=mock_response, usage=MagicMock())

        with pytest.raises(ValueError, match="Cannot chain output type"):
            result_to_query(result)


class TestChainContext:
    def test_aggregates_usage(self, context):
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        result = BlockResult(
            response=MagicMock(), llm_call_id=uuid4(), usage=usage, error=None
        )

        with patch("app.services.llm.chain.chain.Session"):
            context.on_block_completed(0, result)

        assert context.aggregated_usage.input_tokens == 10
        assert context.aggregated_usage.output_tokens == 20
        assert context.aggregated_usage.total_tokens == 30

    def test_aggregates_usage_across_blocks(self, context):
        usage1 = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        usage2 = Usage(input_tokens=5, output_tokens=15, total_tokens=20)

        result1 = BlockResult(
            response=MagicMock(), llm_call_id=uuid4(), usage=usage1, error=None
        )
        result2 = BlockResult(
            response=MagicMock(), llm_call_id=uuid4(), usage=usage2, error=None
        )

        with patch("app.services.llm.chain.chain.Session"):
            context.on_block_completed(0, result1)
            context.on_block_completed(1, result2)

        assert context.aggregated_usage.input_tokens == 15
        assert context.aggregated_usage.total_tokens == 50

    def test_updates_db_on_success(self, context):
        llm_call_id = uuid4()
        result = BlockResult(
            response=MagicMock(), llm_call_id=llm_call_id, usage=MagicMock(), error=None
        )

        with patch("app.services.llm.chain.chain.Session") as mock_session, patch(
            "app.services.llm.chain.chain.update_llm_chain_block_completed"
        ) as mock_update:
            mock_session.return_value.__enter__.return_value = MagicMock()
            context.on_block_completed(0, result)

            mock_update.assert_called_once_with(
                mock_session.return_value.__enter__.return_value,
                chain_id=context.chain_id,
                llm_call_id=llm_call_id,
            )

    def test_sends_intermediate_callback(self, context, text_response):
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.chain.Session") as mock_session,
            patch("app.services.llm.chain.chain.update_llm_chain_block_completed"),
            patch("app.services.llm.chain.chain.send_callback") as mock_callback,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            context.on_block_completed(0, result)

            mock_callback.assert_called_once()
            call_kwargs = mock_callback.call_args[1]
            assert call_kwargs["callback_url"] == "https://example.com/callback"

    def test_skips_intermediate_callback_for_last_block(self, context, text_response):
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.chain.Session") as mock_session,
            patch("app.services.llm.chain.chain.update_llm_chain_block_completed"),
            patch("app.services.llm.chain.chain.send_callback") as mock_callback,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            # Block index 2 = last block (total_blocks=3)
            context.on_block_completed(2, result)

            mock_callback.assert_not_called()

    def test_skips_intermediate_callback_when_flag_false(self, context, text_response):
        context.intermediate_callback_flags = [False, True, False]
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.chain.Session") as mock_session,
            patch("app.services.llm.chain.chain.update_llm_chain_block_completed"),
            patch("app.services.llm.chain.chain.send_callback") as mock_callback,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            context.on_block_completed(0, result)

            mock_callback.assert_not_called()

    def test_skips_db_update_on_error(self, context):
        result = BlockResult(error="Block failed", usage=MagicMock())

        with patch(
            "app.services.llm.chain.chain.update_llm_chain_block_completed"
        ) as mock_update:
            context.on_block_completed(0, result)
            mock_update.assert_not_called()

    def test_intermediate_callback_exception_is_swallowed(self, context, text_response):
        result = BlockResult(
            response=text_response,
            llm_call_id=uuid4(),
            usage=text_response.usage,
            error=None,
        )

        with (
            patch("app.services.llm.chain.chain.Session") as mock_session,
            patch("app.services.llm.chain.chain.update_llm_chain_block_completed"),
            patch(
                "app.services.llm.chain.chain.send_callback",
                side_effect=Exception("Connection error"),
            ),
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            # Should not raise
            context.on_block_completed(0, result)


class TestChainBlock:
    def test_execute_single_block(self, context, text_response):
        query = QueryParams(input="test input")
        config = make_config()
        block = ChainBlock(config=config, index=0, context=context)

        with patch(
            "app.services.llm.chain.chain.execute_llm_call"
        ) as mock_execute, patch.object(context, "on_block_completed"):
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            result = block.execute(query)

            assert result.success
            mock_execute.assert_called_once()

    def test_execute_chains_to_next_block(self, context, text_response):
        query = QueryParams(input="test input")
        config = make_config()
        block1 = ChainBlock(config=config, index=0, context=context)
        block2 = ChainBlock(config=config, index=1, context=context)
        block1.link(block2)

        with patch(
            "app.services.llm.chain.chain.execute_llm_call"
        ) as mock_execute, patch.object(context, "on_block_completed"):
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            result = block1.execute(query)

            assert mock_execute.call_count == 2

    def test_execute_stops_on_failure(self, context):
        query = QueryParams(input="test input")
        config = make_config()
        block1 = ChainBlock(config=config, index=0, context=context)
        block2 = ChainBlock(config=config, index=1, context=context)
        block1.link(block2)

        with patch(
            "app.services.llm.chain.chain.execute_llm_call"
        ) as mock_execute, patch.object(context, "on_block_completed"):
            mock_execute.return_value = BlockResult(error="Provider error")

            result = block1.execute(query)

            assert not result.success
            assert result.error == "Provider error"
            mock_execute.assert_called_once()


class TestLLMChain:
    def test_execute_empty_chain(self):
        chain = LLMChain([])
        query = QueryParams(input="test")

        result = chain.execute(query)

        assert not result.success
        assert result.error == "Chain has no blocks"

    def test_execute_single_block_chain(self, context, text_response):
        config = make_config()
        block = ChainBlock(config=config, index=0, context=context)
        chain = LLMChain([block])

        with patch(
            "app.services.llm.chain.chain.execute_llm_call"
        ) as mock_execute, patch.object(context, "on_block_completed"):
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            result = chain.execute(QueryParams(input="hello"))

            assert result.success
            mock_execute.assert_called_once()

    def test_execute_multi_block_chain(self, context, text_response):
        config = make_config()
        blocks = [ChainBlock(config=config, index=i, context=context) for i in range(3)]
        chain = LLMChain(blocks)

        with patch(
            "app.services.llm.chain.chain.execute_llm_call"
        ) as mock_execute, patch.object(context, "on_block_completed"):
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            result = chain.execute(QueryParams(input="hello"))

            assert result.success
            assert mock_execute.call_count == 3
