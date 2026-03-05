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


class TestChainBlock:
    def test_execute_single_block(self, context, text_response):
        query = QueryParams(input="test input")
        config = make_config()
        block = ChainBlock(config=config, index=0, context=context)

        with patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute:
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            result = block.execute(query)

            assert result.success
            mock_execute.assert_called_once()

    def test_execute_returns_failure(self, context):
        query = QueryParams(input="test input")
        config = make_config()
        block = ChainBlock(config=config, index=0, context=context)

        with patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute:
            mock_execute.return_value = BlockResult(error="Provider error")

            result = block.execute(query)

            assert not result.success
            assert result.error == "Provider error"
            mock_execute.assert_called_once()


class TestLLMChain:
    def test_execute_empty_chain(self, context):
        chain = LLMChain([], context)
        query = QueryParams(input="test")

        result = chain.execute(query)

        assert not result.success
        assert result.error == "Chain has no blocks"

    def test_execute_single_block_chain(self, context, text_response):
        config = make_config()
        block = ChainBlock(config=config, index=0, context=context)
        chain = LLMChain([block], context)

        with patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute:
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            result = chain.execute(QueryParams(input="hello"))

            assert result.success
            mock_execute.assert_called_once()

    def test_execute_multi_block_chain(self, context, text_response):
        config = make_config()
        blocks = [ChainBlock(config=config, index=i, context=context) for i in range(3)]
        chain = LLMChain(blocks, context)

        with patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute:
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            result = chain.execute(QueryParams(input="hello"))

            assert result.success
            assert mock_execute.call_count == 3

    def test_execute_stops_on_failure(self, context, text_response):
        config = make_config()
        blocks = [ChainBlock(config=config, index=i, context=context) for i in range(3)]
        chain = LLMChain(blocks, context)

        with patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute:
            mock_execute.return_value = BlockResult(error="Provider error")

            result = chain.execute(QueryParams(input="hello"))

            assert not result.success
            assert result.error == "Provider error"
            mock_execute.assert_called_once()

    def test_execute_calls_on_block_completed(self, context, text_response):
        config = make_config()
        blocks = [ChainBlock(config=config, index=i, context=context) for i in range(2)]
        chain = LLMChain(blocks, context)
        callback = MagicMock()

        with patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute:
            mock_execute.return_value = BlockResult(
                response=text_response, usage=text_response.usage
            )

            chain.execute(QueryParams(input="hello"), on_block_completed=callback)

            assert callback.call_count == 2
