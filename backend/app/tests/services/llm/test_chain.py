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
from app.services.llm.jobs import (
    DETECTED_LANGUAGE_FALLBACK,
    _substitute_detected_language_marker,
)


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


# ---------- S2S detected-language propagation ----------


def _stt_text_response(language_code: str | None) -> LLMCallResponse:
    """Build a TextOutput-style STT response with the given detected language_code."""
    return LLMCallResponse(
        response=LLMResponse(
            provider_response_id="stt-1",
            conversation_id=None,
            model="saaras:v3",
            provider="sarvamai-native",
            output=TextOutput(
                content=ResponseTextContent(value="नमस्ते", language_code=language_code)
            ),
        ),
        usage=Usage(input_tokens=0, output_tokens=5, total_tokens=5),
        provider_raw_response=None,
    )


class TestDetectedLanguagePropagation:
    """Covers the STT → ChainContext.detected_language → TTS path used by /llm/chain/sts."""

    def test_detected_language_persists_when_stt_yields_language_code(self, context):
        result = BlockResult(response=_stt_text_response("hi-IN"))
        result_to_query(result, context)
        assert context.detected_language == "hi-IN"

    def test_stt_unknown_sentinel_does_not_populate_detected_language(self, context):
        """Sarvam STT signals failed detection with language_code='unknown'.
        Storing that on ChainContext would forward 'unknown' to TTS and defeat
        the en-IN fallback inside execute_llm_call. detected_language must stay
        None so the fallback fires."""
        result = BlockResult(response=_stt_text_response("unknown"))
        result_to_query(result, context)
        assert context.detected_language is None

    def test_detected_language_stays_none_when_stt_yields_no_language(self, context):
        """When STT can't detect, downstream TTS gets detected_language=None and falls
        back to en-IN inside execute_llm_call (see jobs.py {{detected}} handler)."""
        result = BlockResult(response=_stt_text_response(None))
        result_to_query(result, context)
        assert context.detected_language is None

    def test_chain_block_forwards_detected_language_to_execute_llm_call(
        self, context, audio_response
    ):
        """ChainBlock must pass ChainContext.detected_language down so jobs.py can substitute
        the {{detected}} marker in TTS configs."""
        context.detected_language = "ta-IN"
        block = ChainBlock(config=make_config(), index=2, context=context)

        with patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute:
            mock_execute.return_value = BlockResult(
                response=audio_response, usage=audio_response.usage
            )
            block.execute(QueryParams(input="hello"))

        kwargs = mock_execute.call_args.kwargs
        assert kwargs["detected_language"] == "ta-IN"

    def test_three_block_chain_propagates_stt_language_to_tts_block(
        self, context, audio_response
    ):
        """Full STT → RAG → TTS shape: TTS execution sees the language detected by STT."""
        stt_resp = _stt_text_response("kn-IN")
        rag_resp = LLMCallResponse(
            response=LLMResponse(
                provider_response_id="rag-1",
                conversation_id=None,
                model="gpt-4o",
                provider="openai",
                output=TextOutput(
                    content=ResponseTextContent(value="answer", language_code=None)
                ),
            ),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            provider_raw_response=None,
        )
        results = [
            BlockResult(response=stt_resp, usage=stt_resp.usage),
            BlockResult(response=rag_resp, usage=rag_resp.usage),
            BlockResult(response=audio_response, usage=audio_response.usage),
        ]
        blocks = [
            ChainBlock(config=make_config(), index=i, context=context) for i in range(3)
        ]
        chain = LLMChain(blocks, context)

        with patch(
            "app.services.llm.chain.chain.execute_llm_call",
            side_effect=results,
        ) as mock_execute:
            chain.execute(QueryParams(input="hello"))

        tts_call_kwargs = mock_execute.call_args_list[2].kwargs
        assert tts_call_kwargs["detected_language"] == "kn-IN"

    def test_detected_marker_substituted_with_detected_language(self):
        params = {"language_code": "{{detected}}", "voice": "Orus"}
        _substitute_detected_language_marker(params, "hi-IN", uuid4())
        assert params["language_code"] == "hi-IN"

    def test_detected_marker_falls_back_to_en_in_when_no_language_detected(self):
        """Regression guard: jobs.py hardcodes the fallback so a missing STT detection
        still produces a valid TTS request instead of forwarding '{{detected}}' to the
        provider."""
        params = {"target_language_code": "{{detected}}"}
        _substitute_detected_language_marker(params, None, uuid4())
        assert params["target_language_code"] == DETECTED_LANGUAGE_FALLBACK

    def test_detected_marker_substitution_leaves_other_keys_untouched(self):
        params = {
            "language_code": "{{detected}}",
            "voice": "Orus",
            "model": "bulbul:v3",
        }
        _substitute_detected_language_marker(params, "ta-IN", uuid4())
        assert params == {
            "language_code": "ta-IN",
            "voice": "Orus",
            "model": "bulbul:v3",
        }

    def test_detected_marker_substitution_noop_when_no_marker_present(self):
        """Pinned language must not be overwritten by detected_language."""
        params = {"language_code": "hi-IN"}
        _substitute_detected_language_marker(params, "ta-IN", uuid4())
        assert params["language_code"] == "hi-IN"

    def test_stt_failure_short_circuits_rag_and_tts(self, context):
        """A specific S2S guarantee: if STT fails, RAG and TTS are not executed."""
        blocks = [
            ChainBlock(config=make_config(), index=i, context=context) for i in range(3)
        ]
        chain = LLMChain(blocks, context)

        with patch(
            "app.services.llm.chain.chain.execute_llm_call",
            return_value=BlockResult(error="STT provider unavailable"),
        ) as mock_execute:
            result = chain.execute(QueryParams(input="hello"))

        assert not result.success
        assert result.error == "STT provider unavailable"
        assert mock_execute.call_count == 1
