"""
Test cases for Speech-to-Speech (STS) functionality.

Tests cover:
1. Language detection and propagation through STT → RAG → TTS chain
2. BCP-47 language code validation
3. Real-world use cases (auto-detection, explicit languages, cross-language)
"""

from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.models.llm.request import (
    AudioContent,
    AudioInput,
    LLMModel,
    STTModel,
    SpeechToSpeechRequest,
    TTSModel,
)
from app.models.llm.response import (
    LLMCallResponse,
    LLMResponse,
    TextOutput,
    TextContent as ResponseTextContent,
    Usage,
)
from app.services.llm.chain.chain import ChainContext, result_to_query
from app.services.llm.chain.types import BlockResult
from app.services.llm.chain.utils import (
    SUPPORTED_LANGUAGE_CODES,
    build_stt_block,
    build_tts_block,
)


# ============================================================================
# Unit Tests: Language Detection Flow
# ============================================================================


class TestLanguageDetectionFlow:
    """Test language detection and propagation through the chain."""

    def test_result_to_query_preserves_language_code(self):
        """STT output with language_code should be preserved when converting to next block's input."""
        # Simulate STT response with detected Hindi
        stt_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id="stt-resp-1",
                conversation_id=None,
                model="saaras:v3",
                provider="sarvamai-native",
                output=TextOutput(
                    content=ResponseTextContent(
                        value="नमस्ते, आप कैसे हैं?", language_code="hi-IN"
                    )
                ),
            ),
            usage=Usage(input_tokens=0, output_tokens=10, total_tokens=10),
        )

        result = BlockResult(response=stt_response, usage=stt_response.usage)
        context = ChainContext(
            job_id=uuid4(),
            chain_id=uuid4(),
            project_id=1,
            organization_id=1,
            callback_url=None,
            total_blocks=3,
        )

        # Convert STT output to RAG input
        query = result_to_query(result, context)

        # Language code should be preserved
        assert query.input.content.language_code == "hi-IN"
        assert query.input.content.value == "नमस्ते, आप कैसे हैं?"

        # Context should store detected language for TTS
        assert context.detected_language == "hi-IN"

    def test_result_to_query_without_language_code(self):
        """RAG output without language_code should not break the chain."""
        # Simulate RAG response (no language_code)
        rag_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id="rag-resp-1",
                conversation_id=None,
                model="gpt-4o",
                provider="openai",
                output=TextOutput(
                    content=ResponseTextContent(
                        value="The capital of India is New Delhi."
                    )
                ),
            ),
            usage=Usage(input_tokens=50, output_tokens=12, total_tokens=62),
        )

        result = BlockResult(response=rag_response, usage=rag_response.usage)
        context = ChainContext(
            job_id=uuid4(),
            chain_id=uuid4(),
            project_id=1,
            organization_id=1,
            callback_url=None,
            total_blocks=3,
            detected_language="hi-IN",  # From previous STT block
        )

        # Convert RAG output to TTS input
        query = result_to_query(result, context)

        # Should work fine even without language_code
        assert query.input.content.value == "The capital of India is New Delhi."
        # Context should retain previously detected language
        assert context.detected_language == "hi-IN"

    def test_detected_marker_replacement(self):
        """{{detected}} marker in TTS should be replaced with actual detected language."""
        from app.services.llm.jobs import execute_llm_call
        from app.models.llm.request import (
            LLMCallConfig,
            ConfigBlob,
            NativeCompletionConfig,
            QueryParams,
        )

        config = LLMCallConfig(
            blob=ConfigBlob(
                completion=NativeCompletionConfig(
                    provider="sarvamai-native",
                    type="tts",
                    params={
                        "model": "bulbul:v3",
                        "voice": "simran",
                        "target_language_code": "{{detected}}",  # Marker to be replaced
                        "speaker": "simran",
                        "output_audio_codec": "opus",
                    },
                )
            )
        )

        with patch("app.services.llm.jobs.get_llm_provider") as mock_provider, patch(
            "app.services.llm.jobs.Session"
        ):
            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance
            mock_provider_instance.execute.return_value = (None, "test error")

            # Call with detected_language
            execute_llm_call(
                config=config,
                query=QueryParams(input="Test text"),
                job_id=uuid4(),
                project_id=1,
                organization_id=1,
                request_metadata=None,
                langfuse_credentials=None,
                detected_language="ta-IN",  # Detected Tamil
            )

            # Verify {{detected}} was replaced with ta-IN
            # The marker replacement happens in execute_llm_call before provider.execute is called
            # So we check the modified config params
            call_args = mock_provider_instance.execute.call_args
            # execute is called with (completion_config, query, resolved_input, include_provider_raw_response)
            if call_args:
                completion_config = (
                    call_args[1]["completion_config"]
                    if len(call_args) > 1 and "completion_config" in call_args[1]
                    else call_args[0][0]
                )
                assert completion_config.params["target_language_code"] == "ta-IN"


# ============================================================================
# Unit Tests: Block Building
# ============================================================================


class TestSTSBlockBuilding:
    """Test STT and TTS block configuration."""

    def test_build_stt_block_with_auto(self):
        """Auto language should map to 'unknown' for Sarvam."""
        block = build_stt_block(STTModel.SARVAM, "auto")

        params = block.config.blob.completion.params
        assert params["language_code"] == "unknown"
        assert params["model"] == "saaras:v3"
        assert params["mode"] == "transcribe"

    def test_build_stt_block_with_specific_language(self):
        """Specific BCP-47 code should be used as-is."""
        block = build_stt_block(STTModel.SARVAM, "hi-IN")

        params = block.config.blob.completion.params
        assert params["language_code"] == "hi-IN"

    def test_build_tts_block_with_detected_marker(self):
        """TTS should accept {{detected}} marker for dynamic language."""
        block = build_tts_block(TTSModel.SARVAM, "{{detected}}")

        params = block.config.blob.completion.params
        assert params["target_language_code"] == "{{detected}}"
        assert params["model"] == "bulbul:v3"

    def test_build_tts_block_with_specific_language(self):
        """TTS should accept specific BCP-47 codes."""
        block = build_tts_block(TTSModel.SARVAM, "ta-IN")

        params = block.config.blob.completion.params
        assert params["target_language_code"] == "ta-IN"


# ============================================================================
# Integration Tests: Speech-to-Speech Endpoint
# ============================================================================


@pytest.fixture
def mock_audio_input():
    """Sample audio input (base64 encoded)."""
    return AudioInput(
        type="audio",
        content=AudioContent(
            format="base64",
            value="SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//...",
            mime_type="audio/ogg",
        ),
    )


@pytest.fixture
def knowledge_base_ids():
    """Sample knowledge base IDs."""
    return ["kb-india-facts", "kb-general-knowledge"]


class TestSpeechToSpeechEndpoint:
    """Test the /llm/sts endpoint with realistic scenarios."""

    def test_sts_auto_detection_hindi_to_hindi(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        Real-world scenario: User sends Hindi voice note, expects Hindi response.
        Most common use case - auto-detect input, same language output.
        """
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start_job:
            payload = SpeechToSpeechRequest(
                query=mock_audio_input,
                knowledge_base_ids=knowledge_base_ids,
                input_language="auto",  # Auto-detect
                output_language=None,  # Should default to detected language
                stt_model=STTModel.SARVAM,
                tts_model=TTSModel.SARVAM,
                llm_model=LLMModel.GPT4O,
                callback_url="https://example.com/callback",
            )

            response = client.post(
                "api/v1/llm/sts",
                json=payload.model_dump(mode="json"),
                headers=user_api_key_header,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Speech-to-speech processing initiated" in data["data"]["message"]

            # Verify job was started
            mock_start_job.assert_called_once()

    def test_sts_explicit_tamil_to_tamil(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        Scenario: Tamil user explicitly sets language to avoid auto-detection.
        Use case: Better accuracy when language is known.
        """
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start_job:
            payload = SpeechToSpeechRequest(
                query=mock_audio_input,
                knowledge_base_ids=knowledge_base_ids,
                input_language="ta-IN",
                output_language="ta-IN",
                stt_model=STTModel.SARVAM,
                tts_model=TTSModel.SARVAM,
                llm_model=LLMModel.GPT4O_MINI,
                callback_url="https://example.com/callback",
            )

            response = client.post(
                "api/v1/llm/sts",
                json=payload.model_dump(mode="json"),
                headers=user_api_key_header,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_start_job.assert_called_once()

    def test_sts_cross_language_hindi_to_english(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        Scenario: User speaks Hindi but wants response in English.
        Use case: Language learning, multilingual support.
        """
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start_job:
            payload = SpeechToSpeechRequest(
                query=mock_audio_input,
                knowledge_base_ids=knowledge_base_ids,
                input_language="hi-IN",
                output_language="en-IN",  # Respond in English
                stt_model=STTModel.SARVAM,
                tts_model=TTSModel.SARVAM,
                llm_model=LLMModel.GPT4O,
                callback_url="https://example.com/callback",
            )

            response = client.post(
                "api/v1/llm/sts",
                json=payload.model_dump(mode="json"),
                headers=user_api_key_header,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_start_job.assert_called_once()

    def test_sts_invalid_input_language_code(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        Error case: User provides invalid BCP-47 code.
        Should reject with clear error message.
        """
        payload = SpeechToSpeechRequest(
            query=mock_audio_input,
            knowledge_base_ids=knowledge_base_ids,
            input_language="hindi",  # Invalid - should be 'hi-IN'
            output_language="en-IN",
            stt_model=STTModel.SARVAM,
            tts_model=TTSModel.SARVAM,
            llm_model=LLMModel.GPT4O,
        )

        response = client.post(
            "api/v1/llm/sts",
            json=payload.model_dump(mode="json"),
            headers=user_api_key_header,
        )

        assert response.status_code == 200  # API returns 200 with error in body
        data = response.json()
        assert data["success"] is False
        assert "Unsupported input language code" in data["error"]
        assert "hindi" in data["error"]

    def test_sts_invalid_output_language_code(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        Error case: Invalid output language code.
        """
        payload = SpeechToSpeechRequest(
            query=mock_audio_input,
            knowledge_base_ids=knowledge_base_ids,
            input_language="hi-IN",
            output_language="french",  # Invalid - should be BCP-47
            stt_model=STTModel.SARVAM,
            tts_model=TTSModel.SARVAM,
            llm_model=LLMModel.GPT4O,
        )

        response = client.post(
            "api/v1/llm/sts",
            json=payload.model_dump(mode="json"),
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Unsupported output language code" in data["error"]

    def test_sts_case_insensitive_language_codes(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        User-friendly case: BCP-47 codes should be case-insensitive.
        'hi-in' should be normalized to 'hi-IN'.
        """
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start_job:
            payload = SpeechToSpeechRequest(
                query=mock_audio_input,
                knowledge_base_ids=knowledge_base_ids,
                input_language="hi-in",  # Lowercase
                output_language="en-in",  # Lowercase
                stt_model=STTModel.SARVAM,
                tts_model=TTSModel.SARVAM,
                llm_model=LLMModel.GPT4O,
            )

            response = client.post(
                "api/v1/llm/sts",
                json=payload.model_dump(mode="json"),
                headers=user_api_key_header,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_start_job.assert_called_once()

    def test_sts_regional_languages(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        Test support for regional Indian languages.
        Scenario: Malayalam speaker from Kerala.
        """
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start_job:
            payload = SpeechToSpeechRequest(
                query=mock_audio_input,
                knowledge_base_ids=knowledge_base_ids,
                input_language="ml-IN",  # Malayalam
                output_language="ml-IN",
                stt_model=STTModel.SARVAM,
                tts_model=TTSModel.SARVAM,
                llm_model=LLMModel.GPT4O_MINI,
            )

            response = client.post(
                "api/v1/llm/sts",
                json=payload.model_dump(mode="json"),
                headers=user_api_key_header,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_start_job.assert_called_once()

    def test_sts_without_callback_url(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        mock_audio_input,
        knowledge_base_ids,
    ):
        """
        Callback URL is optional - job should still start.
        """
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start_job:
            payload = SpeechToSpeechRequest(
                query=mock_audio_input,
                knowledge_base_ids=knowledge_base_ids,
                input_language="auto",
                stt_model=STTModel.SARVAM,
                tts_model=TTSModel.SARVAM,
                llm_model=LLMModel.GPT4O,
                # No callback_url
            )

            response = client.post(
                "api/v1/llm/sts",
                json=payload.model_dump(mode="json"),
                headers=user_api_key_header,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_start_job.assert_called_once()


# ============================================================================
# Unit Tests: Language Code Validation
# ============================================================================


class TestLanguageCodeSupport:
    """Verify all supported BCP-47 codes are valid."""

    def test_all_supported_codes_are_valid(self):
        """All codes in SUPPORTED_LANGUAGE_CODES should be valid BCP-47 format."""
        valid_codes = {
            "auto",
            "unknown",
            "en-IN",
            "hi-IN",
            "bn-IN",
            "kn-IN",
            "ml-IN",
            "mr-IN",
            "od-IN",
            "pa-IN",
            "ta-IN",
            "te-IN",
            "gu-IN",
            "as-IN",
            "ur-IN",
            "ne-IN",
            "kok-IN",
            "ks-IN",
            "sd-IN",
            "sa-IN",
            "sat-IN",
            "mni-IN",
            "brx-IN",
            "mai-IN",
            "doi-IN",
        }

        assert SUPPORTED_LANGUAGE_CODES == valid_codes

    def test_major_indian_languages_supported(self):
        """Verify major Indian languages are supported."""
        major_languages = {
            "hi-IN",  # Hindi
            "bn-IN",  # Bengali
            "te-IN",  # Telugu
            "mr-IN",  # Marathi
            "ta-IN",  # Tamil
            "ur-IN",  # Urdu
            "gu-IN",  # Gujarati
            "kn-IN",  # Kannada
            "ml-IN",  # Malayalam
            "pa-IN",  # Punjabi
        }

        assert major_languages.issubset(SUPPORTED_LANGUAGE_CODES)
