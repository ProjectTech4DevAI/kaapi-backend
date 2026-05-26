"""Tests for the /llm/chain/sts endpoint.

Covers:
- Default block construction (models, voices, formats)
- Language resolution (auto, pinned, cross-language, BCP-47 normalisation)
- Provider combos (google/sarvamai/elevenlabs for STT/TTS, openai for RAG)
- Inline param overrides on each block
- Stored config references on each block (and mixed)
- Intermediate callback flags
- Metadata construction (defaults, inline overrides, stored-ref labels)
- Error paths: bad language codes, "unknown"/"auto" as output, XOR violations,
  missing required fields
"""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.models.llm.constants import (
    DEFAULT_RAG_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
)
from app.models.llm.request import (
    AudioContent,
    AudioInput,
    RAGBlockSpec,
    SpeechToSpeechRequest,
    STTBlockSpec,
    STTLLMParams,
    TextLLMParams,
    TTSBlockSpec,
    TTSLLMParams,
)

URL = "/api/v1/llm/chain/sts"


# ---------- Fixtures ----------


@pytest.fixture
def audio_input() -> AudioInput:
    return AudioInput(
        content=AudioContent(
            format="base64",
            value="SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Lg==",
            mime_type="audio/ogg",
        )
    )


@pytest.fixture
def audio_url_input() -> AudioInput:
    return AudioInput(
        content=AudioContent(
            format="url",
            value="https://example.com/audio.ogg",
            mime_type="audio/ogg",
        )
    )


@pytest.fixture
def kb_ids() -> list[str]:
    return ["kb-faq", "kb-product"]


def _post(client: TestClient, headers: dict, payload: SpeechToSpeechRequest):
    return client.post(URL, json=payload.model_dump(mode="json"), headers=headers)


def _chain_request(mock_start):
    return mock_start.call_args.kwargs["request"]


# ---------- Defaults ----------


class TestDefaults:
    def test_three_blocks_always_created(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            response = _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        assert response.status_code == 200
        assert len(_chain_request(mock).blocks) == 3

    def test_default_stt_model(self, client, user_api_key_header, audio_input, kb_ids):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        stt_params = _chain_request(mock).blocks[0].config.blob.completion.params
        assert stt_params["model"] == DEFAULT_STT_MODEL

    def test_default_rag_model_and_temperature(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        rag_params = _chain_request(mock).blocks[1].config.blob.completion.params
        assert rag_params["model"] == DEFAULT_RAG_MODEL
        assert rag_params["temperature"] == 0.1

    def test_default_tts_model_voice_format(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["model"] == DEFAULT_TTS_MODEL
        assert tts_params["voice"] == DEFAULT_TTS_VOICE
        assert tts_params["response_format"] == "ogg"

    def test_rag_block_always_has_knowledge_base_ids(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        rag_params = _chain_request(mock).blocks[1].config.blob.completion.params
        assert rag_params["knowledge_base_ids"] == kb_ids

    def test_stt_and_rag_are_intermediate_tts_is_not(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        blocks = _chain_request(mock).blocks
        assert blocks[0].intermediate_callback is True
        assert blocks[1].intermediate_callback is True
        assert blocks[2].intermediate_callback is False

    def test_default_stt_input_language_is_auto(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        stt_params = _chain_request(mock).blocks[0].config.blob.completion.params
        assert stt_params["input_language"] == "auto"


# ---------- Language resolution ----------


class TestLanguageResolution:
    def test_pinned_input_propagates_to_tts_when_output_not_set(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="hi-IN",
                ),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == "hi-IN"

    def test_explicit_output_language_overrides_input(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="hi-IN",
                    output_language="ta-IN",
                ),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == "ta-IN"

    def test_auto_input_with_pinned_output(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    output_language="kn-IN",
                ),
            )
        stt_params = _chain_request(mock).blocks[0].config.blob.completion.params
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert stt_params["input_language"] == "auto"
        assert tts_params["language"] == "kn-IN"

    def test_bcp47_normalisation_lowercase_region(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """'hi-in' should be normalised to 'hi-IN' before validation."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            response = _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="hi-in",
                ),
            )
        assert response.status_code == 200
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == "hi-IN"

    def test_bcp47_normalisation_uppercase_language(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            response = _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="HI-IN",
                ),
            )
        assert response.status_code == 200
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == "hi-IN"

    def test_route_always_owns_stt_input_language(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """User passing input_language via STTBlockSpec params should be overridden by the route."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="bn-IN",
                    stt=STTBlockSpec(
                        params=STTLLMParams(model="saaras:v3", input_language="hi-IN")
                    ),
                ),
            )
        stt_params = _chain_request(mock).blocks[0].config.blob.completion.params
        assert stt_params["input_language"] == "bn-IN"

    def test_route_always_owns_tts_language(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """User passing language via TTSBlockSpec params should be overridden by the route."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="te-IN",
                    tts=TTSBlockSpec(
                        params=TTSLLMParams(model=DEFAULT_TTS_MODEL, language="hi-IN")
                    ),
                ),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == "te-IN"


# ---------- Provider combos ----------


class TestProviders:
    def test_stt_sarvamai_provider(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt_provider="sarvamai",
                ),
            )
        stt_completion = _chain_request(mock).blocks[0].config.blob.completion
        assert stt_completion.provider == "sarvamai"

    def test_tts_sarvamai_provider(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    tts_provider="sarvamai",
                ),
            )
        tts_completion = _chain_request(mock).blocks[2].config.blob.completion
        assert tts_completion.provider == "sarvamai"

    def test_tts_elevenlabs_provider(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    tts_provider="elevenlabs",
                ),
            )
        tts_completion = _chain_request(mock).blocks[2].config.blob.completion
        assert tts_completion.provider == "elevenlabs"

    def test_rag_provider_defaults_to_openai(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        rag_completion = _chain_request(mock).blocks[1].config.blob.completion
        assert rag_completion.provider == "openai"

    def test_sarvamai_stt_with_saaras_model(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt_provider="sarvamai",
                    stt=STTBlockSpec(params=STTLLMParams(model="saaras:v3")),
                ),
            )
        stt_completion = _chain_request(mock).blocks[0].config.blob.completion
        assert stt_completion.provider == "sarvamai"
        assert stt_completion.params["model"] == "saaras:v3"

    def test_google_stt_with_gemini_model(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt_provider="google",
                    stt=STTBlockSpec(params=STTLLMParams(model="gemini-2.5-pro")),
                ),
            )
        stt_completion = _chain_request(mock).blocks[0].config.blob.completion
        assert stt_completion.provider == "google"
        assert stt_completion.params["model"] == "gemini-2.5-pro"

    def test_all_three_providers_set_independently(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt_provider="sarvamai",
                    rag_provider="openai",
                    tts_provider="elevenlabs",
                ),
            )
        blocks = _chain_request(mock).blocks
        assert blocks[0].config.blob.completion.provider == "sarvamai"
        assert blocks[1].config.blob.completion.provider == "openai"
        assert blocks[2].config.blob.completion.provider == "elevenlabs"


# ---------- Inline param overrides ----------


class TestInlineOverrides:
    def test_stt_model_override(self, client, user_api_key_header, audio_input, kb_ids):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt=STTBlockSpec(params=STTLLMParams(model="saaras:v3")),
                ),
            )
        stt_params = _chain_request(mock).blocks[0].config.blob.completion.params
        assert stt_params["model"] == "saaras:v3"

    def test_rag_model_and_instructions_override(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    rag=RAGBlockSpec(
                        params=TextLLMParams(
                            model="gpt-4o-mini",
                            instructions="Be brief.",
                            temperature=0.5,
                        )
                    ),
                ),
            )
        rag_params = _chain_request(mock).blocks[1].config.blob.completion.params
        assert rag_params["model"] == "gpt-4o-mini"
        assert rag_params["instructions"] == "Be brief."
        assert rag_params["temperature"] == 0.5

    def test_rag_inline_still_injects_kb_ids(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """knowledge_base_ids must be injected even when user provides partial RAG params."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    rag=RAGBlockSpec(params=TextLLMParams(model="gpt-4o-mini")),
                ),
            )
        rag_params = _chain_request(mock).blocks[1].config.blob.completion.params
        assert rag_params["knowledge_base_ids"] == kb_ids

    def test_tts_voice_override(self, client, user_api_key_header, audio_input, kb_ids):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    tts=TTSBlockSpec(params=TTSLLMParams(voice="Orus")),
                ),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["voice"] == "Orus"

    def test_tts_format_override(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    tts=TTSBlockSpec(params=TTSLLMParams(response_format="mp3")),
                ),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["response_format"] == "mp3"


# ---------- Stored config references ----------


class TestStoredConfigRefs:
    def test_stored_stt_block(self, client, user_api_key_header, audio_input, kb_ids):
        config_id = uuid4()
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt=STTBlockSpec(config_id=config_id, config_version=2),
                ),
            )
        stt_config = _chain_request(mock).blocks[0].config
        assert stt_config.id == config_id
        assert stt_config.version == 2
        assert stt_config.blob is None

    def test_stored_rag_block(self, client, user_api_key_header, audio_input, kb_ids):
        config_id = uuid4()
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    rag=RAGBlockSpec(config_id=config_id, config_version=1),
                ),
            )
        rag_config = _chain_request(mock).blocks[1].config
        assert rag_config.id == config_id
        assert rag_config.version == 1
        assert rag_config.blob is None

    def test_stored_tts_block(self, client, user_api_key_header, audio_input, kb_ids):
        config_id = uuid4()
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    tts=TTSBlockSpec(config_id=config_id, config_version=3),
                ),
            )
        tts_config = _chain_request(mock).blocks[2].config
        assert tts_config.id == config_id
        assert tts_config.version == 3
        assert tts_config.blob is None

    def test_all_blocks_stored(self, client, user_api_key_header, audio_input, kb_ids):
        stt_id, rag_id, tts_id = uuid4(), uuid4(), uuid4()
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt=STTBlockSpec(config_id=stt_id, config_version=1),
                    rag=RAGBlockSpec(config_id=rag_id, config_version=1),
                    tts=TTSBlockSpec(config_id=tts_id, config_version=1),
                ),
            )
        blocks = _chain_request(mock).blocks
        assert blocks[0].config.id == stt_id
        assert blocks[1].config.id == rag_id
        assert blocks[2].config.id == tts_id

    def test_mixed_stored_and_inline(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """STT stored, RAG inline, TTS stored."""
        stt_id, tts_id = uuid4(), uuid4()
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt=STTBlockSpec(config_id=stt_id, config_version=1),
                    rag=RAGBlockSpec(params=TextLLMParams(model="gpt-4o-mini")),
                    tts=TTSBlockSpec(config_id=tts_id, config_version=2),
                ),
            )
        blocks = _chain_request(mock).blocks
        assert blocks[0].config.id == stt_id
        assert blocks[1].config.blob is not None
        assert blocks[2].config.id == tts_id


# ---------- Metadata ----------


class TestMetadata:
    def test_metadata_has_speech_to_speech_flag(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        meta = _chain_request(mock).request_metadata
        assert meta["speech_to_speech"] is True

    def test_metadata_reflects_resolved_languages(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="mr-IN",
                    output_language="ta-IN",
                ),
            )
        meta = _chain_request(mock).request_metadata
        assert meta["input_language"] == "mr-IN"
        assert meta["output_language"] == "ta-IN"

    def test_metadata_default_model_labels(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        meta = _chain_request(mock).request_metadata
        assert meta["stt_model"] == DEFAULT_STT_MODEL
        assert meta["llm_model"] == DEFAULT_RAG_MODEL
        assert meta["tts_model"] == DEFAULT_TTS_MODEL

    def test_metadata_inline_model_labels(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt=STTBlockSpec(params=STTLLMParams(model="saaras:v3")),
                    rag=RAGBlockSpec(params=TextLLMParams(model="gpt-4o-mini")),
                    tts=TTSBlockSpec(
                        params=TTSLLMParams(model="gemini-2.5-flash-preview-tts")
                    ),
                ),
            )
        meta = _chain_request(mock).request_metadata
        assert meta["stt_model"] == "saaras:v3"
        assert meta["llm_model"] == "gpt-4o-mini"
        assert meta["tts_model"] == "gemini-2.5-flash-preview-tts"

    def test_metadata_stored_ref_label_format(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        config_id = uuid4()
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    stt=STTBlockSpec(config_id=config_id, config_version=4),
                ),
            )
        meta = _chain_request(mock).request_metadata
        assert meta["stt_model"] == f"stored:{config_id}@v4"

    def test_caller_metadata_is_preserved(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    request_metadata={"session_id": "abc123", "user": "test"},
                ),
            )
        meta = _chain_request(mock).request_metadata
        assert meta["session_id"] == "abc123"
        assert meta["user"] == "test"
        assert meta["speech_to_speech"] is True

    def test_caller_metadata_cannot_override_sts_keys(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """STS-injected keys must win over any caller-provided values."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="hi-IN",
                    request_metadata={
                        "input_language": "caller-value",
                        "output_language": "caller-value",
                        "speech_to_speech": False,
                        "stt_model": "caller-value",
                        "llm_model": "caller-value",
                        "tts_model": "caller-value",
                    },
                ),
            )
        meta = _chain_request(mock).request_metadata
        assert meta["input_language"] == "hi-IN"
        assert meta["output_language"] == "hi-IN"
        assert meta["speech_to_speech"] is True
        assert meta["stt_model"] == DEFAULT_STT_MODEL
        assert meta["llm_model"] == DEFAULT_RAG_MODEL
        assert meta["tts_model"] == DEFAULT_TTS_MODEL


# ---------- Error paths ----------


class TestErrorPaths:
    def test_invalid_input_language_returns_422(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        response = _post(
            client,
            user_api_key_header,
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                input_language="hindi",
            ),
        )
        assert response.status_code == 422
        assert "input language" in response.json()["detail"].lower()

    def test_invalid_output_language_returns_422(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        response = _post(
            client,
            user_api_key_header,
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                output_language="english",
            ),
        )
        assert response.status_code == 422
        assert "output language" in response.json()["detail"].lower()

    def test_unknown_as_output_language_returns_422(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """'unknown' is a detection sentinel, not a valid TTS target."""
        response = _post(
            client,
            user_api_key_header,
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                output_language="unknown",
            ),
        )
        assert response.status_code == 422

    def test_auto_as_output_language_returns_422(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """'auto' cannot be pinned as TTS output — TTS needs a concrete language."""
        response = _post(
            client,
            user_api_key_header,
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                output_language="auto",
            ),
        )
        assert response.status_code == 422

    def test_auto_as_input_language_is_valid(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job"):
            response = _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="auto",
                ),
            )
        assert response.status_code == 200

    def test_explicit_null_input_language_defaults_to_auto(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """Sending input_language=null in JSON should still result in STT getting 'auto'."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            response = client.post(
                URL,
                json={
                    "query": audio_input.model_dump(mode="json"),
                    "knowledge_base_ids": kb_ids,
                    "input_language": None,
                },
                headers=user_api_key_header,
            )
        assert response.status_code == 200
        stt_params = _chain_request(mock).blocks[0].config.blob.completion.params
        assert stt_params["input_language"] == "auto"

    def test_empty_knowledge_base_ids_rejected(self, audio_input):
        with pytest.raises(ValidationError):
            SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=[])

    def test_block_spec_rejects_params_and_config_id_together(
        self, audio_input, kb_ids
    ):
        with pytest.raises(ValidationError):
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                stt=STTBlockSpec(
                    config_id=uuid4(),
                    config_version=1,
                    params=STTLLMParams(model="saaras:v3"),
                ),
            )

    def test_block_spec_rejects_config_id_without_version(self, audio_input, kb_ids):
        with pytest.raises(ValidationError):
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                rag=RAGBlockSpec(config_id=uuid4()),
            )

    def test_url_audio_input_accepted(
        self, client, user_api_key_header, audio_url_input, kb_ids
    ):
        with patch("app.api.routes.llm_sts.start_chain_job"):
            response = _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_url_input, knowledge_base_ids=kb_ids),
            )
        assert response.status_code == 200
