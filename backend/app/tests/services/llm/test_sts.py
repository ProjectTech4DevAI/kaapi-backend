"""Tests for the /llm/chain/sts endpoint.

Covers, at the chain-construction layer (start_chain_job mocked):

- Default block construction (models, voices, formats, KB injection, intermediate flags)
- Language resolution (auto, pinned, cross-language, BCP-47 normalisation, route-owns)
- Provider matrix (independent set per block, default RAG provider, Google STT path)
- Inline overrides for the RAG block (richest case) and KB-id preservation under override
- Stored-config references (all-stored + mixed-stored/inline)
- Metadata construction (speech_to_speech flag, language fields, caller-merge, key-precedence)
- Error paths: bad/forbidden language codes, XOR violations, missing fields, URL audio input
- S2S-specific edge cases: multi-KB forwarding, large KB list, payload structure

The deeper edge cases ({{detected}} substitution fallback, STT failure halting RAG/TTS) are
covered in test_chain.py against the chain primitives.
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

    def test_default_models_voice_format_and_temperature(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """One canonical defaults test — covers STT model, RAG model+temp, TTS model/voice/format."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        blocks = _chain_request(mock).blocks
        stt = blocks[0].config.blob.completion.params
        rag = blocks[1].config.blob.completion.params
        tts = blocks[2].config.blob.completion.params

        assert stt["model"] == DEFAULT_STT_MODEL
        assert rag["model"] == DEFAULT_RAG_MODEL
        assert rag["temperature"] == 0.1
        assert tts["model"] == DEFAULT_TTS_MODEL
        assert tts["voice"] == DEFAULT_TTS_VOICE
        assert tts["response_format"] == "ogg"

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

    def test_auto_input_without_output_yields_detected_marker(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """input='auto' + no output must produce the '{{detected}}' marker so
        jobs.py substitutes the STT-detected language into the TTS config at
        execution time. Returning 'auto' would forward 'auto' to TTS providers,
        which 400 on Sarvam (default per docs)."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=kb_ids),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == "{{detected}}"

    def test_unknown_input_without_output_also_yields_detected_marker(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """'unknown' as input has the same semantics as 'auto' for TTS resolution:
        the user wants STT to figure the language out."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language="unknown",
                ),
            )
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == "{{detected}}"

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

    @pytest.mark.parametrize(
        "raw,normalised",
        [
            ("hi-in", "hi-IN"),
            ("HI-IN", "hi-IN"),
            ("Hi-In", "hi-IN"),
        ],
    )
    def test_bcp47_normalisation(
        self, client, user_api_key_header, audio_input, kb_ids, raw, normalised
    ):
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            response = _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    input_language=raw,
                ),
            )
        assert response.status_code == 200
        tts_params = _chain_request(mock).blocks[2].config.blob.completion.params
        assert tts_params["language"] == normalised

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

    def test_google_stt_with_gemini_model(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """Non-Sarvam STT path — the only one not covered by the matrix test below."""
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
    def test_rag_model_instructions_and_temperature_override(
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


# ---------- Stored config references ----------


class TestStoredConfigRefs:
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
        assert all(b.config.blob is None for b in blocks)

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
    def test_metadata_has_speech_to_speech_flag_and_languages(
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
        assert meta["speech_to_speech"] is True
        assert meta["input_language"] == "mr-IN"
        assert meta["output_language"] == "ta-IN"

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
        assert "input language" in response.json()["error"].lower()

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
        assert "output language" in response.json()["error"].lower()

    @pytest.mark.parametrize("forbidden", ["unknown", "auto"])
    def test_detection_sentinels_rejected_as_output_language(
        self, client, user_api_key_header, audio_input, kb_ids, forbidden
    ):
        """'unknown' / 'auto' are STT-only sentinels; TTS needs a concrete language."""
        response = _post(
            client,
            user_api_key_header,
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                output_language=forbidden,
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


# ---------- S2S-specific edge cases ----------


class TestEdgeCases:
    def test_multiple_knowledge_base_ids_forwarded_to_rag(
        self, client, user_api_key_header, audio_input
    ):
        """All KB IDs passed in the request must reach the RAG block, in order."""
        many_kbs = ["kb-a", "kb-b", "kb-c", "kb-d", "kb-e"]
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(query=audio_input, knowledge_base_ids=many_kbs),
            )
        rag_params = _chain_request(mock).blocks[1].config.blob.completion.params
        assert rag_params["knowledge_base_ids"] == many_kbs

    def test_kb_ids_overwrite_user_supplied_kb_ids_in_rag_params(
        self, client, user_api_key_header, audio_input, kb_ids
    ):
        """Top-level knowledge_base_ids is the source of truth; user-supplied params.knowledge_base_ids must be overwritten."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock:
            _post(
                client,
                user_api_key_header,
                SpeechToSpeechRequest(
                    query=audio_input,
                    knowledge_base_ids=kb_ids,
                    rag=RAGBlockSpec(
                        params=TextLLMParams(
                            model="gpt-4o",
                            knowledge_base_ids=["smuggled-kb"],
                        )
                    ),
                ),
            )
        rag_params = _chain_request(mock).blocks[1].config.blob.completion.params
        assert rag_params["knowledge_base_ids"] == kb_ids
