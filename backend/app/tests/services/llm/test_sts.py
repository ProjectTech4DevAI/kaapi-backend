"""Pragmatic real-world tests for the /llm/sts endpoint.

Covers the three ways callers actually use STS:
  1. Minimal request (all defaults)
  2. Inline params override on a block
  3. Stored config reference on a block (config_id + version)

Plus the most-likely error paths (bad language code, conflicting block spec).
"""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.models.llm.constants import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
)
from app.models.llm.request import (
    AudioContent,
    AudioInput,
    RAGBlockSpec,
    SpeechToSpeechRequest,
    STTBlockSpec,
    TextLLMParams,
    TTSBlockSpec,
    TTSLLMParams,
)


# ---------- Fixtures ----------


@pytest.fixture
def audio_input() -> AudioInput:
    return AudioInput(
        type="audio",
        content=AudioContent(
            format="base64",
            value="SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Lg==",
            mime_type="audio/ogg",
        ),
    )


@pytest.fixture
def kb_ids() -> list[str]:
    return ["kb-faq"]


def _post(client: TestClient, headers: dict[str, str], payload: SpeechToSpeechRequest):
    return client.post(
        "api/v1/llm/sts",
        json=payload.model_dump(mode="json"),
        headers=headers,
    )


# ---------- Endpoint tests ----------


class TestSpeechToSpeechEndpoint:
    """The three real shapes users send + the common error paths."""

    def test_minimal_request_uses_all_defaults(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        audio_input: AudioInput,
        kb_ids: list[str],
    ):
        """Most common call: audio + KB ids, everything else defaulted."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start:
            payload = SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
            )
            response = _post(client, user_api_key_header, payload)

        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_start.assert_called_once()

        chain_request = mock_start.call_args.kwargs["request"]
        assert len(chain_request.blocks) == 3

        stt_params = chain_request.blocks[0].config.blob.completion.params
        tts_params = chain_request.blocks[2].config.blob.completion.params
        assert stt_params["model"] == DEFAULT_STT_MODEL
        assert tts_params["model"] == DEFAULT_TTS_MODEL
        assert tts_params["response_format"] == "ogg"

    def test_inline_overrides_on_rag_and_tts(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        audio_input: AudioInput,
        kb_ids: list[str],
    ):
        """Caller tweaks RAG instructions and picks a different TTS voice."""
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start:
            payload = SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                input_language="hi-IN",
                rag=RAGBlockSpec(
                    params=TextLLMParams(
                        model="gpt-4o-mini",
                        instructions="Reply in one short sentence.",
                        temperature=0.2,
                    )
                ),
                tts=TTSBlockSpec(params=TTSLLMParams(voice="Orus")),
            )
            response = _post(client, user_api_key_header, payload)

        assert response.status_code == 200
        assert response.json()["success"] is True

        chain_request = mock_start.call_args.kwargs["request"]
        rag_params = chain_request.blocks[1].config.blob.completion.params
        tts_params = chain_request.blocks[2].config.blob.completion.params

        assert rag_params["model"] == "gpt-4o-mini"
        assert rag_params["instructions"] == "Reply in one short sentence."
        assert rag_params["knowledge_base_ids"] == kb_ids
        assert tts_params["voice"] == "Orus"
        # Route-owned fields still applied:
        assert tts_params["language"] == "hi-IN"
        assert tts_params["response_format"] == "ogg"

    def test_stored_config_reference_for_rag(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        audio_input: AudioInput,
        kb_ids: list[str],
    ):
        """Caller points the RAG block at a saved config instead of inlining params."""
        config_id = uuid4()
        with patch("app.api.routes.llm_sts.start_chain_job") as mock_start:
            payload = SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                rag=RAGBlockSpec(config_id=config_id, config_version=1),
            )
            response = _post(client, user_api_key_header, payload)

        assert response.status_code == 200
        assert response.json()["success"] is True

        chain_request = mock_start.call_args.kwargs["request"]
        rag_config = chain_request.blocks[1].config
        assert rag_config.id == config_id
        assert rag_config.version == 1
        assert rag_config.blob is None

    def test_invalid_language_code_returns_error(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        audio_input: AudioInput,
        kb_ids: list[str],
    ):
        payload = SpeechToSpeechRequest(
            query=audio_input,
            knowledge_base_ids=kb_ids,
            input_language="hindi",  # not BCP-47
        )
        response = _post(client, user_api_key_header, payload)

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert "Unsupported input language" in body["error"]

    def test_block_spec_rejects_both_params_and_config_ref(
        self,
        audio_input: AudioInput,
        kb_ids: list[str],
    ):
        """XOR enforcement: caller can't send both an inline spec and a stored ref."""
        with pytest.raises(ValidationError):
            SpeechToSpeechRequest(
                query=audio_input,
                knowledge_base_ids=kb_ids,
                stt=STTBlockSpec(
                    config_id=uuid4(),
                    config_version=1,
                    params={"model": DEFAULT_STT_MODEL},
                ),
            )
