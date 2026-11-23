"""Tests for the /llm/chain/sts endpoint at the chain-construction layer.

start_chain_job is patched so the assertions run against the LLMChainRequest the
route builds. {{detected}} substitution and STT-failure halting are covered in
test_chain.py against the chain primitives.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlmodel import Session

from app.crud import JobCrud
from app.models import JobType
from app.models.llm.constants import (
    DEFAULT_RAG_MODEL,
    DEFAULT_SARVAM_STT_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
)
from app.models.llm.request import (
    AudioContent,
    ConfigBlob,
    AudioInput,
    LLMStepConfig,
    QueryParams,
    SpeechToSpeechConfig,
    SpeechToSpeechRequest,
    STTLLMParams,
    STTStepConfig,
    TextLLMParams,
    TTSLLMParams,
    TTSStepConfig,
    build_kaapi_completion_config,
)
from app.services.llm.chain.utils import DEFAULT_LLM_INSTRUCTIONS
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_config

URL = "/api/v1/llm/chain/sts"


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
def headers(user_api_key: TestAuthContext) -> dict[str, str]:
    return {"X-API-KEY": user_api_key.key}


@pytest.fixture
def mock_start(db: Session, user_api_key: TestAuthContext) -> Iterator[MagicMock]:
    # The route re-reads the job after start_chain_job, so hand back a real row.
    job = JobCrud(db).create(
        job_type=JobType.LLM_CHAIN, project_id=user_api_key.project_id
    )
    with patch("app.api.routes.llm_sts.start_chain_job", return_value=job.id) as mock:
        mock.job = job
        yield mock


def _stored(db: Session, project_id: int, type: str, provider: str, model: str):
    config = create_test_config(
        db,
        project_id=project_id,
        config_blob=ConfigBlob(
            completion=build_kaapi_completion_config(
                provider=provider, type=type, params={"model": model}
            )
        ),
    )
    return {"id": config.id, "version": 1}


def _request(audio_input: AudioInput, **kwargs) -> SpeechToSpeechRequest:
    return SpeechToSpeechRequest(query=QueryParams(input=audio_input), **kwargs)


def _post(client: TestClient, headers: dict, payload: SpeechToSpeechRequest):
    return client.post(URL, json=payload.model_dump(mode="json"), headers=headers)


def _chain_request(mock_start: MagicMock):
    return mock_start.call_args.kwargs["request"]


def _completions(mock_start: MagicMock) -> list:
    return [b.config.blob.completion for b in _chain_request(mock_start).blocks]


class TestDefaults:
    def test_omitted_config_builds_default_blocks(
        self, client, headers, audio_input, mock_start
    ):
        response = _post(client, headers, _request(audio_input))

        assert response.status_code == 200
        body = response.json()["data"]
        assert body["job_id"] == str(mock_start.job.id)
        assert body["status"] == mock_start.job.status.value
        assert body["message"] == "Your response is being generated"

        stt, llm, tts = _completions(mock_start)
        assert stt.provider is None
        assert stt.params == STTLLMParams(
            model=DEFAULT_STT_MODEL, input_language="auto"
        )
        assert llm.provider == "openai"
        assert llm.params == TextLLMParams(
            model=DEFAULT_RAG_MODEL,
            temperature=0.1,
            instructions=DEFAULT_LLM_INSTRUCTIONS,
        )
        assert llm.params.knowledge_base_ids is None
        assert tts.provider is None
        assert tts.params.model == DEFAULT_TTS_MODEL
        assert tts.params.voice == DEFAULT_TTS_VOICE
        assert tts.params.response_format == "ogg"
        assert tts.params.language == "{{detected}}"

    def test_callback_url_changes_message(
        self, client, headers, audio_input, mock_start
    ):
        response = _post(
            client,
            headers,
            _request(audio_input, callback_url="https://example.com/hook"),
        )

        assert response.status_code == 200
        assert response.json()["data"]["message"] == (
            "Your response is being generated and will be delivered via callback."
        )

    def test_block_flags(self, client, headers, audio_input, mock_start):
        _post(
            client,
            headers,
            _request(audio_input, include_provider_raw_response=True),
        )

        blocks = _chain_request(mock_start).blocks
        assert [b.intermediate_callback for b in blocks] == [True, True, False]
        assert [b.include_provider_raw_response for b in blocks] == [True] * 3

    def test_request_metadata_passes_through_unchanged(
        self, client, headers, audio_input, mock_start
    ):
        metadata = {"session": "abc", "turn": 3}
        _post(client, headers, _request(audio_input, request_metadata=metadata))

        assert _chain_request(mock_start).request_metadata == metadata


class TestInlineSteps:
    def test_inline_step_used_as_written(
        self, client, headers, audio_input, mock_start
    ):
        config = SpeechToSpeechConfig(
            llm=LLMStepConfig(
                provider="google",
                params=TextLLMParams(instructions="Reply in one line."),
            )
        )
        _post(client, headers, _request(audio_input, config=config))

        llm = _completions(mock_start)[1]
        assert llm.provider == "google"
        # No merge with endpoint defaults: model stays unset (default would be DEFAULT_RAG_MODEL).
        assert llm.params == TextLLMParams(instructions="Reply in one line.")

    def test_inline_step_without_params_or_provider(
        self, client, headers, audio_input, mock_start
    ):
        config = SpeechToSpeechConfig(
            stt=STTStepConfig(provider="sarvamai"), llm=LLMStepConfig()
        )
        _post(client, headers, _request(audio_input, config=config))

        stt, llm, _ = _completions(mock_start)
        assert stt.provider == "sarvamai"
        assert stt.params == STTLLMParams(model=DEFAULT_SARVAM_STT_MODEL)
        assert llm.provider == "openai"
        assert llm.params == TextLLMParams()

    def test_stored_steps_pass_id_and_version_only(
        self, db, client, headers, audio_input, mock_start, user_api_key
    ):
        pid = user_api_key.project_id
        refs = [
            _stored(db, pid, "stt", "google", DEFAULT_STT_MODEL),
            _stored(db, pid, "text", "openai", "gpt-4o"),
            _stored(db, pid, "tts", "google", DEFAULT_TTS_MODEL),
        ]
        config = SpeechToSpeechConfig(
            stt=STTStepConfig(**refs[0]),
            llm=LLMStepConfig(**refs[1]),
            tts=TTSStepConfig(**refs[2]),
        )
        response = _post(client, headers, _request(audio_input, config=config))

        assert response.status_code == 200, response.json()
        configs = [b.config for b in _chain_request(mock_start).blocks]
        assert [(c.id, c.version, c.blob) for c in configs] == [
            (r["id"], r["version"], None) for r in refs
        ]

    def test_stored_step_with_wrong_type_rejected(
        self, db, client, headers, audio_input, mock_start, user_api_key
    ):
        text_ref = _stored(db, user_api_key.project_id, "text", "openai", "gpt-4o")
        config = SpeechToSpeechConfig(stt=STTStepConfig(**text_ref))
        response = _post(client, headers, _request(audio_input, config=config))

        assert response.status_code == 422
        assert "config.stt" in response.json()["error"]
        mock_start.assert_not_called()

    def test_unknown_stored_config_not_found(
        self, client, headers, audio_input, mock_start
    ):
        config = SpeechToSpeechConfig(tts=TTSStepConfig(id=uuid4(), version=1))
        response = _post(client, headers, _request(audio_input, config=config))

        assert response.status_code == 404
        mock_start.assert_not_called()


class TestTTSLanguage:
    def _tts_language(self, client, headers, audio_input, mock_start, config):
        response = _post(client, headers, _request(audio_input, config=config))
        assert response.status_code == 200, response.json()
        tts_block = _chain_request(mock_start).blocks[2]
        return tts_block.config.blob.completion.params.language

    def test_inherits_concrete_stt_language(
        self, client, headers, audio_input, mock_start
    ):
        config = SpeechToSpeechConfig(
            stt=STTStepConfig(params=STTLLMParams(input_language="hi-IN"))
        )
        assert (
            self._tts_language(client, headers, audio_input, mock_start, config)
            == "hi-IN"
        )

    def test_explicit_tts_language_kept(self, client, headers, audio_input, mock_start):
        config = SpeechToSpeechConfig(
            stt=STTStepConfig(params=STTLLMParams(input_language="hi-IN")),
            tts=TTSStepConfig(params=TTSLLMParams(language="en-IN")),
        )
        assert (
            self._tts_language(client, headers, audio_input, mock_start, config)
            == "en-IN"
        )

    def test_stored_stt_falls_back_to_detected_marker(
        self, db, client, headers, audio_input, mock_start, user_api_key
    ):
        ref = _stored(db, user_api_key.project_id, "stt", "google", DEFAULT_STT_MODEL)
        config = SpeechToSpeechConfig(stt=STTStepConfig(**ref))
        assert (
            self._tts_language(client, headers, audio_input, mock_start, config)
            == "{{detected}}"
        )


class TestValidation:
    def test_text_input_rejected(self, client, headers, mock_start):
        response = client.post(URL, json={"query": {"input": "hello"}}, headers=headers)
        assert response.status_code == 422
        mock_start.assert_not_called()

    def test_list_input_rejected(self, audio_input):
        with pytest.raises(ValidationError, match="single audio input"):
            SpeechToSpeechRequest(query=QueryParams(input=[audio_input]))

    def test_id_without_version_rejected(self):
        with pytest.raises(ValidationError, match="must be set together"):
            STTStepConfig(id=uuid4())

    def test_id_with_provider_rejected(self):
        with pytest.raises(ValidationError, match="not both"):
            TTSStepConfig(id=uuid4(), version=1, provider="google")

    def test_google_gcp_with_knowledge_base_rejected(self):
        with pytest.raises(ValidationError, match="google-gcp"):
            LLMStepConfig(
                provider="google-gcp",
                params=TextLLMParams(knowledge_base_ids=["kb-1"]),
            )

    def test_google_gcp_without_knowledge_base_allowed(self):
        step = LLMStepConfig(provider="google-gcp", params=TextLLMParams())
        assert step.provider == "google-gcp"
