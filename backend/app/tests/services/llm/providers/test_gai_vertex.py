"""Tests for the Google Vertex AI provider."""

import base64
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.core.audio_utils import AudioRef
from app.models.llm import NativeCompletionConfig, QueryParams
from app.services.llm.providers.gai_vertex import (
    GoogleVertexAIProvider,
    VertexClient,
)


def _stt_response(text: str = "hello world") -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "modelVersion": "gemini-2.5-flash",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }


def _tts_response(pcm_bytes: bytes = b"\x00\x01" * 100) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm",
                                "data": base64.b64encode(pcm_bytes).decode("ascii"),
                            }
                        }
                    ]
                }
            }
        ],
        "modelVersion": "gemini-2.5-flash-preview-tts",
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 0,
            "totalTokenCount": 4,
        },
    }


def _mock_http_ok(json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    resp.json.return_value = json_body
    return resp


def _mock_http_err(status: int = 400, body: str = "bad request") -> MagicMock:
    resp = MagicMock()
    resp.ok = False
    resp.status_code = status
    resp.text = body
    return resp


@pytest.fixture(autouse=True)
def _mock_gcs(monkeypatch):
    """Stub out SM + GCS so STT tests don't touch external services."""
    monkeypatch.setattr(
        "app.services.llm.providers.gai_vertex.get_gcp_service_account",
        lambda **kw: {"type": "service_account", "project_id": "p"},
    )
    monkeypatch.setattr(
        "app.services.llm.providers.gai_vertex.upload_audio_to_gcs",
        lambda *, audio_bytes, bucket_name, sa_info, **kw: f"gs://{bucket_name}/audio/test.wav",
    )


class TestGoogleVertexAIProvider:
    @pytest.fixture
    def client(self) -> VertexClient:
        return VertexClient(
            api_key="k",
            project_id="p",
            location="us-central1",
            gcs_bucket="test-bucket",
        )

    @pytest.fixture
    def provider(self, client) -> GoogleVertexAIProvider:
        return GoogleVertexAIProvider(client=client)

    @pytest.fixture
    def query(self) -> QueryParams:
        return QueryParams(input="ignored")

    @pytest.fixture
    def audio_ref(self) -> AudioRef:
        return AudioRef(bytes_=b"RIFFfake", mime_type="audio/wav")

    @pytest.fixture
    def stt_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-vertex-native",
            type="stt",
            params={"model": "gemini-2.5-flash", "input_language": "auto"},
        )

    @pytest.fixture
    def tts_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-vertex-native",
            type="tts",
            params={"model": "gemini-2.5-flash-preview-tts", "voice": "Kore"},
        )

    # ── create_client ────────────────────────────────────────────────────────
    def test_create_client_requires_all_fields(self):
        with pytest.raises(ValueError, match="project_id, location"):
            GoogleVertexAIProvider.create_client({"api_key": "k"})

    def test_create_client_builds_endpoint(self):
        c = GoogleVertexAIProvider.create_client(
            {"api_key": "k", "project_id": "p", "location": "us-central1"}
        )
        assert "us-central1-aiplatform.googleapis.com" in c.endpoint("m")
        assert "projects/p/locations/us-central1" in c.endpoint("m")
        assert "models/m:generateContent" in c.endpoint("m")

    # ── STT ──────────────────────────────────────────────────────────────────
    def test_stt_happy_path(self, provider, stt_config, query, audio_ref):
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_ok(_stt_response("hi there")),
        ) as mock_post:
            resp, err = provider.execute(stt_config, query, audio_ref)

        assert err is None
        assert resp.response.output.content.value == "hi there"
        assert resp.response.model == "gemini-2.5-flash"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

        kwargs = mock_post.call_args.kwargs
        assert kwargs["params"] == {"key": "k"}
        parts = kwargs["json"]["contents"][0]["parts"]
        assert parts[0]["fileData"]["mimeType"] == "audio/wav"
        assert parts[0]["fileData"]["fileUri"].startswith("gs://test-bucket/")
        assert "Detect the spoken language automatically" in parts[1]["text"]

    def test_stt_rejects_non_audioref_input(self, provider, stt_config, query):
        resp, err = provider.execute(stt_config, query, "/some/path.wav")
        assert resp is None
        assert "AudioRef input" in err

    def test_stt_rejects_unsupported_mime(self, provider, stt_config, query):
        bad = AudioRef(bytes_=b"x", mime_type="audio/xyz")
        resp, err = provider.execute(stt_config, query, bad)
        assert resp is None
        assert "Unsupported audio mime" in err

    def test_stt_gcs_upload_failure_returns_clean_error(
        self, provider, stt_config, query, audio_ref, monkeypatch
    ):
        monkeypatch.setattr(
            "app.services.llm.providers.gai_vertex.upload_audio_to_gcs",
            MagicMock(side_effect=RuntimeError("bucket denied")),
        )
        resp, err = provider.execute(stt_config, query, audio_ref)
        assert resp is None
        assert "Failed to stage audio for Vertex STT" in err
        assert "bucket denied" in err

    def test_stt_http_error_returns_clean_message(
        self, provider, stt_config, query, audio_ref
    ):
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_err(403, "permission denied"),
        ):
            resp, err = provider.execute(stt_config, query, audio_ref)
        assert resp is None
        assert "Vertex AI HTTP 403" in err
        assert "permission denied" in err

    def test_stt_network_error_returns_clean_message(
        self, provider, stt_config, query, audio_ref
    ):
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            side_effect=requests.ConnectionError("dns boom"),
        ):
            resp, err = provider.execute(stt_config, query, audio_ref)
        assert resp is None
        assert "Vertex AI request failed" in err

    def test_stt_missing_transcript_returns_error(
        self, provider, stt_config, query, audio_ref
    ):
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_ok({"candidates": []}),
        ):
            resp, err = provider.execute(stt_config, query, audio_ref)
        assert resp is None
        assert "missing transcript text" in err

    def test_stt_input_language_overrides_prompt(self, provider, query, audio_ref):
        config = NativeCompletionConfig(
            provider="google-vertex-native",
            type="stt",
            params={
                "model": "gemini-2.5-flash",
                "input_language": "hi-IN",
                "output_language": "en-IN",
                "instructions": "be precise",
            },
        )
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_ok(_stt_response()),
        ) as mock_post:
            provider.execute(config, query, audio_ref)

        prompt = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][1]["text"]
        assert prompt.startswith("be precise")
        assert "hi-IN" in prompt
        assert "translate to en-IN" in prompt

    # ── TTS ──────────────────────────────────────────────────────────────────
    def test_tts_happy_path_wav(self, provider, tts_config, query):
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_ok(_tts_response()),
        ) as mock_post:
            resp, err = provider.execute(tts_config, query, "hello")

        assert err is None
        assert resp.response.output.content.format == "base64"
        assert resp.response.output.content.mime_type == "audio/wav"
        decoded = base64.b64decode(resp.response.output.content.value)
        assert decoded[:4] == b"RIFF"

        sent = mock_post.call_args.kwargs["json"]
        assert sent["generationConfig"]["responseModalities"] == ["AUDIO"]
        assert (
            sent["generationConfig"]["speechConfig"]["voiceConfig"][
                "prebuiltVoiceConfig"
            ]["voiceName"]
            == "Kore"
        )

    def test_tts_rejects_non_string_input(self, provider, tts_config, query):
        resp, err = provider.execute(tts_config, query, ["not a string"])
        assert resp is None
        assert "text string as input" in err

    def test_tts_rejects_empty_input(self, provider, tts_config, query):
        resp, err = provider.execute(tts_config, query, "   ")
        assert resp is None
        assert "Text input cannot be empty" in err

    def test_tts_missing_audio_returns_error(self, provider, tts_config, query):
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_ok({"candidates": [{"content": {"parts": []}}]}),
        ):
            resp, err = provider.execute(tts_config, query, "hello")
        assert resp is None
        assert "missing audio data" in err

    def test_tts_language_is_forwarded(self, provider, query):
        config = NativeCompletionConfig(
            provider="google-vertex-native",
            type="tts",
            params={"model": "gemini-2.5-flash-preview-tts", "language": "en-US"},
        )
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_ok(_tts_response()),
        ) as mock_post:
            provider.execute(config, query, "hi")
        speech = mock_post.call_args.kwargs["json"]["generationConfig"]["speechConfig"]
        assert speech["languageCode"] == "en-US"

    # ── execute dispatcher ───────────────────────────────────────────────────
    def test_text_completion_is_rejected(self, provider, query):
        config = NativeCompletionConfig(
            provider="google-vertex-native",
            type="text",
            params={"model": "gemini-2.5-flash"},
        )
        resp, err = provider.execute(config, query, "hello")
        assert resp is None
        assert "does not support completion type 'text'" in err

    def test_raw_response_included_when_requested(
        self, provider, stt_config, query, audio_ref
    ):
        raw = _stt_response()
        with patch(
            "app.services.llm.providers.gai_vertex.requests.post",
            return_value=_mock_http_ok(raw),
        ):
            resp, _ = provider.execute(
                stt_config, query, audio_ref, include_provider_raw_response=True
            )
        assert resp.provider_raw_response == raw
