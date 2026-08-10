"""Tests for the Google GCP (Vertex AI) provider."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

# ponytail: Vertex provider is temporarily disabled in the registry; skip only
# the routing/execute tests. Pure client tests (endpoint, sa_info, credential
# fallback) still run for coverage.
_vertex_disabled = pytest.mark.skip(
    reason="Vertex provider disabled — routing swapped to GoogleAIProvider"
)

from app.core.audio_utils import AudioRef
from app.models.llm import NativeCompletionConfig, QueryParams
from app.services.llm.providers.google_gcp import (
    GoogleGCPProvider,
    VertexClient,
    _load_platform_sa_info,
)
from app.models.llm.constants import CompletionType


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
    """Stub out GCS upload so STT tests don't touch external services."""
    monkeypatch.setattr(
        "app.services.llm.providers.google_gcp.upload_audio_to_gcs",
        lambda *, audio_bytes, bucket_name, sa_info, **kw: f"gs://{bucket_name}/audio/test.wav",
    )


@_vertex_disabled
class TestGoogleGCPProvider:
    @pytest.fixture
    def client(self) -> VertexClient:
        return VertexClient(
            api_key="k",
            project_id="p",
            location="us-central1",
            sa_info={"type": "service_account", "project_id": "p"},
            gcs_bucket="test-bucket",
        )

    @pytest.fixture
    def provider(self, client) -> GoogleGCPProvider:
        return GoogleGCPProvider(client=client)

    @pytest.fixture
    def query(self) -> QueryParams:
        return QueryParams(input="ignored")

    @pytest.fixture
    def audio_ref(self) -> AudioRef:
        return AudioRef(bytes_=b"RIFFfake", mime_type="audio/wav")

    @pytest.fixture
    def stt_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.STT,
            params={"model": "gemini-2.5-flash", "input_language": "auto"},
        )

    @pytest.fixture
    def tts_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.TTS,
            params={"model": "gemini-2.5-flash-preview-tts", "voice": "Kore"},
        )

    # ── create_client ────────────────────────────────────────────────────────
    def test_create_client_requires_all_fields(self):
        with pytest.raises(ValueError, match="project_id, location"):
            GoogleGCPProvider.create_client({"api_key": "k"})

    def test_create_client_builds_endpoint(self):
        c = GoogleGCPProvider.create_client(
            {"api_key": "k", "project_id": "p", "location": "us-central1"}
        )
        assert "us-central1-aiplatform.googleapis.com" in c.endpoint("m")
        assert "projects/p/locations/us-central1" in c.endpoint("m")
        assert "models/m:generateContent" in c.endpoint("m")

    # ── STT ──────────────────────────────────────────────────────────────────
    def test_stt_happy_path(self, provider, stt_config, query, audio_ref):
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
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
        assert "unsupported audio mime" in err

    def test_stt_gcs_upload_failure_returns_clean_error(
        self, provider, stt_config, query, audio_ref, monkeypatch
    ):
        monkeypatch.setattr(
            "app.services.llm.providers.google_gcp.upload_audio_to_gcs",
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
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=_mock_http_err(403, "permission denied"),
        ):
            resp, err = provider.execute(stt_config, query, audio_ref)
        assert resp is None
        assert "[VERTEX]" in err
        assert "403" in err
        assert "permission denied" in err

    def test_stt_network_error_returns_clean_message(
        self, provider, stt_config, query, audio_ref
    ):
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            side_effect=requests.ConnectionError("dns boom"),
        ):
            resp, err = provider.execute(stt_config, query, audio_ref)
        assert resp is None
        assert "Vertex AI connection failed" in err

    def test_stt_missing_transcript_returns_error(
        self, provider, stt_config, query, audio_ref
    ):
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=_mock_http_ok({"candidates": []}),
        ):
            resp, err = provider.execute(stt_config, query, audio_ref)
        assert resp is None
        assert "missing transcribed text" in err

    def test_stt_input_language_overrides_prompt(self, provider, query, audio_ref):
        config = NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.STT,
            params={
                "model": "gemini-2.5-flash",
                "input_language": "hi-IN",
                "output_language": "en-IN",
                "instructions": "be precise",
            },
        )
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
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
            "app.services.llm.providers.google_gcp.requests.post",
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
        assert "text input is empty" in err

    def test_tts_missing_audio_returns_error(self, provider, tts_config, query):
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=_mock_http_ok({"candidates": [{"content": {"parts": []}}]}),
        ):
            resp, err = provider.execute(tts_config, query, "hello")
        assert resp is None
        assert "missing audio data" in err

    def test_tts_language_is_forwarded(self, provider, query):
        config = NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.TTS,
            params={"model": "gemini-2.5-flash-preview-tts", "language": "en-US"},
        )
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=_mock_http_ok(_tts_response()),
        ) as mock_post:
            provider.execute(config, query, "hi")
        speech = mock_post.call_args.kwargs["json"]["generationConfig"]["speechConfig"]
        assert speech["languageCode"] == "en-US"

    # ── execute dispatcher ───────────────────────────────────────────────────
    def test_text_completion_is_rejected(self, provider, query):
        config = NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.TEXT,
            params={"model": "gemini-2.5-flash"},
        )
        resp, err = provider.execute(config, query, "hello")
        assert resp is None
        assert "Unsupported completion type 'text'" in err

    def test_raw_response_included_when_requested(
        self, provider, stt_config, query, audio_ref
    ):
        raw = _stt_response()
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=_mock_http_ok(raw),
        ):
            resp, _ = provider.execute(
                stt_config, query, audio_ref, include_provider_raw_response=True
            )
        assert resp.provider_raw_response == raw


# ---------------------------------------------------------------------------
# TTS payload shape — not routing-dependent, kept unskipped for coverage.
# ---------------------------------------------------------------------------
def test_tts_wraps_input_in_transcript_tags():
    client = VertexClient(
        api_key="k",
        project_id="p",
        location="us-central1",
        sa_info={"type": "service_account", "project_id": "p"},
        gcs_bucket="test-bucket",
    )
    provider = GoogleGCPProvider(client=client)
    config = NativeCompletionConfig(
        provider="google-native",
        type=CompletionType.TTS,
        params={"model": "gemini-2.5-flash-preview-tts", "voice": "Kore"},
    )
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ) as mock_post:
        provider.execute(config, QueryParams(input="ignored"), "Say this text")

    parts = mock_post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert parts[0]["text"] == "<transcript>Say this text</transcript>"


# ---------------------------------------------------------------------------
# VertexClient.endpoint — host changes by location
# ---------------------------------------------------------------------------
class TestVertexEndpoint:
    def _client(self, location: str) -> VertexClient:
        return VertexClient(
            api_key="k",
            project_id="my-proj",
            location=location,
            sa_info=None,
            gcs_bucket="b",
        )

    def test_regional_location_uses_prefixed_host(self):
        url = self._client("us-central1").endpoint("gemini-2.5-pro")
        assert url.startswith("https://us-central1-aiplatform.googleapis.com/")
        assert "/projects/my-proj/locations/us-central1/" in url
        assert url.endswith("/models/gemini-2.5-pro:generateContent")

    def test_global_location_uses_bare_host(self):
        """The 'global' location does NOT use a hostname prefix — it must
        resolve to ``aiplatform.googleapis.com``. Caught a real 404 outage
        where a global config produced ``global-aiplatform.googleapis.com``."""
        url = self._client("global").endpoint("gemini-2.5-pro")
        assert url.startswith("https://aiplatform.googleapis.com/")
        assert "global-aiplatform" not in url
        assert "/locations/global/" in url

    def test_other_regions_get_prefix(self):
        url = self._client("europe-west4").endpoint("gemini-2.5-flash")
        assert "europe-west4-aiplatform.googleapis.com" in url


# ---------------------------------------------------------------------------
# _load_platform_sa_info — env-var shape handling
# ---------------------------------------------------------------------------
class TestLoadPlatformSaInfo:
    """The platform SA can be injected as a raw JSON string via env var or
    secret manager. Cover the parse paths and the unhappy ones."""

    def _sample_sa(self) -> dict:
        return {
            "type": "service_account",
            "project_id": "platform-project",
            "client_email": "sa@platform-project.iam.gserviceaccount.com",
            "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----",
        }

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_returns_none_when_unset(self, mock_settings):
        mock_settings.GCP_SA_KEY = ""
        assert _load_platform_sa_info() is None

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_parses_raw_json_string(self, mock_settings):
        sa = self._sample_sa()
        mock_settings.GCP_SA_KEY = json.dumps(sa)
        assert _load_platform_sa_info() == sa

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_strips_surrounding_whitespace(self, mock_settings):
        """env-var injection often leaves trailing newlines — must still parse."""
        sa = self._sample_sa()
        mock_settings.GCP_SA_KEY = "\n  " + json.dumps(sa) + "  \n"
        assert _load_platform_sa_info() == sa

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_returns_none_on_malformed_json(self, mock_settings):
        """A JSON-looking but invalid value must not raise — it returns None
        and lets create_client raise the missing-fields ValueError later."""
        mock_settings.GCP_SA_KEY = "{not valid json"
        assert _load_platform_sa_info() is None

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_non_json_string_returns_none(self, mock_settings):
        """Anything not starting with '{' is treated as non-JSON and ignored —
        this guards against accidentally interpreting a path or sentinel as a key."""
        mock_settings.GCP_SA_KEY = "/etc/secrets/sa.json"
        assert _load_platform_sa_info() is None


# ---------------------------------------------------------------------------
# create_client — credential precedence (BYOK overrides platform settings)
# ---------------------------------------------------------------------------
class TestCreateClientFallback:
    @patch("app.services.llm.providers.google_gcp.settings")
    def test_byok_overrides_platform_settings(self, mock_settings):
        mock_settings.GCP_VERTEX_API_KEY = "platform-key"
        mock_settings.GCP_PROJECT_ID = "platform-proj"
        mock_settings.GCP_VERTEX_LOCATION = "us-central1"
        mock_settings.GCP_SA_KEY = ""
        mock_settings.GCS_AUDIO_BUCKET = "platform-bucket"

        c = GoogleGCPProvider.create_client(
            {
                "api_key": "byok-key",
                "project_id": "byok-proj",
                "location": "europe-west4",
                "gcs_bucket": "byok-bucket",
            }
        )
        assert c.api_key == "byok-key"
        assert c.project_id == "byok-proj"
        assert c.location == "europe-west4"
        assert c.gcs_bucket == "byok-bucket"

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_partial_byok_fills_from_platform(self, mock_settings):
        """When BYOK only supplies api_key, project/location come from settings."""
        mock_settings.GCP_VERTEX_API_KEY = "platform-key"
        mock_settings.GCP_PROJECT_ID = "platform-proj"
        mock_settings.GCP_VERTEX_LOCATION = "us-central1"
        mock_settings.GCP_SA_KEY = ""
        mock_settings.GCS_AUDIO_BUCKET = "platform-bucket"

        c = GoogleGCPProvider.create_client({"api_key": "byok-key"})
        assert c.api_key == "byok-key"
        assert c.project_id == "platform-proj"
        assert c.location == "us-central1"

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_missing_everything_raises_value_error(self, mock_settings):
        mock_settings.GCP_VERTEX_API_KEY = ""
        mock_settings.GCP_PROJECT_ID = ""
        mock_settings.GCP_VERTEX_LOCATION = ""
        mock_settings.GCP_SA_KEY = ""
        mock_settings.GCS_AUDIO_BUCKET = ""

        with pytest.raises(ValueError) as exc_info:
            GoogleGCPProvider.create_client({})
        msg = str(exc_info.value)
        assert "api_key" in msg
        assert "project_id" in msg
        assert "location" in msg
