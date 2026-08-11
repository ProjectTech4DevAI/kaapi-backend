"""Tests for the Google GCP provider."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.core.audio_utils import AudioRef
from app.models.llm import NativeCompletionConfig, QueryParams
from app.services.llm.providers.google_gcp import (
    GoogleGCPProvider,
    GoogleGCPClient,
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


class TestGoogleGCPProvider:
    @pytest.fixture
    def client(self) -> GoogleGCPClient:
        return GoogleGCPClient(
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
        assert "Failed to stage audio for Google GCP STT" in err
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
        assert "[GOOGLE-GCP]" in err
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
        assert "Google GCP connection failed" in err

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
    def test_text_completion_succeeds(self, provider, query):
        config = NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.TEXT,
            params={"model": "gemini-2.5-flash", "instructions": "be brief"},
        )
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=_mock_http_ok(_stt_response("hello back")),
        ) as mock_post:
            resp, err = provider.execute(config, query, "hello")

        assert err is None
        assert resp.response.output.content.value == "hello back"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
        assert payload["systemInstruction"] == {"parts": [{"text": "be brief"}]}

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
    client = GoogleGCPClient(
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
# GoogleGCPClient.endpoint — host changes by location
# ---------------------------------------------------------------------------
class TestGoogleGCPEndpoint:
    def _client(self, location: str) -> GoogleGCPClient:
        return GoogleGCPClient(
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
        mock_settings.GOOGLE_GCP_SA_KEY = ""
        assert _load_platform_sa_info() is None

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_parses_raw_json_string(self, mock_settings):
        sa = self._sample_sa()
        mock_settings.GOOGLE_GCP_SA_KEY = json.dumps(sa)
        assert _load_platform_sa_info() == sa

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_strips_surrounding_whitespace(self, mock_settings):
        """env-var injection often leaves trailing newlines — must still parse."""
        sa = self._sample_sa()
        mock_settings.GOOGLE_GCP_SA_KEY = "\n  " + json.dumps(sa) + "  \n"
        assert _load_platform_sa_info() == sa

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_returns_none_on_malformed_json(self, mock_settings):
        """A JSON-looking but invalid value must not raise — it returns None
        and lets create_client raise the missing-fields ValueError later."""
        mock_settings.GOOGLE_GCP_SA_KEY = "{not valid json"
        assert _load_platform_sa_info() is None

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_non_json_string_returns_none(self, mock_settings):
        """Anything not starting with '{' is treated as non-JSON and ignored —
        this guards against accidentally interpreting a path or sentinel as a key."""
        mock_settings.GOOGLE_GCP_SA_KEY = "/etc/secrets/sa.json"
        assert _load_platform_sa_info() is None


# ---------------------------------------------------------------------------
# create_client — credential precedence (BYOK overrides platform settings)
# ---------------------------------------------------------------------------
class TestCreateClientFallback:
    @patch("app.services.llm.providers.google_gcp.settings")
    def test_byok_overrides_platform_settings(self, mock_settings):
        mock_settings.GOOGLE_GCP_API_KEY = "platform-key"
        mock_settings.GOOGLE_GCP_PROJECT_ID = "platform-proj"
        mock_settings.GOOGLE_GCP_PROJECT_LOCATION = "us-central1"
        mock_settings.GOOGLE_GCP_SA_KEY = ""
        mock_settings.GOOGLE_GCS_AUDIO_BUCKET = "platform-bucket"

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
        mock_settings.GOOGLE_GCP_API_KEY = "platform-key"
        mock_settings.GOOGLE_GCP_PROJECT_ID = "platform-proj"
        mock_settings.GOOGLE_GCP_PROJECT_LOCATION = "us-central1"
        mock_settings.GOOGLE_GCP_SA_KEY = ""
        mock_settings.GOOGLE_GCS_AUDIO_BUCKET = "platform-bucket"

        c = GoogleGCPProvider.create_client({"api_key": "byok-key"})
        assert c.api_key == "byok-key"
        assert c.project_id == "platform-proj"
        assert c.location == "us-central1"

    @patch("app.services.llm.providers.google_gcp.settings")
    def test_missing_everything_raises_value_error(self, mock_settings):
        mock_settings.GOOGLE_GCP_API_KEY = ""
        mock_settings.GOOGLE_GCP_PROJECT_ID = ""
        mock_settings.GOOGLE_GCP_PROJECT_LOCATION = ""
        mock_settings.GOOGLE_GCP_SA_KEY = ""
        mock_settings.GOOGLE_GCS_AUDIO_BUCKET = ""

        with pytest.raises(ValueError) as exc_info:
            GoogleGCPProvider.create_client({})
        msg = str(exc_info.value)
        assert "api_key" in msg
        assert "project_id" in msg
        assert "location" in msg


# ---------------------------------------------------------------------------
# Text-to-text (_execute_text)
# ---------------------------------------------------------------------------
def _text_response(*texts: str, response_id: str = "r-text") -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": t} for t in texts]}}],
        "responseId": response_id,
        "modelVersion": "gemini-2.5-pro",
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 3,
            "totalTokenCount": 10,
        },
    }


class TestGoogleGCPTextToText:
    @pytest.fixture
    def provider(self) -> GoogleGCPProvider:
        return GoogleGCPProvider(
            client=GoogleGCPClient(api_key="k", project_id="p", location="global")
        )

    @pytest.fixture
    def query(self) -> QueryParams:
        return QueryParams(input="ignored")

    def _config(self, **params) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-gcp-native",
            type=CompletionType.TEXT,
            params=params,
        )

    def _run(self, provider, query, config, resolved_input, body, **kw):
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=_mock_http_ok(body),
        ) as mock_post:
            resp, err = provider.execute(config, query, resolved_input, **kw)
        return resp, err, mock_post

    def test_defaults_model_and_omits_optional_config(self, provider, query):
        resp, err, mock_post = self._run(
            provider, query, self._config(), "hi", _text_response("ok")
        )
        assert err is None
        assert "models/gemini-2.5-pro:generateContent" in mock_post.call_args.args[0]
        payload = mock_post.call_args.kwargs["json"]
        assert "generationConfig" not in payload
        assert "systemInstruction" not in payload

    def test_generation_config_forwarded(self, provider, query):
        config = self._config(
            model="gemini-2.5-pro",
            temperature=0.3,
            max_output_tokens=256,
            reasoning="low",
        )
        _, err, mock_post = self._run(
            provider, query, config, "hi", _text_response("ok")
        )
        assert err is None
        assert mock_post.call_args.kwargs["json"]["generationConfig"] == {
            "temperature": 0.3,
            "maxOutputTokens": 256,
            "thinkingConfig": {"includeThoughts": False, "thinkingLevel": "low"},
        }

    def test_multimodal_input_maps_to_rest_parts(self, provider, query):
        from app.models.llm import ImageContent, PDFContent, TextContent
        from app.services.llm.providers.base import MultiModalInput

        multimodal = MultiModalInput(
            parts=[
                TextContent(value="describe this"),
                ImageContent(format="base64", value="aW1n", mime_type="image/png"),
                PDFContent(
                    format="url",
                    value="https://example.org/doc.pdf",
                    mime_type="application/pdf",
                ),
            ]
        )
        _, err, mock_post = self._run(
            provider, query, self._config(), multimodal, _text_response("ok")
        )
        assert err is None
        parts = mock_post.call_args.kwargs["json"]["contents"][0]["parts"]
        assert parts == [
            {"text": "describe this"},
            {"inlineData": {"data": "aW1n", "mimeType": "image/png"}},
            {
                "fileData": {
                    "fileUri": "https://example.org/doc.pdf",
                    "mimeType": "application/pdf",
                }
            },
        ]

    def test_list_of_parts_input(self, provider, query):
        from app.models.llm import TextContent

        _, err, mock_post = self._run(
            provider,
            query,
            self._config(),
            [TextContent(value="a"), TextContent(value="b")],
            _text_response("ok"),
        )
        assert err is None
        parts = mock_post.call_args.kwargs["json"]["contents"][0]["parts"]
        assert parts == [{"text": "a"}, {"text": "b"}]

    def test_multiple_candidate_parts_are_joined(self, provider, query):
        resp, err, _ = self._run(
            provider, query, self._config(), "hi", _text_response("foo", "bar")
        )
        assert err is None
        assert resp.response.output.content.value == "foobar"

    def test_knowledge_base_ids_rejected_without_http_call(self, provider, query):
        config = self._config(knowledge_base_ids=["stores/kb1"])
        resp, err, mock_post = self._run(
            provider, query, config, "hi", _text_response("ok")
        )
        assert resp is None
        assert "knowledge_base_ids" in err
        assert "google-aistudio" in err
        mock_post.assert_not_called()

    def test_blocked_response_reports_reasons(self, provider, query):
        body = {
            "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
            "promptFeedback": {"blockReason": "SAFETY"},
            "responseId": "r-blocked",
        }
        resp, err, _ = self._run(provider, query, self._config(), "hi", body)
        assert resp is None
        assert "[GOOGLE-GCP]" in err
        assert "finish_reason=SAFETY" in err
        assert "block_reason=SAFETY" in err

    def test_raw_response_included_when_requested(self, provider, query):
        body = _text_response("ok")
        resp, err, _ = self._run(
            provider,
            query,
            self._config(),
            "hi",
            body,
            include_provider_raw_response=True,
        )
        assert err is None
        assert resp.provider_raw_response == body

    def test_usage_extracted(self, provider, query):
        resp, err, _ = self._run(
            provider, query, self._config(), "hi", _text_response("ok")
        )
        assert err is None
        assert resp.usage.input_tokens == 7
        assert resp.usage.output_tokens == 3
        assert resp.usage.total_tokens == 10
        assert resp.response.provider_response_id == "r-text"


# ---------------------------------------------------------------------------
# _post error-status messages
# ---------------------------------------------------------------------------
def _mock_http_err_json(status: int, message: str, google_status: str) -> MagicMock:
    resp = MagicMock()
    resp.ok = False
    resp.status_code = status
    resp.text = message
    resp.json.return_value = {"error": {"message": message, "status": google_status}}
    return resp


class TestGoogleGCPPostErrors:
    @pytest.fixture
    def provider(self) -> GoogleGCPProvider:
        return GoogleGCPProvider(
            client=GoogleGCPClient(api_key="k", project_id="p", location="global")
        )

    def _call(self, provider, response=None, side_effect=None):
        config = NativeCompletionConfig(
            provider="google-gcp-native",
            type=CompletionType.TEXT,
            params={"model": "gemini-2.5-pro"},
        )
        with patch(
            "app.services.llm.providers.google_gcp.requests.post",
            return_value=response,
            side_effect=side_effect,
        ):
            return provider.execute(config, QueryParams(input="x"), "hi")

    @pytest.mark.parametrize(
        "status,google_status,expected",
        [
            (400, "INVALID_ARGUMENT", "Bad request"),
            (401, "UNAUTHENTICATED", "Authentication / permission denied"),
            (404, "NOT_FOUND", "Resource not found"),
            (429, "RESOURCE_EXHAUSTED", "Rate limit / quota exceeded"),
            (503, "UNAVAILABLE", "Server error"),
            (418, "TEAPOT", "HTTP error"),
        ],
    )
    def test_status_code_messages(self, provider, status, google_status, expected):
        resp, err = self._call(
            provider, response=_mock_http_err_json(status, "boom", google_status)
        )
        assert resp is None
        assert err.startswith("[GOOGLE-GCP]")
        assert expected in err
        assert str(status) in err
        assert google_status in err
        assert "boom" in err

    def test_404_names_the_model(self, provider):
        _, err = self._call(
            provider, response=_mock_http_err_json(404, "gone", "NOT_FOUND")
        )
        assert "gemini-2.5-pro" in err

    def test_non_json_error_body_falls_back_to_text(self, provider):
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 400
        resp.text = "<html>plain error</html>"
        resp.json.side_effect = ValueError("not json")
        _, err = self._call(provider, response=resp)
        assert "<html>plain error</html>" in err

    def test_non_json_success_body_returns_error(self, provider):
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        out, err = self._call(provider, response=resp)
        assert out is None
        assert "[GOOGLE-GCP] Returned a non-JSON success response" in err

    def test_timeout_message(self, provider):
        out, err = self._call(provider, side_effect=requests.Timeout("slow"))
        assert out is None
        assert "timed out" in err
        assert "Timeout" in err

    def test_generic_request_exception_message(self, provider):
        out, err = self._call(provider, side_effect=requests.RequestException("weird"))
        assert out is None
        assert "Google GCP request failed" in err
        assert "weird" in err
