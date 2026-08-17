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
        assert "[GOOGLE_GCP]" in err
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
# Standalone _post / _execute_tts / execute() coverage — not routing-dependent,
# kept unskipped (same pattern as test_tts_wraps_input_in_transcript_tags).
# ---------------------------------------------------------------------------
def _provider() -> GoogleGCPProvider:
    client = GoogleGCPClient(
        api_key="k",
        project_id="p",
        location="us-central1",
        sa_info={"type": "service_account", "project_id": "p"},
        gcs_bucket="test-bucket",
    )
    return GoogleGCPProvider(client=client)


def _tts_config(**params) -> NativeCompletionConfig:
    return NativeCompletionConfig(
        provider="google-native",
        type=CompletionType.TTS,
        params={"model": "gemini-2.5-flash-preview-tts", "voice": "Kore", **params},
    )


@pytest.mark.parametrize(
    "status,expected_snippet",
    [
        (400, "Bad request"),
        (401, "Authentication / permission denied"),
        (404, "Resource not found"),
        (429, "Rate limit / quota exceeded"),
        (500, "Server error"),
        (418, "HTTP error"),
    ],
)
def test_post_http_error_status_branches(status, expected_snippet):
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_err(status, "boom"),
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert expected_snippet in err


def test_post_http_error_uses_google_error_envelope():
    provider = _provider()
    resp_mock = MagicMock()
    resp_mock.ok = False
    resp_mock.status_code = 400
    resp_mock.text = "raw text"
    resp_mock.json.return_value = {
        "error": {"message": "invalid field", "status": "INVALID_ARGUMENT"}
    }
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=resp_mock,
    ):
        _, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert "invalid field" in err
    assert "INVALID_ARGUMENT" in err


def test_post_timeout_returns_clean_message():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        side_effect=requests.Timeout("too slow"),
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "timed out" in err


def test_post_request_exception_returns_clean_message():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        side_effect=requests.RequestException("weird"),
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "request failed" in err


def test_post_non_json_success_response_returns_clean_message():
    provider = _provider()
    resp_mock = MagicMock()
    resp_mock.ok = True
    resp_mock.status_code = 200
    resp_mock.json.side_effect = ValueError("no json")
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=resp_mock,
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "non-JSON success response" in err


def test_execute_tts_rejects_non_string_input():
    provider = _provider()
    resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), [1, 2])
    assert resp is None
    assert "text string as input" in err


def test_execute_tts_rejects_empty_input():
    provider = _provider()
    resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "   ")
    assert resp is None
    assert "text input is empty" in err


def test_execute_tts_missing_audio_data_returns_error():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok({"candidates": [{"content": {"parts": []}}]}),
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "missing audio data" in err


def test_execute_tts_invalid_base64_returns_error():
    provider = _provider()
    bad = {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"data": "not-valid-base64!!"}}]}}
        ]
    }
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(bad),
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "invalid base64 audio" in err


def test_execute_tts_empty_audio_bytes_returns_error():
    provider = _provider()
    empty = {"candidates": [{"content": {"parts": [{"inlineData": {"data": ""}}]}}]}
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(empty),
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "empty audio data" in err


def test_execute_tts_mp3_conversion_success():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ), patch(
        "app.services.llm.providers.google_gcp.convert_pcm_to_mp3",
        return_value=(b"mp3bytes", None),
    ):
        resp, err = provider.execute(
            _tts_config(response_format="mp3"), QueryParams(input="ignored"), "hi"
        )
    assert err is None
    assert resp.response.output.content.mime_type == "audio/mp3"


def test_execute_tts_mp3_conversion_failure_returns_error():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ), patch(
        "app.services.llm.providers.google_gcp.convert_pcm_to_mp3",
        return_value=(None, "ffmpeg missing"),
    ):
        resp, err = provider.execute(
            _tts_config(response_format="mp3"), QueryParams(input="ignored"), "hi"
        )
    assert resp is None
    assert "unable to convert" in err
    assert "ffmpeg missing" in err


def test_execute_tts_ogg_conversion_success():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ), patch(
        "app.services.llm.providers.google_gcp.convert_pcm_to_ogg",
        return_value=(b"oggbytes", None),
    ):
        resp, err = provider.execute(
            _tts_config(response_format="ogg"), QueryParams(input="ignored"), "hi"
        )
    assert err is None
    assert resp.response.output.content.mime_type == "audio/ogg"


def test_execute_tts_unsupported_response_format_falls_back_to_wav():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ):
        resp, err = provider.execute(
            _tts_config(response_format="flac"), QueryParams(input="ignored"), "hi"
        )
    assert err is None
    assert resp.response.output.content.mime_type == "audio/wav"


def test_execute_rejects_unsupported_completion_type():
    provider = _provider()
    config = NativeCompletionConfig(
        provider="google-native",
        type=CompletionType.TEXT,
        params={"model": "gemini-2.5-flash"},
    )
    resp, err = provider.execute(config, QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "Unsupported completion type" in err


def test_execute_tts_language_is_forwarded():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ) as mock_post:
        provider.execute(
            _tts_config(language="en-US"), QueryParams(input="ignored"), "hi"
        )
    speech = mock_post.call_args.kwargs["json"]["generationConfig"]["speechConfig"]
    assert speech["languageCode"] == "en-US"


def test_execute_tts_director_notes_set_system_instruction():
    provider = _provider()
    config = _tts_config(provider_specific={"gemini": {"director_notes": "whisper"}})
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ) as mock_post:
        provider.execute(config, QueryParams(input="ignored"), "hi")
    sent = mock_post.call_args.kwargs["json"]
    assert sent["systemInstruction"]["parts"][0]["text"] == "whisper"


def test_execute_tts_ogg_conversion_failure_returns_error():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(_tts_response()),
    ), patch(
        "app.services.llm.providers.google_gcp.convert_pcm_to_ogg",
        return_value=(None, "codec missing"),
    ):
        resp, err = provider.execute(
            _tts_config(response_format="ogg"), QueryParams(input="ignored"), "hi"
        )
    assert resp is None
    assert "unable to convert" in err
    assert "codec missing" in err


def test_execute_tts_raw_response_included_when_requested():
    provider = _provider()
    raw = _tts_response()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=_mock_http_ok(raw),
    ):
        resp, _ = provider.execute(
            _tts_config(),
            QueryParams(input="ignored"),
            "hi",
            include_provider_raw_response=True,
        )
    assert resp.provider_raw_response == raw


def test_post_connection_error_returns_clean_message():
    provider = _provider()
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        side_effect=requests.ConnectionError("dns boom"),
    ):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "connection failed" in err


def test_post_http_error_falls_back_to_raw_text_when_body_not_json():
    provider = _provider()
    resp_mock = MagicMock()
    resp_mock.ok = False
    resp_mock.status_code = 400
    resp_mock.text = "plain text error"
    resp_mock.json.side_effect = ValueError("not json")
    with patch(
        "app.services.llm.providers.google_gcp.requests.post",
        return_value=resp_mock,
    ):
        _, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert "plain text error" in err


def test_execute_dispatches_stt_validation():
    """Only the dispatch branch + input validation are exercised here."""
    provider = _provider()
    config = NativeCompletionConfig(
        provider="google-native",
        type=CompletionType.STT,
        params={"model": "gemini-2.5-flash"},
    )
    resp, err = provider.execute(
        config, QueryParams(input="ignored"), "/not/an/audioref"
    )
    assert resp is None
    assert "AudioRef input" in err


def test_execute_wraps_type_error():
    provider = _provider()
    with patch.object(provider, "_execute_tts", side_effect=TypeError("bad kwarg")):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "Invalid or unexpected parameter" in err
    assert "bad kwarg" in err


def test_execute_wraps_unexpected_exception():
    provider = _provider()
    with patch.object(provider, "_execute_tts", side_effect=RuntimeError("kaboom")):
        resp, err = provider.execute(_tts_config(), QueryParams(input="ignored"), "hi")
    assert resp is None
    assert "Unexpected error" in err
    assert "kaboom" in err


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
