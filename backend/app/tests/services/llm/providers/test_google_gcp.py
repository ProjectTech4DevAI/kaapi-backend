"""Tests for the Google GCP (Vertex) provider."""

import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.auth import exceptions as google_auth_errors
from google.genai._gaos.lib import compat_errors
from google.genai.interactions import Interaction

from app.core.audio_utils import AudioRef, pcm_to_wav
from app.core.config import settings
from app.models.llm import (
    ImageContent,
    NativeCompletionConfig,
    PDFContent,
    QueryParams,
    TextContent,
)
from app.models.llm.constants import CompletionType
from app.services.llm.providers.base import MultiModalInput
from app.services.llm.providers.google_gcp import (
    GoogleGCPClient,
    GoogleGCPProvider,
    _parse_sa_info,
)

GCP_MODULE = "app.services.llm.providers.google_gcp"
SAMPLE_PCM_BYTES = b"\x00\x01" * 100
VERTEX_REQUEST = httpx.Request("POST", "https://aiplatform.googleapis.com")
SA_INFO = {"type": "service_account", "project_id": "sa-proj"}


def make_interaction(
    *,
    text: str | None = "hello world",
    audio: bytes | None = None,
    audio_b64: str | None = None,
    interaction_id: str | None = "int_123",
    model: str | None = "gemini-2.5-flash",
    status: str = "completed",
) -> Interaction:
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    if audio is not None:
        audio_b64 = base64.b64encode(audio).decode("ascii")
    if audio_b64 is not None:
        content.append({"type": "audio", "data": audio_b64, "mime_type": "audio/l16"})
    payload: dict[str, Any] = {
        "status": status,
        "model": model,
        "steps": [{"type": "model_output", "content": content}],
        "usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_thought_tokens": 2,
            "total_tokens": 17,
        },
    }
    if interaction_id is not None:
        payload["id"] = interaction_id
    return Interaction.model_validate(payload)


def tts_interaction(audio: bytes = SAMPLE_PCM_BYTES) -> Interaction:
    return make_interaction(text=None, audio=audio, model="gemini-2.5-flash-tts")


def create_kwargs(client: GoogleGCPClient) -> dict[str, Any]:
    client.genai.interactions.create.assert_called_once()
    return client.genai.interactions.create.call_args.kwargs


def status_error(status_code: int, message: str) -> Exception:
    body = {"error": {"code": status_code, "message": message}}
    response = httpx.Response(status_code, json=body, request=VERTEX_REQUEST)
    return compat_errors.APIError.generate(status_code, body, message, response)


def platform_settings(**overrides: Any):
    values: dict[str, Any] = {
        "GCP_VERTEX_API_KEY": "",
        "GCP_SA_KEY": "",
        "GCP_PROJECT_ID": "",
        "GCP_VERTEX_LOCATION": "",
        "GCS_AUDIO_BUCKET": "",
    }
    values.update(overrides)
    return patch.multiple(settings, **values)


@pytest.fixture
def client() -> GoogleGCPClient:
    return GoogleGCPClient(
        genai_client=MagicMock(),
        project_id="p",
        sa_info=SA_INFO,
        gcs_bucket="test-bucket",
    )


@pytest.fixture
def provider(client: GoogleGCPClient) -> GoogleGCPProvider:
    return GoogleGCPProvider(client=client)


@pytest.fixture
def query() -> QueryParams:
    return QueryParams(input="ignored")


@pytest.fixture
def upload_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    # GCS is an external boundary; STT stages audio there before calling Vertex.
    mock = MagicMock(return_value="gs://test-bucket/audio/test.wav")
    monkeypatch.setattr(f"{GCP_MODULE}.upload_audio_to_gcs", mock)
    return mock


def _text_config(**params: Any) -> NativeCompletionConfig:
    return NativeCompletionConfig(
        provider="google-gcp-native",
        type=CompletionType.TEXT,
        params={"model": "gemini-2.5-flash", **params},
    )


def _stt_config(**params: Any) -> NativeCompletionConfig:
    return NativeCompletionConfig(
        provider="google-gcp-native",
        type=CompletionType.STT,
        params={"model": "gemini-2.5-flash", "input_language": "auto", **params},
    )


def _tts_config(**params: Any) -> NativeCompletionConfig:
    return NativeCompletionConfig(
        provider="google-gcp-native",
        type=CompletionType.TTS,
        params={"model": "gemini-2.5-flash-tts", "voice": "Kore", **params},
    )


class TestParseSaInfo:
    def test_dict_passes_through(self):
        assert _parse_sa_info(SA_INFO) == SA_INFO

    def test_empty_dict_is_none(self):
        assert _parse_sa_info({}) is None

    def test_parses_json_string(self):
        assert _parse_sa_info(json.dumps(SA_INFO)) == SA_INFO

    def test_parses_json_string_with_surrounding_whitespace(self):
        # env-var injection often leaves trailing newlines
        assert _parse_sa_info("\n  " + json.dumps(SA_INFO) + "  \n") == SA_INFO

    def test_malformed_json_returns_none(self):
        assert _parse_sa_info("{not valid json") is None

    def test_non_json_string_returns_none(self):
        assert _parse_sa_info("/etc/secrets/sa.json") is None

    @pytest.mark.parametrize("value", [None, "", 42])
    def test_unset_or_other_types_return_none(self, value):
        assert _parse_sa_info(value) is None


class TestCreateClient:
    def test_service_account_path_builds_vertex_client_with_credentials(self):
        sa_creds = object()
        with (
            platform_settings(GCP_VERTEX_API_KEY="platform-key"),
            patch(f"{GCP_MODULE}.build_gcp_sa_credentials", return_value=sa_creds),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            c = GoogleGCPProvider.create_client(
                {
                    "sa_key": SA_INFO,
                    "project_id": "byok-proj",
                    "location": "europe-west4",
                    "gcs_bucket": "byok-bucket",
                }
            )

        kwargs = mock_client.call_args.kwargs
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "byok-proj"
        assert kwargs["location"] == "europe-west4"
        assert kwargs["credentials"] is sa_creds
        # The SDK rejects credentials + api_key together.
        assert "api_key" not in kwargs
        assert c.genai is mock_client.return_value
        assert c.project_id == "byok-proj"
        assert c.sa_info == SA_INFO
        assert c.gcs_bucket == "byok-bucket"

    def test_project_id_falls_back_to_sa_info(self):
        with (
            platform_settings(),
            patch(f"{GCP_MODULE}.build_gcp_sa_credentials", return_value=object()),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            c = GoogleGCPProvider.create_client({"sa_key": SA_INFO})

        assert mock_client.call_args.kwargs["project"] == "sa-proj"
        assert c.project_id == "sa-proj"

    def test_location_defaults_to_global(self):
        with (
            platform_settings(),
            patch(f"{GCP_MODULE}.build_gcp_sa_credentials", return_value=object()),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            GoogleGCPProvider.create_client({"sa_key": SA_INFO})

        assert mock_client.call_args.kwargs["location"] == "global"

    def test_platform_sa_json_string_is_used(self):
        with (
            platform_settings(GCP_SA_KEY=json.dumps(SA_INFO)),
            patch(f"{GCP_MODULE}.build_gcp_sa_credentials", return_value=object()),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            c = GoogleGCPProvider.create_client({})

        assert "credentials" in mock_client.call_args.kwargs
        assert c.sa_info == SA_INFO

    def test_malformed_sa_falls_back_to_api_key(self):
        with (
            platform_settings(),
            patch(
                f"{GCP_MODULE}.build_gcp_sa_credentials",
                side_effect=ValueError("no private_key"),
            ),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            c = GoogleGCPProvider.create_client({"sa_key": SA_INFO, "api_key": "k"})

        kwargs = mock_client.call_args.kwargs
        assert kwargs["api_key"] == "k"
        assert "credentials" not in kwargs
        assert c.sa_info is None

    def test_malformed_sa_without_api_key_raises(self):
        with (
            platform_settings(),
            patch(
                f"{GCP_MODULE}.build_gcp_sa_credentials",
                side_effect=KeyError("client_email"),
            ),
            patch(f"{GCP_MODULE}.genai.Client"),
            pytest.raises(ValueError, match="Google GCP credentials missing"),
        ):
            GoogleGCPProvider.create_client({"sa_key": SA_INFO})

    def test_sa_without_project_id_uses_api_key(self):
        sa_no_project = {"type": "service_account"}
        with (
            platform_settings(),
            patch(f"{GCP_MODULE}.build_gcp_sa_credentials", return_value=object()),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            c = GoogleGCPProvider.create_client(
                {"sa_key": sa_no_project, "api_key": "k"}
            )

        assert mock_client.call_args.kwargs["api_key"] == "k"
        assert "credentials" not in mock_client.call_args.kwargs
        assert c.sa_info == sa_no_project

    def test_byok_overrides_platform_settings(self):
        with (
            platform_settings(
                GCP_VERTEX_API_KEY="platform-key",
                GCP_PROJECT_ID="platform-proj",
                GCS_AUDIO_BUCKET="platform-bucket",
            ),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            c = GoogleGCPProvider.create_client(
                {
                    "api_key": "byok-key",
                    "project_id": "byok-proj",
                    "gcs_bucket": "byok-bucket",
                }
            )

        assert mock_client.call_args.kwargs["api_key"] == "byok-key"
        assert c.project_id == "byok-proj"
        assert c.gcs_bucket == "byok-bucket"

    def test_partial_byok_fills_from_platform(self):
        with (
            platform_settings(
                GCP_VERTEX_API_KEY="platform-key",
                GCP_PROJECT_ID="platform-proj",
                GCS_AUDIO_BUCKET="platform-bucket",
            ),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
        ):
            c = GoogleGCPProvider.create_client({"api_key": "byok-key"})

        assert mock_client.call_args.kwargs["api_key"] == "byok-key"
        assert c.project_id == "platform-proj"
        assert c.gcs_bucket == "platform-bucket"

    def test_missing_everything_raises_value_error(self):
        with (
            platform_settings(),
            patch(f"{GCP_MODULE}.genai.Client") as mock_client,
            pytest.raises(ValueError) as exc_info,
        ):
            GoogleGCPProvider.create_client({})

        msg = str(exc_info.value)
        assert "sa_key" in msg
        assert "api_key" in msg
        mock_client.assert_not_called()


class TestSTT:
    @pytest.fixture
    def audio_ref(self) -> AudioRef:
        return AudioRef(bytes_=b"RIFFfake", mime_type="audio/wav")

    def test_happy_path(self, provider, client, query, audio_ref, upload_mock):
        client.genai.interactions.create.return_value = make_interaction(
            text="  hi there \n"
        )

        resp, err = provider.execute(_stt_config(), query, audio_ref)

        assert err is None
        assert resp.response.output.content.value == "hi there"
        assert resp.response.model == "gemini-2.5-flash"
        assert resp.response.provider_response_id == "int_123"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.usage.total_tokens == 17

        upload_kwargs = upload_mock.call_args.kwargs
        assert upload_kwargs["audio_bytes"] == b"RIFFfake"
        assert upload_kwargs["bucket_name"] == "test-bucket"
        assert upload_kwargs["content_type"] == "audio/wav"

        kwargs = create_kwargs(client)
        content = kwargs["input"][0]["content"]
        assert kwargs["input"][0]["type"] == "user_input"
        assert content[0]["type"] == "text"
        assert "Detect the spoken language automatically" in content[0]["text"]
        assert content[1] == {
            "type": "audio",
            "uri": "gs://test-bucket/audio/test.wav",
            "mime_type": "audio/wav",
        }

    def test_rejects_non_audioref_input(self, provider, client, query):
        resp, err = provider.execute(_stt_config(), query, "/some/path.wav")
        assert resp is None
        assert "AudioRef input" in err
        client.genai.interactions.create.assert_not_called()

    def test_rejects_unsupported_mime(self, provider, query):
        bad = AudioRef(bytes_=b"x", mime_type="audio/xyz")
        resp, err = provider.execute(_stt_config(), query, bad)
        assert resp is None
        assert "unsupported audio mime" in err

    @pytest.mark.parametrize(
        ("sa_info", "gcs_bucket"), [(None, "test-bucket"), (SA_INFO, None)]
    )
    def test_missing_sa_or_bucket_returns_staging_error(
        self, query, audio_ref, upload_mock, sa_info, gcs_bucket
    ):
        provider = GoogleGCPProvider(
            client=GoogleGCPClient(
                genai_client=MagicMock(),
                project_id="p",
                sa_info=sa_info,
                gcs_bucket=gcs_bucket,
            )
        )
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert "Google GCP STT staging failed" in err
        upload_mock.assert_not_called()

    def test_gcs_upload_failure_returns_clean_error(
        self, provider, client, query, audio_ref, monkeypatch
    ):
        monkeypatch.setattr(
            f"{GCP_MODULE}.upload_audio_to_gcs",
            MagicMock(side_effect=RuntimeError("bucket denied")),
        )
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert "Failed to stage audio for Google GCP STT" in err
        assert "bucket denied" in err
        client.genai.interactions.create.assert_not_called()

    def test_http_error_returns_clean_message(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.side_effect = status_error(
            403, "permission denied"
        )
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert err.startswith("[GOOGLE_GCP]")
        assert "403" in err
        assert "permission denied" in err

    def test_network_error_returns_clean_message(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.side_effect = compat_errors.APIConnectionError(
            request=VERTEX_REQUEST
        )
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert "Google GCP connection failed" in err

    def test_missing_transcript_returns_error(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction(text=None)
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] STT response is missing transcribed text")

    def test_non_completed_status_returns_error(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction(
            text="partial", status="failed"
        )
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert "missing transcribed text" in err
        assert "status=failed" in err

    def test_input_language_and_instructions_shape_prompt(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction()
        config = _stt_config(
            input_language="hi-IN",
            output_language="en-IN",
            instructions="be precise",
        )
        resp, err = provider.execute(config, query, audio_ref)

        assert err is None
        assert resp.response.output.content.language_code == "en-IN"
        prompt = create_kwargs(client)["input"][0]["content"][0]["text"]
        assert prompt.startswith("be precise")
        assert "hi-IN" in prompt
        assert "translate to en-IN" in prompt

    def test_defaults_thinking_level_and_max_tokens_and_drops_temperature(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction()
        resp, err = provider.execute(_stt_config(temperature=0.3), query, audio_ref)
        assert err is None
        assert create_kwargs(client)["generation_config"] == {
            "thinking_level": "low",
            "max_output_tokens": 2048,
        }

    def test_explicit_thinking_level_and_max_tokens_override_defaults(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction()
        provider.execute(
            _stt_config(thinking_level="high", max_output_tokens=512),
            query,
            audio_ref,
        )
        assert create_kwargs(client)["generation_config"] == {
            "thinking_level": "high",
            "max_output_tokens": 512,
        }

    def test_empty_interaction_id_gets_generated_response_id(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction(
            interaction_id=None
        )
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert err is None
        assert resp.response.provider_response_id.startswith("google-gcp-")

    def test_raw_response_included_when_requested(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction()
        resp, _ = provider.execute(
            _stt_config(), query, audio_ref, include_provider_raw_response=True
        )
        assert resp.provider_raw_response["id"] == "int_123"
        assert resp.provider_raw_response["status"] == "completed"

    def test_raw_response_omitted_by_default(
        self, provider, client, query, audio_ref, upload_mock
    ):
        client.genai.interactions.create.return_value = make_interaction()
        resp, _ = provider.execute(_stt_config(), query, audio_ref)
        assert resp.provider_raw_response is None


class TestTTS:
    def test_happy_path_wav(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()

        resp, err = provider.execute(_tts_config(), query, "hello")

        assert err is None
        content = resp.response.output.content
        assert content.format == "base64"
        assert content.mime_type == "audio/wav"
        assert base64.b64decode(content.value) == pcm_to_wav(SAMPLE_PCM_BYTES)
        assert resp.response.model == "gemini-2.5-flash-tts"

        kwargs = create_kwargs(client)
        assert kwargs["model"] == "gemini-2.5-flash-tts"
        assert kwargs["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
        assert kwargs["response_format"] == {"type": "audio"}

    def test_wraps_input_in_transcript_tags(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()
        provider.execute(_tts_config(), query, "Say this text")
        assert create_kwargs(client)["input"] == [
            {
                "type": "user_input",
                "content": [
                    {"type": "text", "text": "<transcript>Say this text</transcript>"}
                ],
            }
        ]

    def test_wav_response_is_stripped_to_pcm(self, provider, client, query):
        # 3.8+ TTS models return a full WAV; re-wrapping it would nest headers.
        client.genai.interactions.create.return_value = tts_interaction(
            pcm_to_wav(SAMPLE_PCM_BYTES)
        )
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert err is None
        decoded = base64.b64decode(resp.response.output.content.value)
        assert decoded == pcm_to_wav(SAMPLE_PCM_BYTES)

    def test_rejects_non_string_input(self, provider, client, query):
        resp, err = provider.execute(_tts_config(), query, ["not a string"])
        assert resp is None
        assert "text string as input" in err
        client.genai.interactions.create.assert_not_called()

    def test_rejects_empty_input(self, provider, query):
        resp, err = provider.execute(_tts_config(), query, "   ")
        assert resp is None
        assert "text input is empty" in err

    def test_missing_audio_returns_error(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction(text="oops")
        resp, err = provider.execute(_tts_config(), query, "hello")
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] TTS response is missing audio data")

    def test_empty_audio_data_string_returns_missing_audio_error(
        self, provider, client, query
    ):
        client.genai.interactions.create.return_value = make_interaction(
            text=None, audio_b64=""
        )
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert "missing audio data" in err

    def test_non_completed_status_returns_error(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction(
            text=None, audio=SAMPLE_PCM_BYTES, status="failed"
        )
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert "missing audio data" in err

    def test_wav_with_zero_frames_returns_empty_audio_error(
        self, provider, client, query
    ):
        client.genai.interactions.create.return_value = tts_interaction(pcm_to_wav(b""))
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] TTS returned empty audio data")

    def test_invalid_base64_returns_unexpected_error(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction(
            text=None, audio_b64="not-valid-base64!!"
        )
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Unexpected error")

    def test_language_is_forwarded(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()
        provider.execute(_tts_config(language="en-US"), query, "hi")
        assert create_kwargs(client)["generation_config"] == {
            "speech_config": [{"voice": "Kore", "language": "en-US"}]
        }

    def test_director_notes_become_speech_metadata_annotation(
        self, provider, client, query
    ):
        client.genai.interactions.create.return_value = tts_interaction()
        config = _tts_config(
            provider_specific={"gemini": {"director_notes": "whisper"}}
        )
        provider.execute(config, query, "hi")
        transcript = create_kwargs(client)["input"][0]["content"][0]
        assert transcript["annotations"] == [
            {"type": "speech_metadata", "style": "whisper"}
        ]

    def test_mp3_conversion_success(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()
        with patch(
            f"{GCP_MODULE}.convert_pcm_to_mp3", return_value=(b"mp3bytes", None)
        ):
            resp, err = provider.execute(
                _tts_config(response_format="mp3"), query, "hi"
            )
        assert err is None
        assert resp.response.output.content.mime_type == "audio/mp3"
        assert base64.b64decode(resp.response.output.content.value) == b"mp3bytes"

    def test_mp3_conversion_failure_returns_error(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()
        with patch(
            f"{GCP_MODULE}.convert_pcm_to_mp3", return_value=(None, "ffmpeg missing")
        ):
            resp, err = provider.execute(
                _tts_config(response_format="mp3"), query, "hi"
            )
        assert resp is None
        assert err.startswith("[KAAPI] Post-processing failure")
        assert "ffmpeg missing" in err

    def test_ogg_conversion_success(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()
        with patch(
            f"{GCP_MODULE}.convert_pcm_to_ogg", return_value=(b"oggbytes", None)
        ):
            resp, err = provider.execute(
                _tts_config(response_format="ogg"), query, "hi"
            )
        assert err is None
        assert resp.response.output.content.mime_type == "audio/ogg"
        assert base64.b64decode(resp.response.output.content.value) == b"oggbytes"

    def test_ogg_conversion_failure_returns_error(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()
        with patch(
            f"{GCP_MODULE}.convert_pcm_to_ogg", return_value=(None, "codec missing")
        ):
            resp, err = provider.execute(
                _tts_config(response_format="ogg"), query, "hi"
            )
        assert resp is None
        assert err.startswith("[KAAPI] Post-processing failure")
        assert "codec missing" in err

    def test_unsupported_response_format_falls_back_to_wav(
        self, provider, client, query
    ):
        client.genai.interactions.create.return_value = tts_interaction()
        resp, err = provider.execute(_tts_config(response_format="flac"), query, "hi")
        assert err is None
        assert resp.response.output.content.mime_type == "audio/wav"

    def test_raw_response_included_when_requested(self, provider, client, query):
        client.genai.interactions.create.return_value = tts_interaction()
        resp, _ = provider.execute(
            _tts_config(), query, "hi", include_provider_raw_response=True
        )
        assert resp.provider_raw_response["id"] == "int_123"


class TestText:
    def test_happy_path_drops_temperature(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction(
            text="generated answer"
        )

        resp, err = provider.execute(_text_config(temperature=0.2), query, "hello")

        assert err is None
        assert resp.response.output.content.value == "generated answer"
        assert resp.response.provider_response_id == "int_123"
        assert resp.usage.reasoning_tokens == 2
        assert create_kwargs(client) == {
            "model": "gemini-2.5-flash",
            "input": [
                {"type": "user_input", "content": [{"type": "text", "text": "hello"}]}
            ],
        }

    def test_forwards_system_instruction(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction()
        provider.execute(_text_config(instructions="be terse"), query, "hi")
        assert create_kwargs(client)["system_instruction"] == "be terse"

    def test_forwards_max_output_tokens(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction()
        provider.execute(_text_config(max_output_tokens=256), query, "hi")
        assert create_kwargs(client)["generation_config"] == {"max_output_tokens": 256}

    def test_forwards_output_schema_as_json_response_format(
        self, provider, client, query
    ):
        schema = {"type": "OBJECT", "properties": {"band": {"type": "STRING"}}}
        client.genai.interactions.create.return_value = make_interaction(
            text='{"band": "high"}'
        )
        provider.execute(_text_config(output_schema=schema), query, "hi")
        kwargs = create_kwargs(client)
        assert kwargs["response_format"] == schema
        assert kwargs["response_mime_type"] == "application/json"

    def test_thinking_config_takes_precedence_over_reasoning(
        self, provider, client, query
    ):
        client.genai.interactions.create.return_value = make_interaction()
        config = _text_config(
            thinking_config={"thinking_level": "high"}, reasoning="low"
        )
        provider.execute(config, query, "hello")
        assert create_kwargs(client)["generation_config"] == {"thinking_level": "high"}

    def test_reasoning_used_as_thinking_level_fallback(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction()
        provider.execute(_text_config(reasoning="minimal"), query, "hello")
        assert create_kwargs(client)["generation_config"] == {
            "thinking_level": "minimal"
        }

    def test_accepts_multimodal_list_input(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction(
            text="described"
        )
        parts = [
            TextContent(value="describe this"),
            ImageContent(format="base64", value="aW1n", mime_type="image/png"),
            ImageContent(
                format="url", value="https://x/img.png", mime_type="image/png"
            ),
            PDFContent(format="base64", value="cGRm", mime_type="application/pdf"),
            PDFContent(
                format="url", value="https://x/doc.pdf", mime_type="application/pdf"
            ),
        ]
        resp, err = provider.execute(_text_config(), query, parts)
        assert err is None
        assert create_kwargs(client)["input"][0]["content"] == [
            {"type": "text", "text": "describe this"},
            {"type": "image", "data": "aW1n", "mime_type": "image/png"},
            {"type": "image", "uri": "https://x/img.png", "mime_type": "image/png"},
            {"type": "document", "data": "cGRm", "mime_type": "application/pdf"},
            {
                "type": "document",
                "uri": "https://x/doc.pdf",
                "mime_type": "application/pdf",
            },
        ]

    def test_accepts_multimodal_input_wrapper(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction()
        multimodal = MultiModalInput(parts=[TextContent(value="hi")])
        resp, err = provider.execute(_text_config(), query, multimodal)
        assert err is None
        assert create_kwargs(client)["input"][0]["content"] == [
            {"type": "text", "text": "hi"}
        ]

    def test_missing_content_returns_error(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction(text=None)
        resp, err = provider.execute(_text_config(), query, "hello")
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] Text response is missing generated content")

    def test_empty_interaction_id_gets_generated_response_id(
        self, provider, client, query
    ):
        client.genai.interactions.create.return_value = make_interaction(
            interaction_id=None
        )
        resp, err = provider.execute(_text_config(), query, "hi")
        assert err is None
        assert resp.response.provider_response_id.startswith("google-gcp-")

    def test_raw_response_included_when_requested(self, provider, client, query):
        client.genai.interactions.create.return_value = make_interaction()
        resp, _ = provider.execute(
            _text_config(), query, "hello", include_provider_raw_response=True
        )
        assert resp.provider_raw_response["id"] == "int_123"


class TestExecuteErrors:
    @pytest.mark.parametrize(
        ("status_code", "expected_prefix"),
        [
            (400, "[GOOGLE_GCP] Bad request (code: 400)"),
            (401, "[GOOGLE_GCP] Authentication / permission denied (code: 401)"),
            (403, "[GOOGLE_GCP] Authentication / permission denied (code: 403)"),
            (404, "[GOOGLE_GCP] Resource not found (code: 404)"),
            (429, "[GOOGLE_GCP] Rate limit / quota exceeded (code: 429)"),
            (500, "[GOOGLE_GCP] Server error (code: 500)"),
            (503, "[GOOGLE_GCP] Server error (code: 503)"),
            (418, "[GOOGLE_GCP] HTTP error (code: 418)"),
        ],
    )
    def test_api_status_error_branches(
        self, provider, client, query, status_code, expected_prefix
    ):
        client.genai.interactions.create.side_effect = status_error(
            status_code, "upstream detail"
        )
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith(f"{expected_prefix}: upstream detail")

    def test_text_http_error_returns_message(self, provider, client, query):
        client.genai.interactions.create.side_effect = status_error(
            403, "permission denied"
        )
        resp, err = provider.execute(_text_config(), query, "hi")
        assert resp is None
        assert "403" in err
        assert "permission denied" in err

    def test_timeout_returns_clean_message(self, provider, client, query):
        client.genai.interactions.create.side_effect = compat_errors.APITimeoutError(
            request=VERTEX_REQUEST
        )
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Google GCP request timed out")

    def test_connection_error_returns_clean_message(self, provider, client, query):
        client.genai.interactions.create.side_effect = compat_errors.APIConnectionError(
            request=VERTEX_REQUEST
        )
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Google GCP connection failed")

    def test_google_auth_error_returns_auth_failed(self, provider, client, query):
        # Raised while minting the SA access token, before any HTTP status.
        client.genai.interactions.create.side_effect = google_auth_errors.RefreshError(
            "token revoked"
        )
        resp, err = provider.execute(_text_config(), query, "hi")
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] Authentication failed (RefreshError)")
        assert "token revoked" in err

    def test_wraps_type_error(self, provider, client, query):
        client.genai.interactions.create.side_effect = TypeError("bad kwarg")
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Invalid or unexpected parameter")
        assert "bad kwarg" in err

    def test_wraps_unexpected_exception(self, provider, client, query):
        client.genai.interactions.create.side_effect = RuntimeError("kaboom")
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Unexpected error")
        assert "kaboom" in err

    def test_unsupported_completion_type_returns_error(self, provider, query):
        # SimpleNamespace bypasses NativeCompletionConfig's enum validation to
        # reach the dispatcher's unsupported-type fallthrough.
        config = SimpleNamespace(provider="google-gcp-native", type="embedding")
        resp, err = provider.execute(config, query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Unsupported completion type")
        assert "embedding" in err
