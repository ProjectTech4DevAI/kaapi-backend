"""Tests for the Google GCP (Vertex) provider."""

import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.auth import exceptions as google_auth_errors
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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
GS_URI = "gs://test-bucket/audio/test.wav"
USAGE = genai_types.GenerateContentResponseUsageMetadata(
    prompt_token_count=10,
    candidates_token_count=5,
    total_token_count=17,
    thoughts_token_count=2,
)


def make_response(
    *,
    text: str | None = "hello world",
    audio: bytes | None = None,
    response_id: str | None = "resp_123",
    model_version: str | None = "gemini-2.5-flash-001",
    finish_reason: genai_types.FinishReason | None = genai_types.FinishReason.STOP,
    usage: genai_types.GenerateContentResponseUsageMetadata | None = USAGE,
) -> genai_types.GenerateContentResponse:
    parts: list[genai_types.Part] = []
    if text is not None:
        parts.append(genai_types.Part(text=text))
    if audio is not None:
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(data=audio, mime_type="audio/L16")
            )
        )
    content = genai_types.Content(role="model", parts=parts) if parts else None
    return genai_types.GenerateContentResponse(
        response_id=response_id,
        model_version=model_version,
        candidates=[
            genai_types.Candidate(content=content, finish_reason=finish_reason)
        ],
        usage_metadata=usage,
    )


def blocked_response() -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        response_id="resp_blocked",
        prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(
            block_reason=genai_types.BlockedReason.SAFETY
        ),
    )


def tts_response(
    audio: bytes = SAMPLE_PCM_BYTES,
) -> genai_types.GenerateContentResponse:
    return make_response(text=None, audio=audio)


def call_kwargs(client: GoogleGCPClient) -> dict[str, Any]:
    client.genai.models.generate_content.assert_called_once()
    return client.genai.models.generate_content.call_args.kwargs


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
def generate(client: GoogleGCPClient) -> MagicMock:
    return client.genai.models.generate_content


@pytest.fixture
def query() -> QueryParams:
    return QueryParams(input="ignored")


@pytest.fixture
def upload_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    # GCS is an external boundary; STT stages audio there before calling Vertex.
    mock = MagicMock(return_value=GS_URI)
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
        assert kwargs["http_options"].retry_options.attempts == 3
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


class TestExtractUsage:
    def test_maps_usage_metadata(self):
        usage = GoogleGCPProvider._extract_usage(make_response(), "google-gcp")
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.reasoning_tokens,
        ) == (10, 5, 17, 2)

    def test_none_counts_become_zero(self):
        response = make_response(
            usage=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=4, total_token_count=4
            )
        )
        usage = GoogleGCPProvider._extract_usage(response, "google-gcp")
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.reasoning_tokens,
        ) == (4, 0, 4, 0)

    def test_missing_usage_metadata_is_all_zeros(self):
        usage = GoogleGCPProvider._extract_usage(
            make_response(usage=None), "google-gcp"
        )
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.reasoning_tokens,
        ) == (0, 0, 0, 0)


class TestSTT:
    @pytest.fixture
    def audio_ref(self) -> AudioRef:
        return AudioRef(bytes_=b"RIFFfake", mime_type="audio/wav")

    def test_happy_path(self, provider, generate, query, audio_ref, upload_mock):
        generate.return_value = make_response(text="  hi there \n")

        resp, err = provider.execute(_stt_config(), query, audio_ref)

        assert err is None
        assert resp.response.output.content.value == "hi there"
        assert resp.response.model == "gemini-2.5-flash-001"
        assert resp.response.provider_response_id == "resp_123"
        assert resp.usage.total_tokens == 17

        upload_kwargs = upload_mock.call_args.kwargs
        assert upload_kwargs["audio_bytes"] == b"RIFFfake"
        assert upload_kwargs["bucket_name"] == "test-bucket"
        assert upload_kwargs["content_type"] == "audio/wav"

        kwargs = generate.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        prompt_part, audio_part = kwargs["contents"][0].parts
        assert "Detect the spoken language automatically" in prompt_part.text
        assert audio_part.file_data.file_uri == GS_URI
        assert audio_part.file_data.mime_type == "audio/wav"

    def test_defaults_thinking_low_and_max_tokens_2048(
        self, provider, generate, query, audio_ref, upload_mock
    ):
        generate.return_value = make_response()
        provider.execute(_stt_config(temperature=0.3), query, audio_ref)
        config = call_kwargs(provider.client)["config"]
        assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW
        assert config.max_output_tokens == 2048
        assert config.temperature is None

    def test_explicit_thinking_level_and_max_tokens_override_defaults(
        self, provider, generate, query, audio_ref, upload_mock
    ):
        generate.return_value = make_response()
        provider.execute(
            _stt_config(thinking_level="high", max_output_tokens=512), query, audio_ref
        )
        config = call_kwargs(provider.client)["config"]
        assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH
        assert config.max_output_tokens == 512

    def test_input_language_and_instructions_shape_prompt(
        self, provider, generate, query, audio_ref, upload_mock
    ):
        generate.return_value = make_response()
        config = _stt_config(
            input_language="hi-IN", output_language="en-IN", instructions="be precise"
        )
        resp, err = provider.execute(config, query, audio_ref)

        assert err is None
        assert resp.response.output.content.language_code == "en-IN"
        prompt = call_kwargs(provider.client)["contents"][0].parts[0].text
        assert prompt.startswith("be precise")
        assert "hi-IN" in prompt
        assert "translate to en-IN" in prompt

    def test_rejects_non_audioref_input(self, provider, generate, query):
        resp, err = provider.execute(_stt_config(), query, "/some/path.wav")
        assert resp is None
        assert "AudioRef input" in err
        generate.assert_not_called()

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
        self, provider, generate, query, audio_ref, monkeypatch
    ):
        monkeypatch.setattr(
            f"{GCP_MODULE}.upload_audio_to_gcs",
            MagicMock(side_effect=RuntimeError("bucket denied")),
        )
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert "Failed to stage audio for Google GCP STT" in err
        assert "bucket denied" in err
        generate.assert_not_called()

    @pytest.mark.parametrize(
        ("response", "reason"),
        [
            (
                make_response(text=None, finish_reason=genai_types.FinishReason.SAFETY),
                "finish_reason=FinishReason.SAFETY",
            ),
            (blocked_response(), "block_reason=BlockedReason.SAFETY"),
        ],
    )
    def test_empty_or_blocked_output_returns_error(
        self, provider, generate, query, audio_ref, upload_mock, response, reason
    ):
        generate.return_value = response
        resp, err = provider.execute(_stt_config(), query, audio_ref)
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] STT response is missing transcribed text")
        assert reason in err

    def test_raw_response_included_only_when_requested(
        self, provider, generate, query, audio_ref, upload_mock
    ):
        generate.return_value = make_response()
        resp, _ = provider.execute(
            _stt_config(), query, audio_ref, include_provider_raw_response=True
        )
        assert resp.provider_raw_response["response_id"] == "resp_123"
        resp, _ = provider.execute(_stt_config(), query, audio_ref)
        assert resp.provider_raw_response is None


class TestTTS:
    def test_request_carries_voice_language_and_transcript(
        self, provider, generate, query
    ):
        generate.return_value = tts_response()

        resp, err = provider.execute(_tts_config(language="en-IN"), query, "Say this")

        assert err is None
        kwargs = call_kwargs(provider.client)
        assert kwargs["model"] == "gemini-2.5-flash-tts"
        assert kwargs["contents"] == "<transcript>Say this</transcript>"
        config = kwargs["config"]
        assert config.response_modalities == [genai_types.Modality.AUDIO]
        speech = config.speech_config
        assert speech.voice_config.prebuilt_voice_config.voice_name == "Kore"
        assert speech.language_code == "en-IN"
        assert config.system_instruction is None

    def test_director_notes_become_system_instruction(self, provider, generate, query):
        generate.return_value = tts_response()
        config = _tts_config(
            provider_specific={"gemini": {"director_notes": "whisper"}}
        )
        provider.execute(config, query, "hi")
        assert call_kwargs(provider.client)["config"].system_instruction == "whisper"

    @pytest.mark.parametrize(
        "audio",
        [SAMPLE_PCM_BYTES, pcm_to_wav(SAMPLE_PCM_BYTES)],
        ids=["raw_pcm", "wav_header"],
    )
    def test_wav_output_is_single_wav_wrapping_of_pcm(
        self, provider, generate, query, audio
    ):
        # 3.8+ TTS models return a full WAV; re-wrapping it would nest headers.
        generate.return_value = tts_response(audio)
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert err is None
        content = resp.response.output.content
        assert content.format == "base64"
        assert content.mime_type == "audio/wav"
        assert base64.b64decode(content.value) == pcm_to_wav(SAMPLE_PCM_BYTES)
        assert resp.response.model == "gemini-2.5-flash-001"

    @pytest.mark.parametrize("fmt", ["mp3", "ogg"])
    @pytest.mark.parametrize(
        "audio",
        [SAMPLE_PCM_BYTES, pcm_to_wav(SAMPLE_PCM_BYTES)],
        ids=["raw_pcm", "wav_header"],
    )
    def test_converted_formats_receive_raw_pcm(
        self, provider, generate, query, audio, fmt
    ):
        generate.return_value = tts_response(audio)
        converter = MagicMock(return_value=(f"{fmt}bytes".encode(), None))
        with patch(f"{GCP_MODULE}.convert_pcm_to_{fmt}", converter):
            resp, err = provider.execute(_tts_config(response_format=fmt), query, "hi")
        assert err is None
        assert converter.call_args.args[0] == SAMPLE_PCM_BYTES
        assert resp.response.output.content.mime_type == f"audio/{fmt}"
        assert (
            base64.b64decode(resp.response.output.content.value)
            == f"{fmt}bytes".encode()
        )

    @pytest.mark.parametrize("fmt", ["mp3", "ogg"])
    def test_conversion_failure_returns_error(self, provider, generate, query, fmt):
        generate.return_value = tts_response()
        with patch(
            f"{GCP_MODULE}.convert_pcm_to_{fmt}", return_value=(None, "codec missing")
        ):
            resp, err = provider.execute(_tts_config(response_format=fmt), query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Post-processing failure")
        assert "codec missing" in err

    def test_unsupported_response_format_falls_back_to_wav(
        self, provider, generate, query
    ):
        generate.return_value = tts_response()
        resp, err = provider.execute(_tts_config(response_format="flac"), query, "hi")
        assert err is None
        assert resp.response.output.content.mime_type == "audio/wav"

    @pytest.mark.parametrize(
        "response",
        [
            make_response(text="oops"),
            tts_response(b""),
            make_response(text=None, finish_reason=genai_types.FinishReason.SAFETY),
            blocked_response(),
        ],
        ids=["text_only", "empty_blob", "safety_finish", "prompt_blocked"],
    )
    def test_missing_audio_returns_error(self, provider, generate, query, response):
        generate.return_value = response
        resp, err = provider.execute(_tts_config(), query, "hello")
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] TTS response is missing audio data")

    def test_wav_with_zero_frames_returns_empty_audio_error(
        self, provider, generate, query
    ):
        generate.return_value = tts_response(pcm_to_wav(b""))
        resp, err = provider.execute(_tts_config(), query, "hi")
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] TTS returned empty audio data")

    def test_rejects_non_string_input(self, provider, generate, query):
        resp, err = provider.execute(_tts_config(), query, ["not a string"])
        assert resp is None
        assert "text string as input" in err
        generate.assert_not_called()

    def test_rejects_empty_input(self, provider, generate, query):
        resp, err = provider.execute(_tts_config(), query, "   ")
        assert resp is None
        assert "text input is empty" in err
        generate.assert_not_called()

    def test_raw_response_included_when_requested(self, provider, generate, query):
        generate.return_value = tts_response()
        resp, err = provider.execute(
            _tts_config(), query, "hi", include_provider_raw_response=True
        )
        assert err is None
        assert resp.provider_raw_response["response_id"] == "resp_123"


class TestText:
    def test_happy_path(self, provider, generate, query):
        generate.return_value = make_response(text="generated answer")

        resp, err = provider.execute(_text_config(temperature=0.2), query, "hello")

        assert err is None
        assert resp.response.output.content.value == "generated answer"
        assert resp.response.provider_response_id == "resp_123"
        assert resp.response.model == "gemini-2.5-flash-001"
        assert (
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            resp.usage.total_tokens,
            resp.usage.reasoning_tokens,
        ) == (10, 5, 17, 2)

        kwargs = call_kwargs(provider.client)
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["contents"].role == "user"
        assert [p.text for p in kwargs["contents"].parts] == ["hello"]
        config = kwargs["config"]
        assert config.temperature is None
        assert config.system_instruction is None
        assert config.thinking_config is None
        assert config.max_output_tokens is None
        assert config.response_mime_type is None

    def test_forwards_instructions_and_max_output_tokens(
        self, provider, generate, query
    ):
        generate.return_value = make_response()
        provider.execute(
            _text_config(instructions="be terse", max_output_tokens=256), query, "hi"
        )
        config = call_kwargs(provider.client)["config"]
        assert config.system_instruction == "be terse"
        assert config.max_output_tokens == 256

    def test_output_schema_requests_json(self, provider, generate, query):
        schema = {"type": "object", "properties": {"band": {"type": "string"}}}
        generate.return_value = make_response(text='{"band": "high"}')
        resp, err = provider.execute(_text_config(output_schema=schema), query, "hi")
        assert err is None
        assert resp.response.output.content.value == '{"band": "high"}'
        config = call_kwargs(provider.client)["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == schema
        assert config.response_schema is None

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (
                {"thinking_config": {"thinking_level": "high"}, "reasoning": "low"},
                genai_types.ThinkingLevel.HIGH,
            ),
            ({"reasoning": "minimal"}, genai_types.ThinkingLevel.MINIMAL),
            (
                {"thinking_config": {}, "reasoning": "medium"},
                genai_types.ThinkingLevel.MEDIUM,
            ),
        ],
    )
    def test_thinking_level_resolution(
        self, provider, generate, query, params, expected
    ):
        generate.return_value = make_response()
        provider.execute(_text_config(**params), query, "hello")
        thinking = call_kwargs(provider.client)["config"].thinking_config
        assert thinking.thinking_level == expected

    def test_multimodal_list_input_maps_to_genai_parts(self, provider, generate, query):
        generate.return_value = make_response(text="described")
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
        text, img_inline, img_uri, pdf_inline, pdf_uri = call_kwargs(provider.client)[
            "contents"
        ].parts
        assert text.text == "describe this"
        assert (img_inline.inline_data.data, img_inline.inline_data.mime_type) == (
            b"img",
            "image/png",
        )
        assert (img_uri.file_data.file_uri, img_uri.file_data.mime_type) == (
            "https://x/img.png",
            "image/png",
        )
        assert (pdf_inline.inline_data.data, pdf_inline.inline_data.mime_type) == (
            b"pdf",
            "application/pdf",
        )
        assert (pdf_uri.file_data.file_uri, pdf_uri.file_data.mime_type) == (
            "https://x/doc.pdf",
            "application/pdf",
        )

    def test_multimodal_input_wrapper(self, provider, generate, query):
        generate.return_value = make_response()
        resp, err = provider.execute(
            _text_config(), query, MultiModalInput(parts=[TextContent(value="hi")])
        )
        assert err is None
        assert [p.text for p in call_kwargs(provider.client)["contents"].parts] == [
            "hi"
        ]

    def test_missing_ids_fall_back(self, provider, generate, query):
        generate.return_value = make_response(response_id=None, model_version=None)
        resp, err = provider.execute(_text_config(), query, "hi")
        assert err is None
        assert resp.response.provider_response_id.startswith("google-gcp-")
        assert len(resp.response.provider_response_id) == len("google-gcp-") + 32
        assert resp.response.model == "gemini-2.5-flash"

    @pytest.mark.parametrize(
        ("response", "reason"),
        [
            (
                make_response(text=None, finish_reason=genai_types.FinishReason.SAFETY),
                "finish_reason=FinishReason.SAFETY",
            ),
            (make_response(text=""), "finish_reason=FinishReason.STOP"),
            (blocked_response(), "block_reason=BlockedReason.SAFETY"),
        ],
        ids=["safety_finish", "empty_text", "prompt_blocked"],
    )
    def test_empty_or_blocked_output_returns_error(
        self, provider, generate, query, response, reason
    ):
        generate.return_value = response
        resp, err = provider.execute(_text_config(), query, "hello")
        assert resp is None
        assert err.startswith("[GOOGLE_GCP] Text response is missing generated content")
        assert reason in err

    def test_raw_response_included_when_requested(self, provider, generate, query):
        generate.return_value = make_response()
        resp, _ = provider.execute(
            _text_config(), query, "hello", include_provider_raw_response=True
        )
        assert resp.provider_raw_response["response_id"] == "resp_123"


class TestExecuteErrors:
    @pytest.mark.parametrize(
        ("error_cls", "code", "expected_prefix"),
        [
            (genai_errors.ClientError, 400, "[GOOGLE_GCP] Bad request (code: 400)"),
            (
                genai_errors.ClientError,
                401,
                "[GOOGLE_GCP] Authentication / permission denied (code: 401)",
            ),
            (
                genai_errors.ClientError,
                403,
                "[GOOGLE_GCP] Authentication / permission denied (code: 403)",
            ),
            (
                genai_errors.ClientError,
                404,
                "[GOOGLE_GCP] Resource not found (code: 404)",
            ),
            (
                genai_errors.ClientError,
                429,
                "[GOOGLE_GCP] Rate limit / quota exceeded (code: 429)",
            ),
            (genai_errors.ServerError, 500, "[GOOGLE_GCP] Server error (code: 500)"),
            (genai_errors.ServerError, 503, "[GOOGLE_GCP] Server error (code: 503)"),
            (genai_errors.ClientError, 418, "[GOOGLE_GCP] HTTP error (code: 418)"),
        ],
    )
    @pytest.mark.parametrize("config_factory", [_text_config, _tts_config])
    def test_api_errors_map_to_messages(
        self,
        provider,
        generate,
        query,
        error_cls,
        code,
        expected_prefix,
        config_factory,
    ):
        generate.side_effect = error_cls(
            code, {"error": {"code": code, "message": "upstream detail"}}
        )
        resp, err = provider.execute(config_factory(), query, "hi")
        assert resp is None
        assert err.startswith(f"{expected_prefix}: upstream detail")

    @pytest.mark.parametrize(
        ("exc", "expected_prefix"),
        [
            (
                httpx.ReadTimeout("read timed out", request=VERTEX_REQUEST),
                "[KAAPI] Google GCP request timed out after 60s (ReadTimeout)",
            ),
            (
                httpx.ConnectTimeout("connect timed out", request=VERTEX_REQUEST),
                "[KAAPI] Google GCP request timed out after 60s (ConnectTimeout)",
            ),
            (
                httpx.ConnectError("dns failure", request=VERTEX_REQUEST),
                "[KAAPI] Google GCP connection failed (ConnectError)",
            ),
            (
                httpx.RemoteProtocolError("peer closed", request=VERTEX_REQUEST),
                "[KAAPI] Google GCP connection failed (RemoteProtocolError)",
            ),
            (
                google_auth_errors.RefreshError("token revoked"),
                "[GOOGLE_GCP] Authentication failed (RefreshError)",
            ),
            (
                TypeError("bad kwarg"),
                "[KAAPI] Invalid or unexpected parameter in Config: bad kwarg",
            ),
            (
                RuntimeError("kaboom"),
                "[KAAPI] Unexpected error while executing Google GCP",
            ),
        ],
        ids=[
            "read_timeout",
            "connect_timeout",
            "connect_error",
            "remote_protocol",
            "auth",
            "type_error",
            "generic",
        ],
    )
    def test_non_api_errors_map_to_messages(
        self, provider, generate, query, exc, expected_prefix
    ):
        generate.side_effect = exc
        resp, err = provider.execute(_text_config(), query, "hi")
        assert resp is None
        assert err.startswith(expected_prefix)
        assert str(exc) in err

    def test_stt_api_error_is_mapped(self, provider, generate, query, upload_mock):
        generate.side_effect = genai_errors.ClientError(
            403, {"error": {"code": 403, "message": "permission denied"}}
        )
        resp, err = provider.execute(
            _stt_config(), query, AudioRef(bytes_=b"x", mime_type="audio/wav")
        )
        assert resp is None
        assert err.startswith(
            "[GOOGLE_GCP] Authentication / permission denied (code: 403)"
        )

    def test_unsupported_completion_type_returns_error(self, provider, query):
        # SimpleNamespace bypasses NativeCompletionConfig's enum validation to
        # reach the dispatcher's unsupported-type fallthrough.
        config = SimpleNamespace(provider="google-gcp-native", type="embedding")
        resp, err = provider.execute(config, query, "hi")
        assert resp is None
        assert err.startswith("[KAAPI] Unsupported completion type")
        assert "embedding" in err
