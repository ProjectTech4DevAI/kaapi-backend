import base64
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai._gaos.lib import compat_errors
from google.genai.interactions import Interaction

from app.core.audio_utils import AudioRef, pcm_to_wav
from app.models.llm import NativeCompletionConfig, QueryParams, TextContent
from app.models.llm.constants import CompletionType
from app.services.llm.providers.google_aistudio import GoogleAIProvider

SAMPLE_PCM_BYTES = b"\x00\x01" * 1000
GEMINI_REQUEST = httpx.Request("POST", "https://generativelanguage.googleapis.com")


def make_interaction(
    *,
    text: str | None = "Hello world",
    audio: bytes | None = None,
    interaction_id: str | None = "int_123",
    model: str | None = "gemini-2.5-pro",
    status: str = "completed",
    errors: list[dict[str, str]] | None = None,
    usage: dict[str, int] | None = None,
) -> Interaction:
    """Build a real SDK Interaction from a wire-shaped payload.

    The SDK derives output_text / output_audio from the model_output step, so
    going through model_validate keeps the tests honest about those accessors.
    """
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    if audio is not None:
        content.append(
            {
                "type": "audio",
                "data": base64.b64encode(audio).decode("ascii"),
                "mime_type": "audio/l16",
            }
        )
    payload: dict[str, Any] = {
        "status": status,
        "model": model,
        "steps": [{"type": "model_output", "content": content}],
        "usage": usage
        if usage is not None
        else {
            "total_input_tokens": 50,
            "total_output_tokens": 100,
            "total_thought_tokens": 7,
            "total_tokens": 157,
        },
    }
    if interaction_id is not None:
        payload["id"] = interaction_id
    if errors is not None:
        payload["errors"] = errors
    return Interaction.model_validate(payload)


def create_kwargs(mock_client: MagicMock) -> dict[str, Any]:
    mock_client.interactions.create.assert_called_once()
    return mock_client.interactions.create.call_args.kwargs


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def provider(mock_client: MagicMock) -> GoogleAIProvider:
    return GoogleAIProvider(client=mock_client)


@pytest.fixture
def query_params() -> QueryParams:
    return QueryParams(input="Hello there")


class TestGoogleAIProviderText:
    @pytest.fixture
    def text_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.TEXT,
            params={"model": "gemini-2.5-pro"},
        )

    def test_text_happy_path(self, provider, mock_client, text_config, query_params):
        mock_client.interactions.create.return_value = make_interaction(text="Hi!")

        result, error = provider.execute(text_config, query_params, "Hello there")

        assert error is None
        assert result.response.output.content.value == "Hi!"
        assert result.response.provider_response_id == "int_123"
        assert result.response.model == "gemini-2.5-pro"
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 100
        assert result.usage.reasoning_tokens == 7
        assert result.usage.total_tokens == 157
        kwargs = create_kwargs(mock_client)
        assert kwargs["model"] == "gemini-2.5-pro"
        assert kwargs["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "Hello there"}]}
        ]

    def test_text_instructions_become_system_instruction(
        self, provider, mock_client, text_config, query_params
    ):
        text_config.params["instructions"] = "Be terse."
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(text_config, query_params, "Ping")

        assert create_kwargs(mock_client)["system_instruction"] == "Be terse."

    def test_text_minimal_request_has_no_optional_fields(
        self, provider, mock_client, text_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(text_config, query_params, "Ping")

        assert set(create_kwargs(mock_client)) == {"model", "input"}

    def test_text_thinking_config_takes_precedence_over_reasoning(
        self, provider, mock_client, text_config, query_params
    ):
        text_config.params["thinking_config"] = {"thinking_level": "high"}
        text_config.params["reasoning"] = "low"
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(text_config, query_params, "Ping")

        assert create_kwargs(mock_client)["generation_config"] == {
            "thinking_level": "high"
        }

    def test_text_reasoning_used_as_thinking_level_fallback(
        self, provider, mock_client, text_config, query_params
    ):
        text_config.params["reasoning"] = "low"
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(text_config, query_params, "Ping")

        assert create_kwargs(mock_client)["generation_config"] == {
            "thinking_level": "low"
        }

    def test_text_max_output_tokens_in_generation_config(
        self, provider, mock_client, text_config, query_params
    ):
        text_config.params["max_output_tokens"] = 256
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(text_config, query_params, "Ping")

        assert create_kwargs(mock_client)["generation_config"] == {
            "max_output_tokens": 256
        }

    def test_text_sampling_params_not_sent(
        self, provider, mock_client, text_config, query_params
    ):
        text_config.params["temperature"] = 0.2
        text_config.params["top_p"] = 0.9
        mock_client.interactions.create.return_value = make_interaction()

        result, error = provider.execute(text_config, query_params, "Ping")

        assert error is None
        assert set(create_kwargs(mock_client)) == {"model", "input"}

    def test_text_list_of_content_parts(
        self, provider, mock_client, text_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction()

        result, error = provider.execute(
            text_config, query_params, [TextContent(value="part one")]
        )

        assert error is None
        assert create_kwargs(mock_client)["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "part one"}]}
        ]

    def test_text_knowledge_base_ids_set_file_search_tool(
        self, provider, mock_client, text_config, query_params
    ):
        store_names = ["fileSearchStores/store-1", "fileSearchStores/store-2"]
        text_config.params["knowledge_base_ids"] = store_names
        mock_client.interactions.create.return_value = make_interaction()

        result, error = provider.execute(text_config, query_params, "Ask the KB")

        assert error is None
        assert create_kwargs(mock_client)["tools"] == [
            {"type": "file_search", "file_search_store_names": store_names}
        ]

    def test_text_missing_model_uses_default(self, provider, mock_client, query_params):
        config = NativeCompletionConfig(
            provider="google-native", type=CompletionType.TEXT, params={}
        )
        mock_client.interactions.create.return_value = make_interaction(model=None)

        result, error = provider.execute(config, query_params, "Hi")

        assert error is None
        assert create_kwargs(mock_client)["model"] == "gemini-2.5-pro"
        assert result.response.model == "gemini-2.5-pro"

    def test_text_include_provider_raw_response(
        self, provider, mock_client, text_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction(text="raw")

        result, error = provider.execute(
            text_config, query_params, "Hi", include_provider_raw_response=True
        )

        assert error is None
        assert result.provider_raw_response["id"] == "int_123"
        assert result.provider_raw_response["output_text"] == "raw"

    def test_text_missing_usage_defaults_to_zeros(
        self, provider, mock_client, text_config, query_params
    ):
        interaction = make_interaction()
        interaction.usage = None
        mock_client.interactions.create.return_value = interaction

        result, error = provider.execute(text_config, query_params, "Hi")

        assert error is None
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 0
        assert result.usage.reasoning_tokens == 0

    def test_text_missing_interaction_id_returns_error(
        self, provider, mock_client, text_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction(
            interaction_id=None
        )

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith("[GEMINI] Text interaction did not complete")

    @pytest.mark.parametrize("status", ["failed", "incomplete"])
    def test_text_non_completed_status_returns_error(
        self, provider, mock_client, text_config, query_params, status
    ):
        mock_client.interactions.create.return_value = make_interaction(
            status=status,
            errors=[{"code": "SAFETY", "message": "blocked by policy"}],
        )

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith("[GEMINI]")
        assert f"status={status}" in error
        assert "SAFETY" in error
        assert "blocked by policy" in error

    def test_text_empty_output_text_returns_error(
        self, provider, mock_client, text_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction(text=None)

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith("[GEMINI]")
        assert "missing generated content" in error


class TestGoogleAIProviderSTT:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        client = MagicMock()
        uploaded = MagicMock()
        uploaded.name = "files/abc"
        uploaded.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc"
        uploaded.mime_type = "audio/wav"
        client.files.upload.return_value = uploaded
        return client

    @pytest.fixture
    def stt_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-aistudio-native",
            type=CompletionType.STT,
            params={"model": "gemini-2.5-pro"},
        )

    @pytest.fixture
    def audio_ref(self) -> AudioRef:
        return AudioRef(bytes_=b"fake audio data", mime_type="audio/wav")

    @staticmethod
    def _prompt(mock_client: MagicMock) -> str:
        content = create_kwargs(mock_client)["input"][0]["content"]
        return content[0]["text"]

    def test_stt_success_with_auto_language(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.interactions.create.return_value = make_interaction(
            text="Hello world"
        )

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert error is None
        assert result.response.output.content.value == "Hello world"
        assert result.response.provider_response_id == "int_123"
        assert result.response.model == "gemini-2.5-pro"
        assert result.response.provider == "google-aistudio-native"
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 100
        assert result.usage.total_tokens == 157
        uploaded_path = mock_client.files.upload.call_args.kwargs["file"]
        assert uploaded_path.endswith(".wav")
        assert "Detect the spoken language automatically" in self._prompt(mock_client)

    def test_stt_references_uploaded_file_uri(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(stt_config, query_params, audio_ref)

        kwargs = create_kwargs(mock_client)
        assert kwargs["model"] == "gemini-2.5-pro"
        assert kwargs["input"][0]["type"] == "user_input"
        assert kwargs["input"][0]["content"][1] == {
            "type": "audio",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
            "mime_type": "audio/wav",
        }

    def test_stt_defaults_thinking_level_to_low(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(stt_config, query_params, audio_ref)

        assert create_kwargs(mock_client)["generation_config"] == {
            "thinking_level": "low"
        }

    def test_stt_explicit_thinking_level_overrides_default(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        stt_config.params["thinking_level"] = "high"
        mock_client.interactions.create.return_value = make_interaction()

        provider.execute(stt_config, query_params, audio_ref)

        assert create_kwargs(mock_client)["generation_config"] == {
            "thinking_level": "high"
        }

    def test_stt_sampling_params_not_sent(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        stt_config.params["temperature"] = 0.4
        stt_config.params["top_p"] = 0.9
        mock_client.interactions.create.return_value = make_interaction()

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert error is None
        kwargs = create_kwargs(mock_client)
        assert set(kwargs) == {"model", "input", "generation_config"}
        assert kwargs["generation_config"] == {"thinking_level": "low"}

    def test_stt_with_specific_input_language(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        stt_config.params["input_language"] = "English"
        mock_client.interactions.create.return_value = make_interaction()

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert error is None
        assert "Transcribe the audio from English" in self._prompt(mock_client)

    def test_stt_with_translation(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        stt_config.params["input_language"] = "Spanish"
        stt_config.params["output_language"] = "English"
        mock_client.interactions.create.return_value = make_interaction()

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert error is None
        prompt = self._prompt(mock_client)
        assert "Transcribe the audio from Spanish" in prompt
        assert "translate to English" in prompt
        assert result.response.output.content.language_code == "English"

    def test_stt_with_custom_instructions(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        stt_config.params["instructions"] = "Include timestamps"
        mock_client.interactions.create.return_value = make_interaction()

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert error is None
        assert self._prompt(mock_client).startswith("Include timestamps. ")

    def test_stt_empty_output_text_returns_error(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.interactions.create.return_value = make_interaction(text=None)

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert result is None
        assert error.startswith("[GEMINI]")
        assert "missing transcribed text" in error

    def test_stt_failed_status_returns_error(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.interactions.create.return_value = make_interaction(status="failed")

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert result is None
        assert "STT interaction did not complete (status=failed" in error

    def test_stt_missing_interaction_id_returns_error(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.interactions.create.return_value = make_interaction(
            interaction_id=None
        )

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert result is None
        assert error.startswith("[GEMINI] STT interaction did not complete")

    def test_stt_type_error(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.interactions.create.side_effect = TypeError(
            "unexpected keyword argument 'invalid_param'"
        )

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert result is None
        assert "Invalid or unexpected parameter in Config" in error

    def test_stt_generic_exception(
        self, provider, mock_client, stt_config, query_params, audio_ref
    ):
        mock_client.files.upload.side_effect = Exception("File upload failed")

        result, error = provider.execute(stt_config, query_params, audio_ref)

        assert result is None
        assert "[KAAPI] Unexpected error" in error

    def test_stt_invalid_input_type(
        self, provider, mock_client, stt_config, query_params
    ):
        result, error = provider.execute(stt_config, query_params, {"invalid": "data"})

        assert result is None
        assert "STT requires an AudioRef" in error
        mock_client.interactions.create.assert_not_called()


class TestGoogleAIProviderTTS:
    @pytest.fixture
    def tts_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-aistudio-native",
            type=CompletionType.TTS,
            params={
                "model": "gemini-2.5-pro-preview-tts",
                "voice": "Kore",
                "language": "en-IN",
            },
        )

    @pytest.fixture
    def pcm_interaction(self) -> Interaction:
        return make_interaction(
            text=None, audio=SAMPLE_PCM_BYTES, model="gemini-2.5-pro-preview-tts"
        )

    def test_tts_request_shape(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        mock_client.interactions.create.return_value = pcm_interaction

        result, error = provider.execute(tts_config, query_params, "Say this text")

        assert error is None
        kwargs = create_kwargs(mock_client)
        assert kwargs["model"] == "gemini-2.5-pro-preview-tts"
        assert kwargs["input"] == [
            {
                "type": "user_input",
                "content": [
                    {"type": "text", "text": "<transcript>Say this text</transcript>"}
                ],
            }
        ]
        assert kwargs["response_format"] == {"type": "audio"}

    def test_tts_passes_full_bcp47_language(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        mock_client.interactions.create.return_value = pcm_interaction

        provider.execute(tts_config, query_params, "Hello")

        assert create_kwargs(mock_client)["generation_config"] == {
            "speech_config": [{"voice": "Kore", "language": "en-IN"}]
        }

    def test_tts_omits_language_when_unset(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        del tts_config.params["language"]
        mock_client.interactions.create.return_value = pcm_interaction

        provider.execute(tts_config, query_params, "Hello")

        assert create_kwargs(mock_client)["generation_config"] == {
            "speech_config": [{"voice": "Kore"}]
        }

    def test_tts_director_notes_become_style_annotation(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        tts_config.params["provider_specific"] = {
            "gemini": {"director_notes": "Speak in a cheerful tone"}
        }
        mock_client.interactions.create.return_value = pcm_interaction

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        transcript = create_kwargs(mock_client)["input"][0]["content"][0]
        assert transcript["annotations"] == [
            {"type": "speech_metadata", "style": "Speak in a cheerful tone"}
        ]
        assert "system_instruction" not in create_kwargs(mock_client)

    def test_tts_without_director_notes_has_no_annotations(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        mock_client.interactions.create.return_value = pcm_interaction

        provider.execute(tts_config, query_params, "Hello")

        transcript = create_kwargs(mock_client)["input"][0]["content"][0]
        assert "annotations" not in transcript

    def test_tts_default_wav_wraps_pcm_once(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        mock_client.interactions.create.return_value = pcm_interaction

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result.response.output.content.mime_type == "audio/wav"
        decoded = base64.b64decode(result.response.output.content.value)
        assert decoded == pcm_to_wav(SAMPLE_PCM_BYTES, sample_rate=24000)

    def test_tts_wav_returned_by_server_is_not_double_wrapped(
        self, provider, mock_client, tts_config, query_params
    ):
        # 3.8 TTS models reply with a full WAV rather than raw PCM.
        server_wav = pcm_to_wav(SAMPLE_PCM_BYTES, sample_rate=24000)
        mock_client.interactions.create.return_value = make_interaction(
            text=None, audio=server_wav
        )

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        decoded = base64.b64decode(result.response.output.content.value)
        assert decoded == pcm_to_wav(SAMPLE_PCM_BYTES, sample_rate=24000)
        assert decoded.count(b"RIFF") == 1

    def test_tts_wav_returned_by_server_is_unwrapped_before_mp3(
        self, provider, mock_client, tts_config, query_params
    ):
        tts_config.params["response_format"] = "mp3"
        server_wav = pcm_to_wav(SAMPLE_PCM_BYTES, sample_rate=24000)
        mock_client.interactions.create.return_value = make_interaction(
            text=None, audio=server_wav
        )

        with patch(
            "app.services.llm.providers.google_aistudio.convert_pcm_to_mp3",
            return_value=(b"fake-mp3", None),
        ) as mock_convert:
            result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        # ffmpeg conversion is a local external process; mocked to check its input.
        assert mock_convert.call_args.args[0] == SAMPLE_PCM_BYTES

    def test_tts_mp3_format(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        tts_config.params["response_format"] = "mp3"
        mock_client.interactions.create.return_value = pcm_interaction

        with patch(
            "app.services.llm.providers.google_aistudio.convert_pcm_to_mp3",
            return_value=(b"fake-mp3-content", None),
        ) as mock_convert:
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert error is None
        assert result.response.output.content.mime_type == "audio/mp3"
        assert base64.b64decode(result.response.output.content.value) == (
            b"fake-mp3-content"
        )
        mock_convert.assert_called_once_with(SAMPLE_PCM_BYTES)

    def test_tts_ogg_format(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        tts_config.params["response_format"] = "ogg"
        mock_client.interactions.create.return_value = pcm_interaction

        with patch(
            "app.services.llm.providers.google_aistudio.convert_pcm_to_ogg",
            return_value=(b"fake-ogg-content", None),
        ) as mock_convert:
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert error is None
        assert result.response.output.content.mime_type == "audio/ogg"
        assert base64.b64decode(result.response.output.content.value) == (
            b"fake-ogg-content"
        )
        mock_convert.assert_called_once_with(SAMPLE_PCM_BYTES)

    def test_tts_unsupported_format_falls_back_to_wav(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        tts_config.params["response_format"] = "flac"
        mock_client.interactions.create.return_value = pcm_interaction

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result.response.output.content.mime_type == "audio/wav"
        decoded = base64.b64decode(result.response.output.content.value)
        assert decoded.startswith(b"RIFF")

    def test_tts_mp3_conversion_failure(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        tts_config.params["response_format"] = "mp3"
        mock_client.interactions.create.return_value = pcm_interaction

        with patch(
            "app.services.llm.providers.google_aistudio.convert_pcm_to_mp3",
            return_value=(None, "ffmpeg not found"),
        ):
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert result is None
        assert "unable to convert Gemini PCM audio to MP3" in error
        assert "ffmpeg not found" in error

    def test_tts_ogg_conversion_failure(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        tts_config.params["response_format"] = "ogg"
        mock_client.interactions.create.return_value = pcm_interaction

        with patch(
            "app.services.llm.providers.google_aistudio.convert_pcm_to_ogg",
            return_value=(None, "codec error"),
        ):
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert result is None
        assert "unable to convert Gemini PCM audio to OGG" in error

    def test_tts_missing_output_audio_returns_error(
        self, provider, mock_client, tts_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction(text="oops")

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert error.startswith("[GEMINI] Failed to extract audio bytes")

    def test_tts_incomplete_status_returns_error(
        self, provider, mock_client, tts_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction(
            text=None, audio=SAMPLE_PCM_BYTES, status="incomplete"
        )

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert "TTS interaction did not complete (status=incomplete" in error

    def test_tts_missing_interaction_id_returns_error(
        self, provider, mock_client, tts_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction(
            text=None, audio=SAMPLE_PCM_BYTES, interaction_id=None
        )

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert error.startswith("[GEMINI] TTS interaction did not complete")

    def test_tts_empty_input(self, provider, mock_client, tts_config, query_params):
        result, error = provider.execute(tts_config, query_params, "   ")

        assert result is None
        assert "text input is empty" in error
        mock_client.interactions.create.assert_not_called()

    def test_tts_non_string_input(
        self, provider, mock_client, tts_config, query_params
    ):
        result, error = provider.execute(tts_config, query_params, {"invalid": "data"})

        assert result is None
        assert "TTS requires a text string" in error

    def test_tts_missing_usage_defaults_to_zeros(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        pcm_interaction.usage = None
        mock_client.interactions.create.return_value = pcm_interaction

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 0
        assert result.usage.reasoning_tokens == 0

    def test_tts_include_provider_raw_response(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        mock_client.interactions.create.return_value = pcm_interaction

        result, error = provider.execute(
            tts_config, query_params, "Hello", include_provider_raw_response=True
        )

        assert error is None
        assert result.provider_raw_response["id"] == "int_123"
        assert result.provider_raw_response["output_audio"]["mime_type"] == "audio/l16"

    def test_tts_without_provider_raw_response(
        self, provider, mock_client, tts_config, query_params, pcm_interaction
    ):
        mock_client.interactions.create.return_value = pcm_interaction

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result.provider_raw_response is None

    def test_tts_model_falls_back_to_config_model(
        self, provider, mock_client, tts_config, query_params
    ):
        mock_client.interactions.create.return_value = make_interaction(
            text=None, audio=SAMPLE_PCM_BYTES, model=None
        )

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result.response.model == "gemini-2.5-pro-preview-tts"

    def test_tts_default_model(self, provider, mock_client, query_params):
        config = NativeCompletionConfig(
            provider="google-aistudio-native", type=CompletionType.TTS, params={}
        )
        mock_client.interactions.create.return_value = make_interaction(
            text=None, audio=SAMPLE_PCM_BYTES, model=None
        )

        result, error = provider.execute(config, query_params, "Hello")

        assert error is None
        assert create_kwargs(mock_client)["model"] == "gemini-3.8-flash-tts"

    def test_tts_generic_exception(
        self, provider, mock_client, tts_config, query_params
    ):
        mock_client.interactions.create.side_effect = Exception("API unavailable")

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert "[KAAPI] Unexpected error" in error


def status_error(status_code: int, status: str, message: str) -> Exception:
    body = {"error": {"code": status_code, "status": status, "message": message}}
    response = httpx.Response(status_code, json=body, request=GEMINI_REQUEST)
    return compat_errors.APIError.generate(status_code, body, message, response)


class TestGoogleAIProviderErrors:
    @pytest.fixture
    def text_config(self) -> NativeCompletionConfig:
        return NativeCompletionConfig(
            provider="google-native",
            type=CompletionType.TEXT,
            params={"model": "gemini-2.5-pro"},
        )

    @pytest.mark.parametrize(
        ("status_code", "status", "expected_prefix"),
        [
            (400, "INVALID_ARGUMENT", "[GEMINI] Bad request (code: 400)"),
            (
                401,
                "UNAUTHENTICATED",
                "[GEMINI] Authentication / permission denied (code: 401)",
            ),
            (
                403,
                "PERMISSION_DENIED",
                "[GEMINI] Authentication / permission denied (code: 403)",
            ),
            (404, "NOT_FOUND", "[GEMINI] Resource not found (code: 404)"),
            (
                429,
                "RESOURCE_EXHAUSTED",
                "[GEMINI] Rate limit / quota exceeded (code: 429)",
            ),
            (500, "INTERNAL", "[GEMINI] Server error (code: 500)"),
            (503, "UNAVAILABLE", "[GEMINI] Server error (code: 503)"),
            (409, "ABORTED", "[GEMINI] Client error (code: 409)"),
        ],
    )
    def test_api_status_error_maps_to_gemini_message(
        self,
        provider,
        mock_client,
        text_config,
        query_params,
        status_code,
        status,
        expected_prefix,
    ):
        mock_client.interactions.create.side_effect = status_error(
            status_code, status, "upstream detail"
        )

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith(f"{expected_prefix}: upstream detail")

    def test_api_status_error_without_envelope_uses_exception_message(
        self, provider, mock_client, text_config, query_params
    ):
        response = httpx.Response(502, text="Bad Gateway", request=GEMINI_REQUEST)
        mock_client.interactions.create.side_effect = compat_errors.APIStatusError(
            "gateway exploded", response=response, body="Bad Gateway"
        )

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith("[GEMINI] Server error (code: 502): gateway exploded")

    def test_timeout_maps_to_could_not_reach_gemini(
        self, provider, mock_client, text_config, query_params
    ):
        mock_client.interactions.create.side_effect = compat_errors.APITimeoutError(
            request=GEMINI_REQUEST
        )

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith("[KAAPI] Could not reach Gemini (APITimeoutError)")

    def test_connection_error_maps_to_could_not_reach_gemini(
        self, provider, mock_client, text_config, query_params
    ):
        mock_client.interactions.create.side_effect = compat_errors.APIConnectionError(
            request=GEMINI_REQUEST
        )

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith("[KAAPI] Could not reach Gemini (APIConnectionError)")

    def test_unsupported_completion_type_returns_kaapi_error(
        self, provider, mock_client, text_config, query_params
    ):
        text_config.type = "video"

        result, error = provider.execute(text_config, query_params, "Hi")

        assert result is None
        assert error.startswith("[KAAPI] Unsupported completion type 'video'")
        mock_client.interactions.create.assert_not_called()
