import base64
import io
import json
import logging
import uuid
import wave
from typing import Any, cast

from google import genai
from google.genai import types as genai_types
from google.auth import exceptions as google_auth_errors

# Interactions raises its own error hierarchy, not google.genai.errors; in
# google-genai 2.29 this private module is the only path that exports it.
from google.genai._gaos.lib import compat_errors as interactions_errors
from google.genai.interactions import Interaction

from app.core.audio_utils import (
    AudioRef,
    convert_pcm_to_mp3,
    convert_pcm_to_ogg,
    pcm_to_wav,
)
from app.core.cloud.storage import build_gcp_sa_credentials, upload_audio_to_gcs
from app.core.config import settings
from app.models.llm import (
    LLMCallResponse,
    LLMResponse,
    NativeCompletionConfig,
    QueryParams,
    TextContent,
    TextOutput,
)
from app.models.llm.constants import (
    DEFAULT_STT_MODEL,
    DEFAULT_TEXT_MODELS,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    CompletionType,
)
from app.models.llm.response import AudioContent, AudioOutput
from app.services.llm.providers.base import BaseProvider, ContentPart, MultiModalInput
from app.services.llm.providers.google_aistudio import GoogleAIProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_MS = 60_000
DEFAULT_VERTEX_LOCATION = "global"
SUPPORTED_AUDIO_MIMES = {
    "audio/wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
}


def _parse_sa_info(value: Any) -> dict | None:
    """Accept the SA key as a dict (credential row) or a JSON string (env /
    secret manager). Unparseable values count as absent instead of raising, so a
    bad key degrades to the next auth option rather than failing client creation.
    """
    if isinstance(value, dict):
        return value or None
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.warning(f"[_parse_sa_info] SA key is not valid JSON | error={e}")
    return None


class GoogleGCPClient:
    """Vertex ``genai.Client`` plus the SA + bucket STT needs to stage audio in GCS."""

    def __init__(
        self,
        genai_client: genai.Client,
        project_id: str | None,
        sa_info: dict | None = None,
        gcs_bucket: str | None = None,
    ):
        self.genai = genai_client
        self.project_id = project_id
        self.sa_info = sa_info
        self.gcs_bucket = gcs_bucket


class GoogleGCPProvider(BaseProvider):
    """Google GCP (Vertex AI) provider on the google-genai Interactions API.

    Supports text, STT (audio → text), and TTS (text → audio) via Gemini
    multimodal models on GCP.
    """

    def __init__(self, client: GoogleGCPClient):
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        # Each field falls back to platform settings on its own, so a BYOK row
        # only needs the fields it wants to override.
        credentials = credentials or {}
        api_key = credentials.get("api_key") or settings.GCP_VERTEX_API_KEY
        sa_info = _parse_sa_info(credentials.get("sa_key")) or _parse_sa_info(
            settings.GCP_SA_KEY
        )
        project_id = (
            credentials.get("project_id")
            or settings.GCP_PROJECT_ID
            or (sa_info or {}).get("project_id")
        )
        location = (
            credentials.get("location")
            or settings.GCP_VERTEX_LOCATION
            or DEFAULT_VERTEX_LOCATION
        )
        gcs_bucket = credentials.get("gcs_bucket") or settings.GCS_AUDIO_BUCKET

        sa_credentials = None
        if sa_info:
            try:
                sa_credentials = build_gcp_sa_credentials(sa_info)
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"[create_client] google-gcp SA key is malformed, falling "
                    f"back to API key | error={e}"
                )
                sa_info = None

        http_options = genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        # The SDK rejects credentials + api_key together. Prefer the SA since STT
        # needs it for GCS anyway; otherwise use the API key (Vertex express mode).
        if sa_credentials and project_id:
            auth_mode = "service_account"
            genai_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=sa_credentials,
                http_options=http_options,
            )
        elif api_key:
            auth_mode = "api_key"
            genai_client = genai.Client(
                vertexai=True, api_key=api_key, http_options=http_options
            )
        else:
            raise ValueError(
                "Google GCP credentials missing: provide sa_key (with project_id) "
                "or api_key, on the project credentials or in platform settings."
            )

        logger.info(
            f"[create_client] google-gcp creds | "
            f"source={'byok' if credentials else 'platform'}, auth={auth_mode}, "
            f"project_id={project_id}, location={location}"
        )
        return GoogleGCPClient(
            genai_client=genai_client,
            project_id=project_id,
            sa_info=sa_info,
            gcs_bucket=gcs_bucket,
        )

    def _execute_stt(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: "AudioRef",
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider = completion_config.provider
        params = completion_config.params

        if not isinstance(resolved_input, AudioRef):
            error_message = (
                f"[KAAPI] STT validation failed: {provider} STT requires an "
                f"AudioRef input, but received {type(resolved_input).__name__}. "
                f"Ensure the audio is uploaded and resolved before invoking STT."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_stt] {error_message} | provider={provider}"
            )
            return None, error_message

        mime_type = resolved_input.mime_type or "audio/wav"
        if mime_type not in SUPPORTED_AUDIO_MIMES:
            error_message = (
                f"[KAAPI] STT validation failed: unsupported audio mime "
                f"'{mime_type}' for Google GCP STT. Supported MIME types are: "
                f"{', '.join(sorted(SUPPORTED_AUDIO_MIMES))}."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_stt] {error_message} | provider={provider}"
            )
            return None, error_message

        # Vertex reads audio from GCS by gs:// URI, which bypasses the inline size cap.
        if not self.client.sa_info or not self.client.gcs_bucket:
            error_message = (
                "[KAAPI] Google GCP STT staging failed: the ``google-gcp`` "
                "sa_key or gcs_bucket is not configured on this project's "
                "credentials or in platform settings, so audio cannot be "
                "uploaded to GCS for transcription."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_stt] {error_message} | provider={provider}"
            )
            return None, error_message

        try:
            gs_uri = upload_audio_to_gcs(
                audio_bytes=resolved_input.bytes_,
                bucket_name=self.client.gcs_bucket,
                sa_info=self.client.sa_info,
                project_id=self.client.project_id,
                content_type=mime_type,
            )
        except Exception as e:
            error_message = (
                f"[KAAPI] Failed to stage audio for Google GCP STT: GCS upload "
                f"to bucket '{self.client.gcs_bucket}' failed ({str(e)}). "
                f"Verify the service account has write access to the bucket "
                f"and that the bucket exists in project "
                f"'{self.client.project_id}'."
            )
            logger.error(
                f"[GoogleGCPProvider._execute_stt] {error_message} | provider={provider}",
                exc_info=True,
            )
            return None, error_message

        model = params.get("model") or DEFAULT_STT_MODEL
        instructions = params.get("instructions")
        input_language = params.get("input_language") or "auto"
        output_language = params.get("output_language")
        max_output_tokens = params.get("max_output_tokens") or 2048
        thinking_level = params.get("thinking_level") or "low"

        if input_language == "auto":
            lang_instruction = (
                "Detect the spoken language automatically and transcribe the audio"
            )
        else:
            lang_instruction = f"Transcribe the audio from {input_language} in the native script of {input_language}"

        if output_language and output_language != input_language:
            lang_instruction += (
                f" and translate to {output_language} in the native script of "
                f"{output_language} and only return transcribed script in {output_language}."
            )

        forced = "Only return transcribed text and no other text."
        if instructions:
            prompt = f"{instructions}. {lang_instruction}. {forced}"
        else:
            prompt = f"{lang_instruction}. {forced}"

        interaction = cast(
            Interaction,
            self.client.genai.interactions.create(
                model=model,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "audio", "uri": gs_uri, "mime_type": mime_type},
                        ],
                    }
                ],
                generation_config={
                    "thinking_level": thinking_level,
                    "max_output_tokens": max_output_tokens,
                },
            ),
        )

        # Vertex may not persist interactions, so the id can come back empty.
        response_id = interaction.id or f"google-gcp-{uuid.uuid4().hex}"
        if interaction.status != "completed" or not interaction.output_text:
            error_message = (
                f"[GOOGLE_GCP] STT response is missing transcribed text "
                f"(status={interaction.status}, errors={interaction.errors}). "
                f"This typically means the response was blocked by safety "
                f"filters or truncated. Review the prompt and safety settings, "
                f"then retry."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_stt] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response_id,
                model=interaction.model or model,
                provider=provider,
                output=TextOutput(
                    content=TextContent(
                        value=interaction.output_text.strip(),
                        language_code=output_language,
                    )
                ),
            ),
            usage=GoogleAIProvider._extract_usage(interaction, provider),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = interaction.model_dump(mode="json")

        logger.info(
            f"[GoogleGCPProvider._execute_stt] Transcribed audio | provider={provider}, model={model}"
        )
        return llm_response, None

    def _execute_tts(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider = completion_config.provider
        params = completion_config.params

        if not isinstance(resolved_input, str):
            error_message = (
                f"[KAAPI] TTS validation failed: {provider} TTS requires a "
                f"text string as input, but received "
                f"{type(resolved_input).__name__}. Provide the text to "
                f"synthesize as a plain string."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_tts] {error_message} | provider={provider}"
            )
            return None, error_message
        if not resolved_input.strip():
            error_message = (
                "[KAAPI] TTS validation failed: text input is empty or "
                "whitespace-only. Provide non-empty text to synthesize."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_tts] {error_message} | provider={provider}"
            )
            return None, error_message

        model = params.get("model") or DEFAULT_TTS_MODEL
        voice = params.get("voice") or DEFAULT_TTS_VOICE
        language = params.get("language")
        response_format = params.get("response_format") or "wav"

        provider_specific = params.get("provider_specific", {}) or {}
        gemini_params = provider_specific.get("gemini", {}) or {}
        director_notes = gemini_params.get("director_notes")

        transcript: dict[str, Any] = {
            "type": "text",
            "text": f"<transcript>{resolved_input}</transcript>",
        }
        if director_notes:
            transcript["annotations"] = [
                {"type": "speech_metadata", "style": director_notes}
            ]

        speech_config: dict[str, Any] = {"voice": voice}
        if language:
            speech_config["language"] = language

        interaction = cast(
            Interaction,
            self.client.genai.interactions.create(
                model=model,
                input=[{"type": "user_input", "content": [transcript]}],
                generation_config={"speech_config": [speech_config]},
                response_format={"type": "audio"},
            ),
        )

        response_id = interaction.id or f"google-gcp-{uuid.uuid4().hex}"
        audio_b64 = interaction.output_audio.data if interaction.output_audio else None
        if interaction.status != "completed" or not audio_b64:
            error_message = (
                f"[GOOGLE_GCP] TTS response is missing audio data (status="
                f"{interaction.status}, errors={interaction.errors}). Google GCP "
                f"was unable to generate audio from the input. Ensure the input "
                f"text is properly formatted and does not contain unsupported "
                f"control sequences."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        raw_pcm = base64.b64decode(audio_b64)
        # Preview TTS models return headerless 24 kHz PCM while 3.8+ return a
        # full WAV; normalise to PCM for pcm_to_wav / convert_pcm_to_*.
        if raw_pcm.startswith(b"RIFF"):
            with wave.open(io.BytesIO(raw_pcm), "rb") as wav_file:
                raw_pcm = wav_file.readframes(wav_file.getnframes())

        if not raw_pcm:
            error_message = (
                "[GOOGLE_GCP] TTS returned empty audio data. Google GCP accepted the "
                "request and returned a payload that decoded to zero bytes — "
                "this is typically a Google GCP server-side issue. Wait a "
                "minute and retry; if the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}"
            )
            return None, error_message

        actual_format = "wav"
        encoded_content = base64.b64encode(pcm_to_wav(raw_pcm)).decode("ascii")

        if response_format == "mp3":
            converted, convert_err = convert_pcm_to_mp3(raw_pcm)
            if convert_err:
                error_message = (
                    f"[KAAPI] Post-processing failure: unable to convert "
                    f"Google GCP PCM audio to MP3 ({convert_err}). Falling back "
                    f"to WAV is possible by setting response_format='wav'."
                )
                logger.error(
                    f"[GoogleGCPProvider._execute_tts] {error_message} | "
                    f"provider={provider}, model={model}, pcm_bytes={len(raw_pcm)}"
                )
                return None, error_message
            encoded_content = base64.b64encode(converted or b"").decode("ascii")
            actual_format = "mp3"
        elif response_format == "ogg":
            converted, convert_err = convert_pcm_to_ogg(raw_pcm)
            if convert_err:
                error_message = (
                    f"[KAAPI] Post-processing failure: unable to convert "
                    f"Google GCP PCM audio to OGG ({convert_err}). Falling back "
                    f"to WAV is possible by setting response_format='wav'."
                )
                logger.error(
                    f"[GoogleGCPProvider._execute_tts] {error_message} | "
                    f"provider={provider}, model={model}, pcm_bytes={len(raw_pcm)}"
                )
                return None, error_message
            encoded_content = base64.b64encode(converted or b"").decode("ascii")
            actual_format = "ogg"
        elif response_format and response_format != "wav":
            logger.warning(
                f"[GoogleGCPProvider._execute_tts] Unsupported response_format "
                f"'{response_format}', returning native WAV | provider={provider}"
            )

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response_id,
                model=interaction.model or model,
                provider=provider,
                output=AudioOutput(
                    content=AudioContent(
                        format="base64",
                        value=encoded_content,
                        mime_type=f"audio/{actual_format}",
                    )
                ),
            ),
            usage=GoogleAIProvider._extract_usage(interaction, provider),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = interaction.model_dump(mode="json")

        logger.info(
            f"[GoogleGCPProvider._execute_tts] Synthesised audio | "
            f"provider={provider}, model={model}, format={actual_format}, "
            f"raw_pcm_bytes={len(raw_pcm)}"
        )
        return llm_response, None

    def _execute_text(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str | list[ContentPart] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider = completion_config.provider
        params = completion_config.params
        model = params.get("model") or DEFAULT_TEXT_MODELS["google-gcp"]

        if isinstance(resolved_input, MultiModalInput):
            content = GoogleAIProvider.format_parts(resolved_input.parts)
        elif isinstance(resolved_input, list):
            content = GoogleAIProvider.format_parts(resolved_input)
        else:
            content = [{"type": "text", "text": resolved_input}]

        instructions = params.get("instructions")
        max_output_tokens = params.get("max_output_tokens")
        thinking_level = (params.get("thinking_config") or {}).get(
            "thinking_level"
        ) or params.get("reasoning")

        request: dict[str, Any] = {
            "model": model,
            "input": [{"type": "user_input", "content": content}],
        }
        if instructions:
            request["system_instruction"] = instructions

        generation_config: dict[str, Any] = {}
        if thinking_level:
            generation_config["thinking_level"] = thinking_level
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens
        if generation_config:
            request["generation_config"] = generation_config

        # create() is typed Interaction | Stream; we never pass stream=True.
        interaction = cast(
            Interaction, self.client.genai.interactions.create(**request)
        )

        response_id = interaction.id or f"google-gcp-{uuid.uuid4().hex}"
        if interaction.status != "completed" or not interaction.output_text:
            error_message = (
                f"[GOOGLE_GCP] Text response is missing generated content "
                f"(status={interaction.status}, errors={interaction.errors}). "
                f"This typically means the response was blocked by safety "
                f"filters or truncated by token limits. Review the prompt and "
                f"safety settings, then retry."
            )
            logger.warning(
                f"[GoogleGCPProvider._execute_text] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response_id,
                model=interaction.model or model,
                provider=provider,
                output=TextOutput(content=TextContent(value=interaction.output_text)),
            ),
            usage=GoogleAIProvider._extract_usage(interaction, provider),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = interaction.model_dump(mode="json")

        logger.info(
            f"[GoogleGCPProvider._execute_text] Generated text | "
            f"provider={provider}, model={model}"
        )
        return llm_response, None

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | list[ContentPart] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider = completion_config.provider
        completion_type = completion_config.type
        try:
            if completion_type == CompletionType.TEXT:
                return self._execute_text(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            if completion_type == CompletionType.STT:
                return self._execute_stt(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            if completion_type == CompletionType.TTS:
                return self._execute_tts(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            error_message = (
                f"[KAAPI] Unsupported completion type '{completion_type}' for "
                f"google-gcp provider. Google GCP supports 'text', 'stt' and 'tts'."
            )
            logger.warning(
                f"[GoogleGCPProvider.execute] {error_message} | provider={provider}"
            )
            return None, error_message

        except TypeError as e:
            error_message = (
                f"[KAAPI] Invalid or unexpected parameter in Config: {str(e)}. "
                f"Review the completion config; one of the parameters does "
                f"not match the Google GCP provider's expected signature."
            )
            logger.warning(
                f"[GoogleGCPProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except interactions_errors.APIStatusError as e:
            code = e.status_code
            if code == 400:
                error_message = (
                    f"[GOOGLE_GCP] Bad request (code: 400): {e.message}. Review "
                    f"your config parameters and input payload — the request "
                    f"shape, model, or content may be invalid for this Google "
                    f"GCP endpoint."
                )
            elif code in (401, 403):
                error_message = (
                    f"[GOOGLE_GCP] Authentication / permission denied (code: "
                    f"{code}): {e.message}. Verify the Google GCP API key or "
                    f"service account is valid, the project_id and location are "
                    f"correct, and the account has access to the requested model."
                )
            elif code == 404:
                error_message = (
                    f"[GOOGLE_GCP] Resource not found (code: 404): {e.message}. "
                    f"Check that the model exists and is available in your "
                    f"project and location."
                )
            elif code == 429:
                error_message = (
                    f"[GOOGLE_GCP] Rate limit / quota exceeded (code: 429): "
                    f"{e.message}. You have hit Google GCP's per-minute or "
                    f"per-day quota for this model. Wait at least 1 minute and "
                    f"retry; if the issue persists, request a quota increase "
                    f"from Google or contact Kaapi."
                )
            elif code is not None and code >= 500:
                error_message = (
                    f"[GOOGLE_GCP] Server error (code: {code}): {e.message}. "
                    f"This is typically transient (Google GCP overloaded or "
                    f"internal error) — retry in a few seconds. If the issue "
                    f"persists, contact Kaapi."
                )
            else:
                error_message = (
                    f"[GOOGLE_GCP] HTTP error (code: {code}): {e.message}. If "
                    f"the issue persists, contact Kaapi."
                )
            # 5xx is escalation-worthy; 4xx (including 429) is the caller's.
            log = logger.error if code is not None and code >= 500 else logger.warning
            log(
                f"[GoogleGCPProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}"
            )
            return None, error_message

        except interactions_errors.APITimeoutError as e:
            error_message = (
                f"[KAAPI] Google GCP request timed out after "
                f"{REQUEST_TIMEOUT_MS // 1000}s ({type(e).__name__}): {e}. "
                f"Retry with a smaller payload or contact Kaapi if the issue "
                f"persists."
            )
            logger.error(
                f"[GoogleGCPProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except interactions_errors.APIConnectionError as e:
            error_message = (
                f"[KAAPI] Google GCP connection failed ({type(e).__name__}): "
                f"{e}. Network or DNS issue reaching Google GCP — check network "
                f"connectivity from the Kaapi backend. If the issue persists, "
                f"contact Kaapi."
            )
            logger.error(
                f"[GoogleGCPProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except google_auth_errors.GoogleAuthError as e:
            # Raised while minting the SA access token, before any HTTP status.
            error_message = (
                f"[GOOGLE_GCP] Authentication failed ({type(e).__name__}): {e}. "
                f"Verify the service-account key is valid and not revoked."
            )
            logger.warning(
                f"[GoogleGCPProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}"
            )
            return None, error_message

        except Exception as e:
            error_message = (
                f"[KAAPI] Unexpected error while executing Google GCP "
                f"{completion_type or 'request'}: {str(e)}. This was not "
                f"raised by the Google GCP API — likely a Kaapi-side failure. "
                f"Contact Kaapi if the issue persists."
            )
            logger.error(
                f"[GoogleGCPProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message
