import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import requests

from app.core.audio_utils import (
    AudioRef,
    convert_pcm_to_mp3,
    convert_pcm_to_ogg,
    pcm_to_wav,
)
from app.core.cloud.storage import upload_audio_to_gcs
from app.core.config import settings
from app.models.llm import (
    LLMCallResponse,
    LLMResponse,
    NativeCompletionConfig,
    QueryParams,
    TextContent,
    TextOutput,
    Usage,
)
from app.models.llm.constants import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
)
from app.models.llm.response import AudioContent, AudioOutput
from app.services.llm.providers.base import BaseProvider, ContentPart, MultiModalInput

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60
SUPPORTED_AUDIO_MIMES = {
    "audio/wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
}


def _load_platform_sa_info() -> dict | None:
    """Load the platform-default GCP SA JSON.

    Supports two configuration shapes for settings.GCP_SA_KEY:
      1. Raw JSON string (e.g. injected via env var / secret manager)
      2. Filesystem path to a JSON key file
    """
    sa_value = settings.GCP_SA_KEY
    if not sa_value:
        return None

    stripped = sa_value.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.warning(
                f"[_load_platform_sa_info] GCP_SA_KEY looks like JSON but "
                f"failed to parse | error={e}"
            )
            return None


class VertexClient:
    """Holds Vertex AI connection details. Pure config — no SDK session.

    BYOK: per-project SA JSON + GCS bucket are passed via credentials and
    stored directly on the client; falls back to platform-shared values
    in settings when not provided by the project credential row.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        location: str,
        sa_info: dict | None = None,
        gcs_bucket: str | None = None,
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.sa_info = sa_info
        self.gcs_bucket = gcs_bucket or settings.GCS_AUDIO_BUCKET

    def endpoint(self, model: str) -> str:
        # The "global" location uses the unprefixed host; regional locations
        # use the "{location}-" prefix.
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{self.location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/v1"
            f"/projects/{self.project_id}/locations/{self.location}"
            f"/publishers/google/models/{model}:generateContent"
        )


class GoogleVertexAIProvider(BaseProvider):
    """Google Vertex AI provider using REST + API key auth.

    Supports STT (audio → text) and TTS (text → audio) via Gemini multimodal
    models on Vertex. Text-only completions are routed through the standard
    `google` provider.
    """

    def __init__(self, client: VertexClient):
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        # Fall back to platform-shared defaults from settings for any field
        # the caller didn't provide. The SA JSON falls back to the file at
        # settings.GCP_SA_KEY; BYOK rows pass `sa_key` inline.
        credentials = credentials or {}
        api_key = credentials.get("api_key") or settings.GCP_VERTEX_API_KEY
        logger.info(f"Vertex API Key {api_key}")
        project_id = credentials.get("project_id") or settings.GCP_PROJECT_ID
        location = credentials.get("location") or settings.GCP_VERTEX_LOCATION
        gcs_bucket = credentials.get("gcs_bucket") or settings.GCS_AUDIO_BUCKET
        sa_info = credentials.get("sa_key") or _load_platform_sa_info()

        source = "byok" if credentials.get("api_key") else "platform"
        logger.info(
            f"[create_client] vertex creds | source={source}, "
            f"project_id={project_id}, location={location}"
        )

        missing = [
            name
            for name, value in (
                ("api_key", api_key),
                ("project_id", project_id),
                ("location", location),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Google Vertex AI credentials missing required fields: {', '.join(missing)}"
            )
        return VertexClient(
            api_key=api_key,
            project_id=project_id,
            location=location,
            sa_info=sa_info,
            gcs_bucket=gcs_bucket,
        )

    def _post(self, model: str, payload: dict) -> tuple[dict | None, str | None]:
        url = self.client.endpoint(model)
        logger.debug(f"[_post] vertex url={url}")
        try:
            resp = requests.post(
                url,
                params={"key": self.client.api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            return None, f"Vertex AI request failed: {str(e)}"

        if not resp.ok:
            return None, f"Vertex AI HTTP {resp.status_code}: {resp.text[:500]}"

        try:
            return resp.json(), None
        except ValueError as e:
            return None, f"Vertex AI returned non-JSON response: {str(e)}"

    @staticmethod
    def _extract_usage(data: dict) -> Usage:
        meta = data.get("usageMetadata") or {}
        input_tokens = meta.get("promptTokenCount") or 0
        output_tokens = meta.get("candidatesTokenCount") or 0
        total_tokens = meta.get("totalTokenCount") or (input_tokens + output_tokens)
        reasoning_tokens = meta.get("thoughtsTokenCount") or 0
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
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
            return None, f"{provider} STT requires AudioRef input"

        mime_type = resolved_input.mime_type or "audio/wav"
        if mime_type not in SUPPORTED_AUDIO_MIMES:
            return None, (
                f"Unsupported audio mime '{mime_type}' for Vertex STT. "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_MIMES))}"
            )

        # Push bytes straight to GCS — no disk I/O. fileData.fileUri bypasses
        # the 20 MB inline cap.
        if not self.client.sa_info:
            return (
                None,
                "google-vertex sa_key not configured; cannot stage audio for STT",
            )
        try:
            gs_uri = upload_audio_to_gcs(
                audio_bytes=resolved_input.bytes_,
                bucket_name=self.client.gcs_bucket,
                sa_info=self.client.sa_info,
                project_id=self.client.project_id,
                content_type=mime_type,
            )
        except Exception as e:
            logger.error(
                f"[GoogleVertexAIProvider._execute_stt] GCS upload failed | "
                f"provider={provider}, error={e}",
                exc_info=True,
            )
            return None, f"Failed to stage audio for Vertex STT: {str(e)}"

        model = params.get("model") or DEFAULT_STT_MODEL
        instructions = params.get("instructions")
        input_language = params.get("input_language") or "auto"
        output_language = params.get("output_language")
        temperature = params.get("temperature")
        max_output_tokens = params.get("max_output_tokens") or 2048

        # Build transcription/translation instruction
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

        generation_config: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
        if temperature is not None:
            generation_config["temperature"] = temperature

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": mime_type, "fileUri": gs_uri}},
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": generation_config,
        }

        data, err = self._post(model, payload)
        logger.error(f"[_execute_stt] Error post making the call to Vertes is {err}")
        if err:
            return None, err

        try:
            transcript = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return None, "Vertex STT response missing transcript text"

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=data.get("responseId")
                or f"vertex-{uuid.uuid4().hex}",
                model=data.get("modelVersion") or model,
                provider=provider,
                output=TextOutput(content=TextContent(value=transcript.strip())),
            ),
            usage=self._extract_usage(data),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = data

        logger.info(
            f"[GoogleVertexAIProvider._execute_stt] Transcribed audio | provider={provider}, model={model}"
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
            return None, f"{provider} TTS requires text string as input"
        if not resolved_input.strip():
            return None, "Text input cannot be empty"

        model = params.get("model") or DEFAULT_TTS_MODEL
        voice = params.get("voice") or DEFAULT_TTS_VOICE
        language = params.get("language")
        response_format = params.get("response_format") or "wav"

        speech_config: dict[str, Any] = {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
        }
        if language:
            speech_config["languageCode"] = language

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": resolved_input}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": speech_config,
            },
        }

        provider_specific = params.get("provider_specific", {}) or {}
        gemini_params = provider_specific.get("gemini", {}) or {}
        director_notes = gemini_params.get("director_notes")
        if director_notes:
            payload["systemInstruction"] = {"parts": [{"text": director_notes}]}

        data, err = self._post(model, payload)
        if err:
            return None, err

        try:
            inline = data["candidates"][0]["content"]["parts"][0]["inlineData"]
            audio_b64 = inline["data"]
        except (KeyError, IndexError, TypeError):
            return None, "Vertex TTS response missing audio data"

        try:
            raw_pcm = base64.b64decode(audio_b64)
        except (ValueError, TypeError) as e:
            return None, f"Vertex TTS returned invalid base64 audio: {str(e)}"

        if not raw_pcm:
            return None, "Vertex TTS returned empty audio"

        actual_format = "wav"
        wav_bytes = pcm_to_wav(raw_pcm)
        encoded_content = base64.b64encode(wav_bytes).decode("ascii")

        if response_format == "mp3":
            converted, convert_err = convert_pcm_to_mp3(raw_pcm)
            if convert_err:
                return None, f"Failed to convert audio to MP3: {convert_err}"
            encoded_content = base64.b64encode(converted or b"").decode("ascii")
            actual_format = "mp3"
        elif response_format == "ogg":
            converted, convert_err = convert_pcm_to_ogg(raw_pcm)
            if convert_err:
                return None, f"Failed to convert audio to OGG: {convert_err}"
            encoded_content = base64.b64encode(converted or b"").decode("ascii")
            actual_format = "ogg"
        elif response_format and response_format != "wav":
            logger.warning(
                f"[GoogleVertexAIProvider._execute_tts] Unsupported response_format "
                f"'{response_format}', returning native WAV | provider={provider}"
            )

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=data.get("responseId")
                or f"vertex-{uuid.uuid4().hex}",
                model=data.get("modelVersion") or model,
                provider=provider,
                output=AudioOutput(
                    content=AudioContent(
                        format="base64",
                        value=encoded_content,
                        mime_type=f"audio/{actual_format}",
                    )
                ),
            ),
            usage=self._extract_usage(data),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = data

        logger.info(
            f"[GoogleVertexAIProvider._execute_tts] Synthesised audio | "
            f"provider={provider}, model={model}, format={actual_format}, "
            f"raw_pcm_bytes={len(raw_pcm)}"
        )
        return llm_response, None

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | list[ContentPart] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        try:
            completion_type = completion_config.type
            if completion_type == "stt":
                return self._execute_stt(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            if completion_type == "tts":
                return self._execute_tts(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            return (
                None,
                f"google-vertex provider does not support completion type "
                f"'{completion_type}'. Use the 'google' provider for text completions.",
            )
        except TypeError as e:
            return None, f"Invalid or unexpected parameter in Config: {str(e)}"
        except Exception as e:
            logger.error(
                f"[GoogleVertexAIProvider.execute] Unexpected error: {str(e)} | "
                f"provider={completion_config.provider}",
                exc_info=True,
            )
            return None, "Unexpected error occurred"
