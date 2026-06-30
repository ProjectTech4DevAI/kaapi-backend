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
    CompletionType,
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
        host = ""
        if self.location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{self.location}-aiplatform.googleapis.com"

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
        # settings.GCP_SA_KEY; BYOK rows pass `sa_key` inline.
        credentials = credentials or {}
        api_key = credentials.get("api_key") or settings.GCP_VERTEX_API_KEY
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

    def _post(
        self, model: str, payload: dict, log_context: str = ""
    ) -> tuple[dict | None, str | None]:
        """POST to Vertex generateContent and return parsed JSON or a
        descriptive, pre-logged error message.

        Maps:
        - ``requests.Timeout`` / ``ConnectionError`` / ``RequestException``
          → ``[KAAPI]`` network-side errors
        - HTTP 4xx/5xx → ``[VERTEX]`` errors, branched by status code, with
          Google's ``error.message`` / ``error.status`` surfaced when the
          response body is the standard error envelope.
        - Non-JSON 200 body → ``[VERTEX]`` malformed-response error
        """
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
        except requests.Timeout as e:
            error_message = (
                f"[KAAPI] Vertex AI request timed out after {REQUEST_TIMEOUT}s "
                f"(code: {type(e).__name__}): {str(e)}. The request took too "
                f"long to complete — retry with a smaller payload or contact "
                f"Kaapi if the issue persists."
            )
            logger.error(
                f"[GoogleVertexAIProvider._post] {error_message} | model={model}, {log_context}",
                exc_info=True,
            )
            return None, error_message
        except requests.ConnectionError as e:
            error_message = (
                f"[KAAPI] Vertex AI connection failed (code: "
                f"{type(e).__name__}): {str(e)}. Network or DNS issue "
                f"reaching Vertex — check network connectivity from the "
                f"Kaapi backend. If the issue persists, contact Kaapi."
            )
            logger.error(
                f"[GoogleVertexAIProvider._post] {error_message} | model={model}, {log_context}",
                exc_info=True,
            )
            return None, error_message
        except requests.RequestException as e:
            error_message = (
                f"[KAAPI] Vertex AI request failed (code: "
                f"{type(e).__name__}): {str(e)}. Unexpected requests-library "
                f"error — contact Kaapi if the issue persists."
            )
            logger.error(
                f"[GoogleVertexAIProvider._post] {error_message} | model={model}, {log_context}",
                exc_info=True,
            )
            return None, error_message

        if not resp.ok:
            status_code = resp.status_code
            google_msg = resp.text[:500]
            google_status = None
            try:
                err_envelope = resp.json().get("error", {})
                google_msg = err_envelope.get("message") or google_msg
                google_status = err_envelope.get("status")
            except (ValueError, AttributeError):
                # Body wasn't JSON or wasn't the expected envelope shape;
                # fall back to the raw text already captured above.
                pass

            status_label = f" {google_status}" if google_status else ""

            if status_code == 400:
                error_message = (
                    f"[VERTEX] Bad request (code: 400{status_label}): "
                    f"{google_msg}. Review your config parameters and input "
                    f"payload — the request shape, model, or content may be "
                    f"invalid for this Vertex endpoint."
                )
            elif status_code in (401, 403):
                error_message = (
                    f"[VERTEX] Authentication / permission denied (code: "
                    f"{status_code}{status_label}): {google_msg}. Verify the "
                    f"Vertex API key is valid and not expired, the project_id "
                    f"and location are correct, and the service account has "
                    f"access to the requested model."
                )
            elif status_code == 404:
                error_message = (
                    f"[VERTEX] Resource not found (code: 404{status_label}): "
                    f"{google_msg}. Check that the model '{model}' exists and "
                    f"is available in your project and location."
                )
            elif status_code == 429:
                error_message = (
                    f"[VERTEX] Rate limit / quota exceeded (code: 429"
                    f"{status_label}): {google_msg}. You have hit Vertex AI's "
                    f"per-minute or per-day quota for this model. Wait at "
                    f"least 1 minute and retry; if the issue persists, "
                    f"request a quota increase from Google or contact Kaapi."
                )
            elif 500 <= status_code < 600:
                error_message = (
                    f"[VERTEX] Server error (code: {status_code}"
                    f"{status_label}): {google_msg}. This is typically "
                    f"transient (Vertex overloaded or internal error) — "
                    f"retry in a few seconds. If the issue persists, contact "
                    f"Kaapi."
                )
            else:
                error_message = (
                    f"[VERTEX] HTTP error (code: {status_code}{status_label}): "
                    f"{google_msg}. If the issue persists, contact Kaapi."
                )

            # 5xx server errors are escalation-worthy; 4xx (including 429)
            # are caller's fault and only need a warning.
            log = logger.error if 500 <= status_code < 600 else logger.warning
            log(
                f"[GoogleVertexAIProvider._post] {error_message} | "
                f"model={model}, {log_context}"
            )
            return None, error_message

        try:
            return resp.json(), None
        except ValueError as e:
            error_message = (
                f"[VERTEX] Returned a non-JSON success response: {str(e)}. "
                f"This indicates an unexpected payload shape from Vertex — "
                f"retry the request. If the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._post] {error_message} | "
                f"model={model}, {log_context}"
            )
            return None, error_message

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
        """Execute STT via Vertex generateContent.

        Note:
            HTTP / network errors come back from ``_post()`` already-logged
            and tagged. This method only handles Kaapi-side input validation,
            staging failures, and Vertex response-shape checks.
        """
        provider = completion_config.provider
        params = completion_config.params

        if not isinstance(resolved_input, AudioRef):
            error_message = (
                f"[KAAPI] STT validation failed: {provider} STT requires an "
                f"AudioRef input, but received {type(resolved_input).__name__}. "
                f"Ensure the audio is uploaded and resolved before invoking STT."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_stt] {error_message} | provider={provider}"
            )
            return None, error_message

        mime_type = resolved_input.mime_type or "audio/wav"
        if mime_type not in SUPPORTED_AUDIO_MIMES:
            error_message = (
                f"[KAAPI] STT validation failed: unsupported audio mime "
                f"'{mime_type}' for Vertex STT. Supported MIME types are: "
                f"{', '.join(sorted(SUPPORTED_AUDIO_MIMES))}."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_stt] {error_message} | provider={provider}"
            )
            return None, error_message

        # Push bytes straight to GCS — no disk I/O. fileData.fileUri bypasses
        # the 20 MB inline cap.
        if not self.client.sa_info:
            error_message = (
                "[KAAPI] Vertex STT staging failed: ``google`` sa_key is "
                "not configured on this project's credentials, so audio "
                "cannot be uploaded to GCS for transcription. Add the "
                "service-account key to the project's ``google`` credentials."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_stt] {error_message} | provider={provider}"
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
                f"[KAAPI] Failed to stage audio for Vertex STT: GCS upload "
                f"to bucket '{self.client.gcs_bucket}' failed ({str(e)}). "
                f"Verify the service account has write access to the bucket "
                f"and that the bucket exists in project "
                f"'{self.client.project_id}'."
            )
            logger.error(
                f"[GoogleVertexAIProvider._execute_stt] {error_message} | provider={provider}",
                exc_info=True,
            )
            return None, error_message

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

        data, err = self._post(
            model, payload, log_context=f"provider={provider}, type=stt"
        )
        if err:
            return None, err

        try:
            transcript = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            error_message = (
                "[VERTEX] STT response is missing transcribed text. Vertex "
                "returned a 200 response but the expected "
                "candidates[0].content.parts[0].text path is absent — this "
                "typically means the response was blocked by safety filters "
                "or truncated. Review the prompt and safety settings, then "
                "retry."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_stt] {error_message} | "
                f"provider={provider}, model={model}, response_id={data.get('responseId')}"
            )
            return None, error_message

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
        """Execute TTS via Vertex generateContent.

        Note:
            HTTP / network errors come back from ``_post()`` already-logged
            and tagged. This method only handles Kaapi-side input validation,
            Vertex response-shape checks, and audio post-processing failures.
        """
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
                f"[GoogleVertexAIProvider._execute_tts] {error_message} | provider={provider}"
            )
            return None, error_message
        if not resolved_input.strip():
            error_message = (
                "[KAAPI] TTS validation failed: text input is empty or "
                "whitespace-only. Provide non-empty text to synthesize."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_tts] {error_message} | provider={provider}"
            )
            return None, error_message

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

        data, err = self._post(
            model, payload, log_context=f"provider={provider}, type=tts"
        )
        if err:
            return None, err

        try:
            inline = data["candidates"][0]["content"]["parts"][0]["inlineData"]
            audio_b64 = inline["data"]
        except (KeyError, IndexError, TypeError):
            error_message = (
                "[VERTEX] TTS response is missing audio data. Vertex returned "
                "a 200 response but the expected "
                "candidates[0].content.parts[0].inlineData path is absent — "
                "this typically means Vertex was unable to generate audio "
                "from the input. Ensure the input text is properly formatted "
                "and does not contain unsupported control sequences."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}, response_id={data.get('responseId')}"
            )
            return None, error_message

        try:
            raw_pcm = base64.b64decode(audio_b64)
        except (ValueError, TypeError) as e:
            error_message = (
                f"[VERTEX] TTS returned invalid base64 audio: {str(e)}. The "
                f"audio payload could not be decoded — this indicates a "
                f"corrupted response from Vertex. Retry the request; if the "
                f"issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}",
                exc_info=True,
            )
            return None, error_message

        if not raw_pcm:
            error_message = (
                "[VERTEX] TTS returned empty audio data. Vertex accepted the "
                "request and returned a base64 payload that decoded to zero "
                "bytes — this is typically a Vertex server-side issue. Wait "
                "a minute and retry; if the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleVertexAIProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}"
            )
            return None, error_message

        actual_format = "wav"
        wav_bytes = pcm_to_wav(raw_pcm)
        encoded_content = base64.b64encode(wav_bytes).decode("ascii")

        if response_format == "mp3":
            converted, convert_err = convert_pcm_to_mp3(raw_pcm)
            if convert_err:
                error_message = (
                    f"[KAAPI] Post-processing failure: unable to convert "
                    f"Vertex PCM audio to MP3 ({convert_err}). Falling back "
                    f"to WAV is possible by setting response_format='wav'."
                )
                logger.error(
                    f"[GoogleVertexAIProvider._execute_tts] {error_message} | "
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
                    f"Vertex PCM audio to OGG ({convert_err}). Falling back "
                    f"to WAV is possible by setting response_format='wav'."
                )
                logger.error(
                    f"[GoogleVertexAIProvider._execute_tts] {error_message} | "
                    f"provider={provider}, model={model}, pcm_bytes={len(raw_pcm)}"
                )
                return None, error_message
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
        provider = completion_config.provider
        completion_type = completion_config.type
        try:
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
                f"google provider. Vertex supports 'stt' and 'tts' only; "
                f"use the 'google-aistudio' provider for text completions."
            )
            logger.warning(
                f"[GoogleVertexAIProvider.execute] {error_message} | provider={provider}"
            )
            return None, error_message

        except TypeError as e:
            error_message = (
                f"[KAAPI] Invalid or unexpected parameter in Config: {str(e)}. "
                f"Review the completion config; one of the parameters does "
                f"not match the Vertex provider's expected signature."
            )
            logger.warning(
                f"[GoogleVertexAIProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except Exception as e:
            error_message = (
                f"[KAAPI] Unexpected error while executing Vertex "
                f"{completion_type or 'request'}: {str(e)}. This was not "
                f"raised inside the Vertex HTTP call — likely a Kaapi-side "
                f"failure. Contact Kaapi if the issue persists."
            )
            logger.error(
                f"[GoogleVertexAIProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message
