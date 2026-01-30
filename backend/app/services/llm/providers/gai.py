import logging
import os

from google import genai
from google.genai.types import GenerateContentResponse
from typing import Any

from app.models.llm import (
    NativeCompletionConfig,
    LLMCallResponse,
    QueryParams,
    LLMOutput,
    LLMResponse,
    Usage,
)
from app.services.llm.providers.base import BaseProvider


logger = logging.getLogger(__name__)


class GoogleAIProvider(BaseProvider):
    def __init__(self, client: genai.Client):
        """Initialize Google AI provider with client.

        Args:
            client: Google AI client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            raise ValueError("API Key for Google Gemini Not Set")
        return genai.Client(api_key=credentials["api_key"])

    def _parse_input(self, query_input, completion_type, provider) -> str:
        if completion_type == "stt":
            if isinstance(query_input, str):
                return query_input
            else:
                raise ValueError(f"{provider} STT require file path")

    def _execute_stt(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute speech-to-text completion using Google AI.

        Args:
            completion_config: Configuration for the completion request
            resolved_input: File path to the audio input
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (LLMCallResponse, error_message)
        """
        provider = completion_config.provider
        generation_params = completion_config.params

        # Parse and validate input
        parsed_input = self._parse_input(
            query_input=resolved_input,
            completion_type="stt",
            provider=provider,
        )

        model = generation_params.get("model")
        if not model:
            return None, "Missing 'model' in native params"

        instructions = generation_params.get("instructions", "")
        input_language = generation_params.get("input_language") or "auto"
        output_language = generation_params.get("output_language", "")

        # Build transcription/translation instruction
        if input_language == "auto":
            lang_instruction = (
                "Detect the spoken language automatically and transcribe the audio"
            )
        else:
            lang_instruction = f"Transcribe the audio from {input_language} in the native script of {input_language}"

        if output_language and output_language != input_language:
            lang_instruction += f" and translate to {output_language} in the native script of {output_language}"

        forced_trascription_text = "Only return transcribed text and no other text."
        # Merge user instructions with language instructions
        if instructions:
            merged_instruction = (
                f"{instructions}. {lang_instruction}. {forced_trascription_text}"
            )
        else:
            merged_instruction = f"{lang_instruction}. {forced_trascription_text}"

        # Upload file and generate content
        gemini_file = self.client.files.upload(file=parsed_input)

        contents = []
        if merged_instruction:
            contents.append(merged_instruction)
        contents.append(gemini_file)

        response: GenerateContentResponse = self.client.models.generate_content(
            model=model, contents=contents
        )

        # Build response
        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response.response_id,
                model=response.model_version,
                provider=provider,
                output=LLMOutput(text=response.text),
            ),
            usage=Usage(
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count,
                reasoning_tokens=response.usage_metadata.thoughts_token_count,
            ),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = response.model_dump()

        logger.info(
            f"[GoogleAIProvider._execute_stt] Successfully generated STT response: {response.response_id}"
        )

        return llm_response, None

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str,
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

        except TypeError as e:
            # handle unexpected arguments gracefully
            error_message = f"Invalid or unexpected parameter in Config: {str(e)}"
            return None, error_message

        except Exception as e:
            error_message = "Unexpected error occurred"
            logger.error(
                f"[GoogleAIProvider.execute] {error_message}: {str(e)}", exc_info=True
            )
            return None, error_message
