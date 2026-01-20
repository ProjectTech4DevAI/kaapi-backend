import logging

import openai
from openai import OpenAI
from google import genai
from typing import Any, Tuple
from openai.types.responses.response import Response

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
        """Initialize OpenAI provider with client.

        Args:
            client: OpenAI client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            return f"Gemini API Key not configured."
        return genai.Client(api_key=credentials["api_key"])

    def _parse_input(self, query_input, completion_type, provider) -> str:
        if completion_type == "stt":
            if isinstance(query_input, str):
                return query_input
            else:
                raise ValueError(f"{provider} STT require file path")

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        include_provider_raw_response: bool = False,
    ) -> tuple[Any, str | None]:
        response: Response | None = None
        error_message: str | None = None

        completion_type = completion_config.type
        provider = completion_config.provider

        generation_params = completion_config.params

        try:
            if completion_type == "stt":
                # query.input would be an audio_file object, or pathname

                parsed_input = self._parse_input(
                    query_input=query.input,
                    completion_type=completion_type,
                    provider=provider,
                )

                model = generation_params.get("model")
                instructions = generation_params.get("instructions")

                if not model:
                    return None, "Missing 'model' in native params"
                parsed_input = self._parse_input(query.input, completion_type, provider)

                gemini_file = self.client.files.upload(file=parsed_input)

                contents = []

                if instructions:
                    contents.append(instructions)
                contents.append(gemini_file)
                response = self.client.models.generate_content(
                    model=model, contents=contents
                )
                return response.text, None

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
