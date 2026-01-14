import logging
from typing import Any, Dict
from google import genai
from google.genai import types
from google.cloud.speech_v2 import SpeechClient
from app.models.llm import (
    NativeCompletionConfig,
    LLMCallResponse,
    QueryParams,
    LLMOutput,
    LLMResponse,
    Usage,
    KaapiCompletionConfig,
)
from app.services.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class GoogleAIProvider(BaseProvider):
    def __init__(self, client: genai.Client):
        super().__init__(client)

    @staticmethod
    def create_client(credentials: Dict[str, Any]) -> Any:
        """
        Initializes the Gemini client.
        Supports Gemini API Key

        """
        if "api_key" not in credentials:
            return f"Gemini API Key not set"

        return genai.Client(api_key=credentials["api_key"])

    def execute(
        self,
        completion_config: KaapiCompletionConfig,
        query: QueryParams,
        include_provider_raw_response: bool = False,
    ):
        try:
            params = completion_config.params
            thinking_config = None
            if params.reasoning:
                budget_map = {"low": 1024, "medium": 8192, "high": 24576}
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=budget_map.get(params.reasoning, 1024),
                )

            tools = []
            # TODO Figure out how knowledge base works in gemini
            # google search placeholder for now
            if (
                params.knowledge_base_ids
                and "google_search" in params.knowledge_base_ids
            ):
                tools.append(types.Tool(google_search=types.GoogleSearch()))

            # build the generation config
            generation_config = types.GenerateContentConfig(
                system_instruction=params.instructions,
                temperature=params.temperature
                if params.temperature is not None
                else 0.2,
                tools=tools if tools else None,
                thinking_config=thinking_config,
                candidate_count=params.max_num_results if params.max_num_results else 1,
            )

            #   gemini api call. imp to make sure the keys are correct.
            response = self.client.models.generate_content(
                model=params.model,
                contents=query.input if query.input is not None else "",
                config=generation_config,
            )

            final_text = ""
            reasoning_content = ""
            for part in response.candidates[0].content.parts:
                if part.thought:
                    reasoning_content += str(part.text)
                elif part.text:
                    final_text += str(part.text)

            llm_response_wrapper = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=response.response_id,
                    provider=params.model,
                    model=params.model,
                    output=LLMOutput(
                        text=final_text.strip(),
                        reasoning=reasoning_content.strip()
                        if reasoning_content
                        else None,
                    ),
                ),
                usage=Usage(
                    input_tokens=response.usage_metadata.prompt_token_count
                    if response.usage_metadata.prompt_token_count
                    else None,
                    output_tokens=response.usage_metadata.candidates_token_count
                    if response.usage_metadata.candidates_token_count
                    else None,
                    total_tokens=response.usage_metadata.total_token_count
                    if response.usage_metadata.total_token_count
                    else None,
                ),
                provider_raw_response=response.model_dump()
                if include_provider_raw_response
                else None,
            )
            logger.info(
                f"[GoogleAIProvider.execute] successfully generated response: {response.response_id}"
            )

            return llm_response_wrapper, None

        except Exception as e:
            logger.error(f"some error occured. Write an in depth error later {str(e)}")
            return
