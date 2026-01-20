import logging

import openai
from openai import OpenAI
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


class OpenAIProvider(BaseProvider):
    def __init__(self, client: OpenAI):
        """Initialize OpenAI provider with client.

        Args:
            client: OpenAI client instance
        """
        super().__init__(client)
        self.client = client

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        response: Response | None = None
        error_message: str | None = None

        try:
            params = {
                **completion_config.params,
            }
            params["input"] = query.input

            conversation_cfg = query.conversation

            if conversation_cfg and conversation_cfg.id:
                params["conversation"] = {"id": conversation_cfg.id}

            elif conversation_cfg and conversation_cfg.auto_create:
                conversation = self.client.conversations.create()
                params["conversation"] = {"id": conversation.id}

            else:
                # only accept conversation_id if explicitly provided
                params.pop("conversation", None)

            response = self.client.responses.create(**params)

            conversation_id = (
                response.conversation.id if response.conversation else None
            )

            # Build response
            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=response.id,
                    conversation_id=conversation_id,
                    model=response.model,
                    provider=completion_config.provider,
                    output=LLMOutput(text=response.output_text),
                ),
                usage=Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
            )

            if include_provider_raw_response:
                llm_response.provider_raw_response = response.model_dump()

            logger.info(
                f"[OpenAIProvider.execute] Successfully generated response: {response.id}"
            )
            return llm_response, None

        except TypeError as e:
            # handle unexpected arguments gracefully
            error_message = f"Invalid or unexpected parameter in Config: {str(e)}"
            return None, error_message

        except openai.OpenAIError as e:
            # imported here to avoid circular imports
            from app.utils import handle_openai_error

            error_message = handle_openai_error(e)
            logger.error(
                f"[OpenAIProvider.execute] OpenAI API error: {error_message}",
                exc_info=True,
            )
            return None, error_message

        except Exception as e:
            error_message = "Unexpected error occurred"
            logger.error(
                f"[OpenAIProvider.execute] {error_message}: {str(e)}", exc_info=True
            )
            return None, error_message
