import logging
from typing import Any

import openai
from openai import OpenAI
from openai.types.responses.response import Response

from app.models.llm import (
    NativeCompletionConfig,
    LLMCallResponse,
    QueryParams,
    LLMResponse,
    Usage,
    TextOutput,
    TextContent,
    ImageContent,
    PDFContent,
)
from app.services.llm.providers.base import BaseProvider, ContentPart, MultiModalInput

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    def __init__(self, client: OpenAI):
        """Initialize OpenAI provider with client.

        Args:
            client: OpenAI client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            raise ValueError("OpenAI credentials not configured for this project.")
        return OpenAI(api_key=credentials["api_key"])

    @staticmethod
    def format_parts(
        parts: list[ContentPart],
    ) -> list[dict]:
        items = []
        for part in parts:
            if isinstance(part, TextContent):
                items.append({"type": "input_text", "text": part.value})

            elif isinstance(part, ImageContent):
                if part.format == "base64":
                    url = f"data:{part.mime_type};base64,{part.value}"
                else:
                    url = part.value
                items.append({"type": "input_image", "image_url": url})

            elif isinstance(part, PDFContent):
                if part.format == "base64":
                    url = f"data:{part.mime_type};base64,{part.value}"
                else:
                    url = part.value
                items.append({"type": "input_file", "file_url": url})

        return items

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | list[ImageContent] | list[PDFContent] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        response: Response | None = None
        error_message: str | None = None

        try:
            params = {
                **completion_config.params,
            }
            if isinstance(resolved_input, MultiModalInput):
                params["input"] = [
                    {"role": "user", "content": self.format_parts(resolved_input.parts)}
                ]
            elif isinstance(resolved_input, list):
                params["input"] = [
                    {"role": "user", "content": self.format_parts(resolved_input)}
                ]
            else:
                params["input"] = resolved_input

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
                    output=TextOutput(content=TextContent(value=response.output_text)),
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
