import base64
import io
import logging
from typing import Any

import anthropic
from anthropic import Anthropic
from anthropic.types import Message

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

DEFAULT_MAX_TOKENS = 4096
FILES_API_BETA = "files-api-2025-04-14"


class ClaudeProvider(BaseProvider):
    def __init__(self, client: Anthropic):
        """Initialize Anthropic Claude provider with client.

        Args:
            client: Anthropic client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            raise ValueError("Anthropic credentials not configured for this project.")
        return Anthropic(api_key=credentials["api_key"])

    @staticmethod
    def format_parts(
        parts: list[ContentPart],
    ) -> list[dict]:
        items = []
        for part in parts:
            if isinstance(part, TextContent):
                items.append({"type": "text", "text": part.value})

            elif isinstance(part, ImageContent):
                if part.format == "base64":
                    items.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.mime_type,
                                "data": part.value,
                            },
                        }
                    )
                else:
                    items.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": part.value},
                        }
                    )

            elif isinstance(part, PDFContent):
                if part.format == "base64":
                    items.append(
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": part.mime_type,
                                "data": part.value,
                            },
                        }
                    )
                else:
                    items.append(
                        {
                            "type": "document",
                            "source": {"type": "url", "url": part.value},
                        }
                    )

        return items

    @staticmethod
    def _is_base64_file_block(block: Any) -> bool:
        if not isinstance(block, dict):
            return False
        if block.get("type") not in ("document", "image"):
            return False
        source = block.get("source") or {}
        return source.get("type") == "base64"

    def _upload_to_files_api(self, source: dict, block_type: str) -> str:
        file_bytes = base64.b64decode(source["data"])
        filename = "document.pdf" if block_type == "document" else "image"
        upload = self.client.beta.files.upload(
            file=(filename, io.BytesIO(file_bytes), source["media_type"]),
        )
        logger.info(
            f"[ClaudeProvider._upload_to_files_api] Uploaded {block_type} | file_id={upload.id}"
        )
        return upload.id

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | list[ContentPart] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        response: Message | None = None
        error_message: str | None = None

        try:
            params = {**completion_config.params}

            # Anthropic requires max_tokens; default if caller did not supply
            params.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

            # Kaapi exposes "instructions"; Anthropic uses "system". Always
            # strip "instructions" — Anthropic rejects unknown kwargs.
            if "instructions" in params:
                if "system" not in params:
                    params["system"] = params["instructions"]
                params.pop("instructions")

            if isinstance(resolved_input, MultiModalInput):
                content = self.format_parts(resolved_input.parts)
            elif isinstance(resolved_input, list):
                content = self.format_parts(resolved_input)
            else:
                content = resolved_input

            # Upload any base64 PDFs/images to the Files API and reference by file_id.
            # Keeps request payloads small and lets large files bypass inline size limits.
            uploaded_file = False
            if isinstance(content, list):
                for block in content:
                    if not self._is_base64_file_block(block):
                        continue

                    file_id = self._upload_to_files_api(block["source"], block["type"])
                    block["source"] = {"type": "file", "file_id": file_id}
                    uploaded_file = True

            params["messages"] = [{"role": "user", "content": content}]

            # Anthropic Messages API has no first-class conversation primitive,
            # callers must replay prior messages themselves. Strip conversation
            # config so it never leaks into the API call.
            params.pop("conversation", None)

            if uploaded_file:
                existing_betas = params.pop("betas", []) or []
                params["betas"] = [*existing_betas, FILES_API_BETA]
                response = self.client.beta.messages.create(**params)
            else:
                response = self.client.messages.create(**params)

            output_text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=response.id,
                    conversation_id=None,
                    model=response.model,
                    provider=completion_config.provider,
                    output=TextOutput(content=TextContent(value=output_text)),
                ),
                usage=Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens
                    + response.usage.output_tokens,
                ),
            )

            if include_provider_raw_response:
                llm_response.provider_raw_response = response.model_dump()

            logger.info(
                f"[ClaudeProvider.execute] Successfully generated response | "
                f"request_id={response.id}, provider={completion_config.provider}, model={response.model}"
            )
            return llm_response, None

        except TypeError as e:
            error_message = f"Invalid or unexpected parameter in Config: {str(e)}"
            return None, error_message

        except anthropic.AnthropicError as e:
            error_message = f"Anthropic API error: {str(e)}"
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except Exception as e:
            error_message = "Unexpected error occurred"
            logger.error(
                f"[ClaudeProvider.execute] {error_message}: {str(e)} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message
