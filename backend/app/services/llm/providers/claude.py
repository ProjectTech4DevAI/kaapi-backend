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
from app.models.llm.constants import (
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    DEFAULT_TEXT_MODELS,
)
from app.services.llm.providers.base import BaseProvider, ContentPart, MultiModalInput

logger = logging.getLogger(__name__)

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

            # Anthropic requires model and max_tokens; default if caller did not supply
            params["model"] = params.get("model") or DEFAULT_TEXT_MODELS["anthropic"]
            params["max_tokens"] = (
                params.get("max_tokens") or DEFAULT_ANTHROPIC_MAX_TOKENS
            )

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
            error_message = (
                f"[KAAPI] Invalid or unexpected parameter in Config: {str(e)}. "
                f"Review the completion config; one of the parameters does "
                f"not match Anthropic's expected signature."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.BadRequestError as e:
            error_message = (
                f"[ANTHROPIC] Bad request (code: 400, request_id={e.request_id}): "
                f"{e.message}. Review your config parameters and input "
                f"payload — the request shape, model, or content may be "
                f"invalid for this Anthropic endpoint."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.AuthenticationError as e:
            error_message = (
                f"[ANTHROPIC] Authentication failed (code: 401, "
                f"request_id={e.request_id}): {e.message}. Verify the "
                f"Anthropic API key is valid, has not expired, and has been "
                f"correctly configured for this project."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.PermissionDeniedError as e:
            error_message = (
                f"[ANTHROPIC] Permission denied (code: 403, "
                f"request_id={e.request_id}): {e.message}. The API key does "
                f"not have access to the requested model or feature — check "
                f"your Anthropic plan and key scopes."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.NotFoundError as e:
            error_message = (
                f"[ANTHROPIC] Resource not found (code: 404, "
                f"request_id={e.request_id}): {e.message}. Verify the model "
                f"name and any referenced IDs (e.g. file_id) in your config "
                f"are correct and available on your Anthropic plan."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.ConflictError as e:
            error_message = (
                f"[ANTHROPIC] Conflict (code: 409, request_id={e.request_id}): "
                f"{e.message}. The request conflicts with the current "
                f"resource state — review concurrent requests before retrying."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.UnprocessableEntityError as e:
            error_message = (
                f"[ANTHROPIC] Unprocessable entity (code: 422, "
                f"request_id={e.request_id}): {e.message}. Anthropic rejected "
                f"the request payload — check input format, message shape, "
                f"and parameter values against the Messages API spec."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.RateLimitError as e:
            error_message = (
                f"[ANTHROPIC] Rate limit exceeded (code: 429, "
                f"request_id={e.request_id}): {e.message}. You have hit "
                f"Anthropic's request rate or token quota. Wait at least 1 "
                f"minute and retry; if the issue persists, request a quota "
                f"increase from Anthropic or contact Kaapi."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.InternalServerError as e:
            error_message = (
                f"[ANTHROPIC] Internal server error (code: {e.status_code}, "
                f"request_id={e.request_id}): {e.message}. This is typically "
                f"transient — retry in a few seconds. If the issue persists, "
                f"contact Kaapi."
            )
            logger.error(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.APITimeoutError as e:
            # Must come before APIConnectionError — APITimeoutError is a subclass.
            error_message = (
                f"[KAAPI] Anthropic request timed out (code: "
                f"{type(e).__name__}): {e.message}. The request took too "
                f"long to complete — retry with a smaller payload or shorter "
                f"max_tokens. If the issue persists, contact Kaapi."
            )
            logger.error(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.APIConnectionError as e:
            error_message = (
                f"[KAAPI] Anthropic connection failed (code: "
                f"{type(e).__name__}): {e.message}. Network or DNS issue "
                f"reaching Anthropic — check network connectivity from the "
                f"Kaapi backend. If the issue persists, contact Kaapi."
            )
            logger.error(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.APIResponseValidationError as e:
            error_message = (
                f"[ANTHROPIC] Returned a response that failed schema "
                f"validation (code: {e.status_code}): {e.message}. This "
                f"indicates an unexpected payload shape from Anthropic — "
                f"retry the request. If the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.APIStatusError as e:
            status = e.status_code
            if status == 413:
                error_message = (
                    f"[ANTHROPIC] Request too large (code: 413, "
                    f"request_id={e.request_id}): {e.message}. The input "
                    f"payload exceeds Anthropic's size limit — reduce prompt "
                    f"length, shrink attached files, or upload via the Files API."
                )
            elif status == 503:
                error_message = (
                    f"[ANTHROPIC] Service unavailable (code: 503, "
                    f"request_id={e.request_id}): {e.message}. Anthropic is "
                    f"temporarily down or undergoing maintenance — retry in "
                    f"a few seconds. If the issue persists, contact Kaapi."
                )
            elif status == 504:
                error_message = (
                    f"[ANTHROPIC] Deadline exceeded (code: 504, "
                    f"request_id={e.request_id}): {e.message}. Anthropic took "
                    f"too long to complete the request — retry with a smaller "
                    f"payload or shorter max_tokens."
                )
            elif status == 529:
                error_message = (
                    f"[ANTHROPIC] Overloaded (code: 529, "
                    f"request_id={e.request_id}): {e.message}. Anthropic's "
                    f"infrastructure is currently overloaded — this is "
                    f"transient. Retry with exponential backoff. If the "
                    f"issue persists, contact Kaapi."
                )
            else:
                error_message = (
                    f"[ANTHROPIC] API status error (code: {status}, "
                    f"request_id={e.request_id}): {e.message}. If the issue "
                    f"persists, contact Kaapi."
                )
            # 5xx server errors are escalation-worthy; 4xx (including 413)
            # are caller's fault and only need a warning.
            log = logger.error if status and status >= 500 else logger.warning
            log(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except anthropic.AnthropicError as e:
            # Final Anthropic catch-all for any non-APIStatusError SDK exception.
            error_message = (
                f"[ANTHROPIC] SDK error: {str(e)}. If the issue persists, "
                f"contact Kaapi."
            )
            logger.warning(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message

        except Exception as e:
            error_message = (
                f"[KAAPI] Unexpected error during Anthropic execution: "
                f"{str(e)}. This was not raised by the Anthropic SDK "
                f"directly — likely a Kaapi-side failure. Contact Kaapi if "
                f"the issue persists."
            )
            logger.error(
                f"[ClaudeProvider.execute] {error_message} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message
