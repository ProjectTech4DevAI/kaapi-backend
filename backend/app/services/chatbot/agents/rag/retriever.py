"""Knowledge-base search over OpenAI vector stores; Anthropic has no native file search."""

import html
import logging

import openai
from langchain_core.tools import BaseTool, StructuredTool
from openai import OpenAI
from sqlmodel import Session

from app.crud import get_provider_credential
from app.crud.rag.open_ai import OPENAI_TIMEOUT_SECONDS
from app.models.llm.constants import Provider
from app.services.chatbot.agents.rag.prompts import (
    NO_PASSAGES_FOUND,
    PASSAGE_SEPARATOR,
    PASSAGE_TEMPLATE,
    SEARCH_TOOL_DESCRIPTION,
    SEARCH_TOOL_NAME,
)
from app.services.chatbot.agents.rag.types import (
    KnowledgeBasePassage,
    SearchKnowledgeBaseArgs,
)
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.provider_errors import PROVIDER_SERVER_ERROR_STATUS
from app.utils import mask_string

logger = logging.getLogger(__name__)

DEFAULT_MAX_NUM_RESULTS = 5
OPENAI_MAX_NUM_RESULTS = 50
# A search is an idempotent read, so one SDK retry absorbs a transient blip
# without stretching the turn far past OPENAI_TIMEOUT_SECONDS.
KB_SEARCH_MAX_RETRIES = 1
CREDENTIAL_API_KEY = "api_key"
PASSAGE_CHUNK_SEPARATOR = "\n"


def get_knowledge_base_client(
    *, session: Session, organization_id: int, project_id: int
) -> OpenAI:
    """OpenAI client from the project's credentials, used only for vector-store search."""
    credentials = get_provider_credential(
        session=session,
        org_id=organization_id,
        project_id=project_id,
        provider=Provider.OPENAI.value,
    )
    if not isinstance(credentials, dict) or not credentials.get(CREDENTIAL_API_KEY):
        error_message = (
            "[KAAPI] The rag agent searches knowledge bases with the project's OpenAI "
            "credentials, but none are configured; add OpenAI credentials to this project."
        )
        logger.warning(
            f"[get_knowledge_base_client] {error_message} | project_id: {project_id}"
        )
        raise ChatbotInputError(error_message)

    api_key = credentials[CREDENTIAL_API_KEY]
    logger.info(
        f"[get_knowledge_base_client] OpenAI client configured | "
        f"project_id: {project_id}, api_key: {mask_string(api_key)}"
    )
    return OpenAI(
        api_key=api_key,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=KB_SEARCH_MAX_RETRIES,
    )


def _resolve_top_k(max_num_results: int | None) -> int:
    if max_num_results is None:
        return DEFAULT_MAX_NUM_RESULTS
    return min(max_num_results, OPENAI_MAX_NUM_RESULTS)


def _to_search_error(error: openai.OpenAIError, *, vector_store_id: str) -> str:
    # APITimeoutError subclasses APIConnectionError, so it must be checked first.
    if isinstance(error, openai.APITimeoutError):
        error_message = (
            "[KAAPI] Knowledge base search timed out — retry the turn. If persistent, "
            "contact Kaapi. (code: APITimeoutError)"
        )
        logger.error(
            f"[search_knowledge_bases] {error_message} | provider=openai, "
            f"vector_store_id: {vector_store_id}",
            exc_info=error,
        )
        return error_message
    if isinstance(error, openai.APIConnectionError):
        error_message = (
            "[KAAPI] Network/DNS issue reaching OpenAI for knowledge base search — "
            "check connectivity. If persistent, contact Kaapi. (code: APIConnectionError)"
        )
        logger.error(
            f"[search_knowledge_bases] {error_message} | provider=openai, "
            f"vector_store_id: {vector_store_id}",
            exc_info=error,
        )
        return error_message
    if isinstance(error, openai.APIStatusError):
        error_message = (
            f"[OPENAI] Knowledge base search failed for '{vector_store_id}': "
            f"{error.message} (code: {error.status_code}, request_id: {error.request_id})"
        )
        # 5xx is provider-side (alert-worthy); 4xx is caller's fault (noise if alerted)
        log = (
            logger.error
            if error.status_code >= PROVIDER_SERVER_ERROR_STATUS
            else logger.warning
        )
        log(
            f"[search_knowledge_bases] {error_message} | provider=openai, "
            f"vector_store_id: {vector_store_id}",
            exc_info=error,
        )
        return error_message

    error_message = (
        f"[OPENAI] Knowledge base search failed for '{vector_store_id}': {error} "
        f"— retry the turn. If persistent, contact Kaapi. (code: {type(error).__name__})"
    )
    logger.error(
        f"[search_knowledge_bases] {error_message} | provider=openai, "
        f"vector_store_id: {vector_store_id}",
        exc_info=error,
    )
    return error_message


def search_knowledge_bases(
    *,
    client: OpenAI,
    knowledge_base_ids: tuple[str, ...],
    query: str,
    max_num_results: int | None,
) -> list[KnowledgeBasePassage]:
    """Search every knowledge base and keep the overall top passages by score."""
    top_k = _resolve_top_k(max_num_results)
    passages: list[KnowledgeBasePassage] = []
    for vector_store_id in knowledge_base_ids:
        logger.info(
            f"[search_knowledge_bases] Searching knowledge base | "
            f"vector_store_id: {vector_store_id}, max_num_results: {top_k}"
        )
        try:
            page = client.vector_stores.search(
                vector_store_id, query=query, max_num_results=top_k
            )
        except openai.OpenAIError as e:
            error_message = _to_search_error(e, vector_store_id=vector_store_id)
            raise ChatbotProviderError(error_message) from e

        for result in page.data:
            chunk_texts: list[str] = []
            for chunk in result.content:
                chunk_texts.append(chunk.text)
            passages.append(
                KnowledgeBasePassage(
                    file_id=result.file_id,
                    filename=result.filename,
                    score=result.score,
                    text=PASSAGE_CHUNK_SEPARATOR.join(chunk_texts),
                )
            )

    passages.sort(key=lambda passage: passage.score, reverse=True)
    top_passages = passages[:top_k]
    logger.info(
        f"[search_knowledge_bases] Knowledge base search finished | "
        f"knowledge_bases: {len(knowledge_base_ids)}, passages: {len(top_passages)}"
    )
    return top_passages


def format_passages(passages: list[KnowledgeBasePassage]) -> str:
    if not passages:
        return NO_PASSAGES_FOUND

    rendered: list[str] = []
    for passage in passages:
        rendered.append(
            PASSAGE_TEMPLATE.format(
                filename=html.escape(passage.filename, quote=True), text=passage.text
            )
        )
    return PASSAGE_SEPARATOR.join(rendered)


def build_search_tool(
    *,
    client: OpenAI,
    knowledge_base_ids: tuple[str, ...],
    max_num_results: int | None,
) -> BaseTool:
    """`search_knowledge_base` tool: passages as content, the structured hits as artifact."""

    def search_knowledge_base(
        query: str,
    ) -> tuple[str, list[KnowledgeBasePassage]]:
        passages = search_knowledge_bases(
            client=client,
            knowledge_base_ids=knowledge_base_ids,
            query=query,
            max_num_results=max_num_results,
        )
        return format_passages(passages), passages

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name=SEARCH_TOOL_NAME,
        description=SEARCH_TOOL_DESCRIPTION,
        args_schema=SearchKnowledgeBaseArgs,
        response_format="content_and_artifact",
    )
