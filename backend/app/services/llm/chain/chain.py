import logging
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID

from app.models.llm.request import (
    LLMCallConfig,
    QueryParams,
    TextInput,
    TextContent,
    AudioInput,
)
from app.models.llm.response import (
    TextOutput,
    AudioOutput,
    Usage,
)
from app.services.llm.chain.types import BlockResult
from app.services.llm.jobs import execute_llm_call


logger = logging.getLogger(__name__)


@dataclass
class ChainContext:
    """Shared state for chain execution."""

    job_id: UUID
    chain_id: UUID
    project_id: int
    organization_id: int
    callback_url: str | None
    total_blocks: int

    langfuse_credentials: dict[str, Any] | None = None
    request_metadata: dict | None = None
    intermediate_callback_flags: list[bool] = field(default_factory=list)
    aggregated_usage: Usage = field(
        default_factory=lambda: Usage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
    )


def result_to_query(result: BlockResult) -> QueryParams:
    """Convert a block's output into the next block's QueryParams.

    Text output → TextInput query
    Audio output → AudioInput query
    """
    output = result.response.response.output

    if isinstance(output, TextOutput):
        return QueryParams(
            input=TextInput(content=TextContent(value=output.content.value))
        )
    elif isinstance(output, AudioOutput):
        return QueryParams(input=AudioInput(content=output.content))
    else:
        raise ValueError(f"Cannot chain output type: {output.type}")


class ChainBlock:
    """A single block in the chain. Only responsible for executing itself."""

    def __init__(
        self,
        *,
        config: LLMCallConfig,
        index: int,
        context: ChainContext,
        include_provider_raw_response: bool = False,
    ):
        self._config = config
        self._index = index
        self._context = context
        self._include_provider_raw_response = include_provider_raw_response

    def execute(self, query: QueryParams) -> BlockResult:
        """Execute this block and return the result."""
        logger.info(
            f"[ChainBlock.execute] Executing block {self._index} | "
            f"job_id={self._context.job_id}"
        )

        return execute_llm_call(
            config=self._config,
            query=query,
            job_id=self._context.job_id,
            project_id=self._context.project_id,
            organization_id=self._context.organization_id,
            request_metadata=self._context.request_metadata,
            langfuse_credentials=self._context.langfuse_credentials,
            include_provider_raw_response=self._include_provider_raw_response,
            chain_id=self._context.chain_id,
        )


class LLMChain:
    """Orchestrates sequential execution of ChainBlocks."""

    def __init__(self, blocks: list[ChainBlock], context: ChainContext):
        self._blocks = blocks
        self._context = context

    def execute(
        self,
        query: QueryParams,
        on_block_completed: Callable[[int, BlockResult], None] | None = None,
    ) -> BlockResult:
        """Execute blocks sequentially, passing output of each to the next."""
        if not self._blocks:
            return BlockResult(error="Chain has no blocks")

        current_query = query
        result: BlockResult | None = None

        for block in self._blocks:
            result = block.execute(current_query)

            if on_block_completed:
                on_block_completed(block._index, result)

            if not result.success:
                logger.warning(
                    f"[LLMChain.execute] Block {block._index} failed: {result.error} | "
                    f"job_id={self._context.job_id}"
                )
                return result

            if block is not self._blocks[-1]:
                current_query = result_to_query(result)

        return result
