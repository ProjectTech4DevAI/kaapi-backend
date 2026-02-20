import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlmodel import Session

from app.core.db import engine
from app.crud.llm_chain import update_llm_chain_block_completed
from app.models.llm.request import (
    LLMCallConfig,
    QueryParams,
    TextInput,
    TextContent,
    AudioInput,
)
from app.models.llm.response import (
    IntermediateChainResponse,
    TextOutput,
    AudioOutput,
    Usage,
)
from app.services.llm.chain.types import BlockResult
from app.services.llm.jobs import execute_llm_call
from app.utils import APIResponse, send_callback


logger = logging.getLogger(__name__)


@dataclass
class ChainContext:
    """Shared state passed to all blocks. Accumulates responses."""

    job_id: UUID
    chain_id: UUID
    project_id: int
    organization_id: int
    callback_url: str
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

    def on_block_completed(self, block_index: int, result: BlockResult) -> None:
        """Called after each block completes. Updates chain state in DB and sends intermediate callback."""

        if result.usage:
            self.aggregated_usage.input_tokens += result.usage.input_tokens
            self.aggregated_usage.output_tokens += result.usage.output_tokens
            self.aggregated_usage.total_tokens += result.usage.total_tokens

        if result.success and result.llm_call_id:
            with Session(engine) as session:
                update_llm_chain_block_completed(
                    session,
                    chain_id=self.chain_id,
                    llm_call_id=result.llm_call_id,
                )

            if (
                block_index < len(self.intermediate_callback_flags)
                and self.intermediate_callback_flags[block_index]
                and self.callback_url
            ):
                self._send_intermediate_callback(block_index, result)

    def _send_intermediate_callback(
        self, block_index: int, result: BlockResult
    ) -> None:
        """Send intermediate callback for a completed block."""
        try:
            intermediate = IntermediateChainResponse(
                block_index=block_index + 1,
                total_blocks=self.total_blocks,
                response=result.response.response,
                usage=result.usage,
                provider_raw_response=result.response.provider_raw_response,
            )
            callback_data = APIResponse.success_response(
                data=intermediate,
                metadata=self.request_metadata,
            )
            send_callback(
                callback_url=self.callback_url,
                data=callback_data.model_dump(),
            )
            logger.info(
                f"[ChainContext] Sent intermediate callback | "
                f"block={block_index + 1}/{self.total_blocks}, job_id={self.job_id}"
            )
        except Exception as e:
            logger.warning(
                f"[ChainContext] Failed to send intermediate callback: {e} | "
                f"block={block_index + 1}/{self.total_blocks}, job_id={self.job_id}"
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
    """A single node in the linked chain.

    Wraps execute_block() with linking capability.
    Each block knows its next block and forwards output to it.
    """

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
        self._next: ChainBlock | None = None

    def link(self, next_block: "ChainBlock") -> "ChainBlock":
        """Link to the next block in the chain."""
        self._next = next_block
        return next_block

    def execute(self, query: QueryParams) -> BlockResult:
        """Execute this block, then flow to next.

        No loop. Each block calls the next via the linked reference.
        Data flows through the chain like a linked list traversal.
        """
        logger.info(
            f"[ChainBlock.execute] Executing block {self._index} | "
            f"job_id={self._context.job_id}"
        )

        result = execute_llm_call(
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

        self._context.on_block_completed(self._index, result)

        if not result.success:
            logger.error(
                f"[ChainBlock.execute] Block {self._index} failed: {result.error} | "
                f"job_id={self._context.job_id}"
            )
            return result

        if self._next:
            next_query = result_to_query(result)
            return self._next.execute(next_query)

        logger.info(
            f"[ChainBlock.execute] Block {self._index} is the last block | "
            f"job_id={self._context.job_id}"
        )
        return result


class LLMChain:
    """Links ChainBlocks together into a sequential chain.

    Construction builds the linked structure.
    Execution pushes input into the head — it flows through to the tail.
    """

    def __init__(self, blocks: list[ChainBlock]):
        self._head: ChainBlock | None = None
        self._tail: ChainBlock | None = None
        self._link_blocks(blocks)

    def _link_blocks(self, blocks: list[ChainBlock]) -> None:
        """Link all blocks in sequence."""
        if not blocks:
            return
        self._head = blocks[0]
        self._tail = blocks[-1]
        prev = blocks[0]
        for curr in blocks[1:]:
            prev.link(curr)
            prev = curr

    def execute(self, query: QueryParams) -> BlockResult:
        """Push input into the chain head. It flows through to the tail."""
        if not self._head:
            return BlockResult(error="Chain has no blocks")
        return self._head.execute(query)
