import logging
from typing import Any
from uuid import UUID

from sqlmodel import Session

from app.core.util import now
from app.models.llm.request import ChainStatus, LlmChain

logger = logging.getLogger(__name__)


def create_llm_chain(
    session: Session,
    *,
    job_id: UUID,
    project_id: int,
    organization_id: int,
    total_blocks: int,
    input: str,
    configs: list[dict[str, Any]],
) -> LlmChain:
    """Create a new LLM chain record.
    Args:
        session: Database session
        job_id: Reference to the parent job
        project_id: Reference to the project
        organization_id: Reference to the organization
        total_blocks: Total number of blocks to execute
        input: Serialized input string (via serialize_input)
        configs: Ordered list of block configs as submitted

    Returns:
        LlmChain: The created chain record
    """
    db_llm_chain = LlmChain(
        job_id=job_id,
        project_id=project_id,
        organization_id=organization_id,
        status=ChainStatus.PENDING,
        total_blocks=total_blocks,
        number_of_blocks_processed=0,
        input=input,
        configs=configs,
        block_sequences=[],
    )

    session.add(db_llm_chain)
    session.commit()
    session.refresh(db_llm_chain)

    logger.info(
        f"[create_llm_chain] Created LLM chain id={db_llm_chain.id}, "
        f"job_id={job_id}, total_blocks={total_blocks}"
    )

    return db_llm_chain


def update_llm_chain_status(
    session: Session,
    *,
    chain_id: UUID,
    status: ChainStatus,
    output: dict[str, Any] | None = None,
    total_usage: dict[str, Any] | None = None,
    error: str | None = None,
) -> LlmChain:
    """Update chain record status and related fields.
    Args:
        session: Database session
        chain_id: The chain record ID
        status: New chain status
        output: Last block's output dict (only for COMPLETED)
        total_usage: Aggregated token usage across all blocks (for COMPLETED/FAILED)
        error: Error message (only for FAILED)

    Returns:
        LlmChain: The updated chain record
    """
    db_chain = session.get(LlmChain, chain_id)
    if not db_chain:
        raise ValueError(f"LLM chain not found with id={chain_id}")

    db_chain.status = status
    db_chain.updated_at = now()

    if status == ChainStatus.FAILED:
        db_chain.error = error
        db_chain.total_usage = total_usage

    if status == ChainStatus.COMPLETED:
        db_chain.output = output
        db_chain.total_usage = total_usage

    session.add(db_chain)
    session.commit()
    session.refresh(db_chain)

    logger.info(
        f"[update_llm_chain_status] Chain {chain_id} → {status.value} | "
        f"has_output={output is not None}, "
        f"blocks={db_chain.number_of_blocks_processed}/{db_chain.total_blocks}, "
        f"error={error}"
    )
    return db_chain


def update_llm_chain_block_completed(
    session: Session,
    *,
    chain_id: UUID,
    llm_call_id: UUID,
) -> LlmChain:
    """Update chain progress after a block completes.
    Args:
        session: Database session
        chain_id: The chain record ID
        llm_call_id: The llm_call record ID for the completed block

    Returns:
        LlmChain: The updated chain record
    """
    db_chain = session.get(LlmChain, chain_id)
    if not db_chain:
        raise ValueError(f"LLM chain not found with id={chain_id}")

    # Append to block_sequences
    sequences = list(db_chain.block_sequences or [])
    sequences.append(str(llm_call_id))
    db_chain.block_sequences = sequences

    # Increment progress
    db_chain.number_of_blocks_processed = len(sequences)
    db_chain.updated_at = now()

    session.add(db_chain)
    session.commit()
    session.refresh(db_chain)

    logger.info(
        f"[update_llm_chain_block_completed] Chain {chain_id} | "
        f"block={db_chain.number_of_blocks_processed}/{db_chain.total_blocks}, "
        f"llm_call_id={llm_call_id}"
    )
    return db_chain
