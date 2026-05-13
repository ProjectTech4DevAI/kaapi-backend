import logging

from sqlmodel import Session

from app.core.cloud.storage import get_cloud_storage
from app.core.db import engine
from app.crud.jobs import JobCrud
from app.crud.llm_chain import update_llm_chain_block_completed, update_llm_chain_status
from app.models import JobStatus, JobUpdate
from app.models.llm.request import (
    ChainStatus,
    LLMChainRequest,
)
from app.models.llm.response import (
    AudioOutput,
    IntermediateChainResponse,
    LLMChainResponse,
)
from app.services.llm.chain.chain import ChainContext, LLMChain
from app.services.llm.chain.types import BlockResult
from app.utils import APIResponse, get_webhook_secret, send_callback

logger = logging.getLogger(__name__)


class ChainExecutor:
    """Manage the lifecycle of an LLM chain execution."""

    def __init__(
        self,
        *,
        chain: LLMChain,
        context: ChainContext,
        request: LLMChainRequest,
    ):
        self._chain = chain
        self._context = context
        self._request = request
        self._webhook_secret: str | None = None

    def run(self) -> dict:
        """Execute the full chain lifecycle. Returns serialized APIResponse."""
        try:
            self._setup()

            result = self._chain.execute(
                self._request.query,
                on_block_completed=self._on_block_completed,
            )

            return self._teardown(result)

        except Exception as e:
            return self._handle_unexpected_error(e)

    def _setup(self) -> None:
        with Session(engine) as session:
            JobCrud(session).update(
                job_id=self._context.job_id,
                job_update=JobUpdate(status=JobStatus.PROCESSING),
            )

            update_llm_chain_status(
                session=session,
                chain_id=self._context.chain_id,
                status=ChainStatus.RUNNING,
            )

        self._webhook_secret = get_webhook_secret(
            self._context.project_id, self._context.organization_id
        )

    def _resolve_presigned_url(self, output) -> None:
        """Swap the s3:// URI in content.uri for a presigned URL in-place.

        Non-fatal: clears uri on failure so clients don't receive a raw s3:// address.
        """
        if isinstance(output, AudioOutput) and output.content.uri:
            try:
                with Session(engine) as session:
                    storage = get_cloud_storage(session, self._context.project_id)
                output.content.uri = storage.get_signed_url(
                    output.content.uri, expires_in=3600
                )
            except Exception as e:
                logger.warning(
                    f"[_resolve_presigned_url] Failed to generate presigned URL: {e} | "
                    f"job_id={self._context.job_id}",
                    exc_info=True,
                )
                output.content.uri = None

    def _teardown(self, result: BlockResult) -> dict:
        """Finalize chain record, send callback, and update job status."""

        if result.success:
            if result.response:
                self._resolve_presigned_url(result.response.response.output)

            final = LLMChainResponse(
                response=result.response.response,
                usage=result.usage,
                provider_raw_response=result.response.provider_raw_response,
            )
            callback_response = APIResponse.success_response(
                data=final, metadata=self._request.request_metadata
            )
            if self._request.callback_url:
                send_callback(
                    callback_url=str(self._request.callback_url),
                    data=callback_response.model_dump(),
                    webhook_secret=self._webhook_secret,
                )
            with Session(engine) as session:
                JobCrud(session).update(
                    job_id=self._context.job_id,
                    job_update=JobUpdate(status=JobStatus.SUCCESS),
                )
                update_llm_chain_status(
                    session=session,
                    chain_id=self._context.chain_id,
                    status=ChainStatus.COMPLETED,
                    output=result.response.response.output.model_dump(),
                    total_usage=self._context.aggregated_usage.model_dump(),
                )
            return callback_response.model_dump()
        else:
            return self._handle_error(result.error)

    def _handle_error(self, error: str) -> dict:
        callback_response = APIResponse.failure_response(
            error=error or "Unknown error occurred",
            metadata=self._request.request_metadata,
        )
        logger.warning(
            f"[_handle_error] Chain execution failed | "
            f"chain_id={self._context.chain_id}, job_id={self._context.job_id}, error={error}"
        )

        if self._request.callback_url:
            send_callback(
                callback_url=str(self._request.callback_url),
                data=callback_response.model_dump(),
                webhook_secret=self._webhook_secret,
            )

        with Session(engine) as session:
            update_llm_chain_status(
                session,
                chain_id=self._context.chain_id,
                status=ChainStatus.FAILED,
                output=None,
                total_usage=self._context.aggregated_usage.model_dump(),
                error=error,
            )
            JobCrud(session).update(
                job_id=self._context.job_id,
                job_update=JobUpdate(status=JobStatus.FAILED, error_message=error),
            )
        return callback_response.model_dump()

    def _on_block_completed(self, block_index: int, result: BlockResult) -> None:
        """Handle side effects after each block completes."""
        if result.usage:
            self._context.aggregated_usage.input_tokens += result.usage.input_tokens
            self._context.aggregated_usage.output_tokens += result.usage.output_tokens
            self._context.aggregated_usage.total_tokens += result.usage.total_tokens

        if result.success and result.llm_call_id:
            with Session(engine) as session:
                update_llm_chain_block_completed(
                    session,
                    chain_id=self._context.chain_id,
                    llm_call_id=result.llm_call_id,
                )

            if (
                block_index < len(self._context.intermediate_callback_flags)
                and self._context.intermediate_callback_flags[block_index]
                and self._request.callback_url
                and block_index < self._context.total_blocks - 1
            ):
                self._send_intermediate_callback(block_index, result)

    def _send_intermediate_callback(
        self, block_index: int, result: BlockResult
    ) -> None:
        """Send intermediate callback for a completed block."""
        try:
            if result.response:
                self._resolve_presigned_url(result.response.response.output)

            intermediate = IntermediateChainResponse(
                block_index=block_index + 1,
                total_blocks=self._context.total_blocks,
                response=result.response.response,
                usage=result.usage,
                provider_raw_response=result.response.provider_raw_response,
            )
            callback_data = APIResponse.success_response(
                data=intermediate,
                metadata=self._context.request_metadata,
            )
            send_callback(
                callback_url=str(self._request.callback_url),
                data=callback_data.model_dump(),
                webhook_secret=self._webhook_secret,
            )
            logger.info(
                f"[_send_intermediate_callback] Sent intermediate callback | "
                f"block={block_index + 1}/{self._context.total_blocks}, job_id={self._context.job_id}"
            )
        except Exception as e:
            logger.warning(
                f"[_send_intermediate_callback] Failed to send intermediate callback: {e} | "
                f"block={block_index + 1}/{self._context.total_blocks}, job_id={self._context.job_id}"
            )

    def _handle_unexpected_error(self, e: Exception) -> dict:
        logger.error(
            f"[ChainExecutor.run] Unexpected error: {e} | "
            f"job_id={self._context.job_id}",
            exc_info=True,
        )
        return self._handle_error("Unexpected error occurred")
