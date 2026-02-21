import logging

from sqlmodel import Session

from app.core.db import engine
from app.crud.jobs import JobCrud
from app.crud.llm_chain import update_llm_chain_status
from app.models import JobStatus, JobUpdate
from app.models.llm.request import (
    ChainStatus,
    LLMChainRequest,
)
from app.models.llm.response import LLMChainResponse
from app.services.llm.chain.chain import ChainContext, LLMChain
from app.services.llm.chain.types import BlockResult
from app.utils import APIResponse, send_callback

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

    def run(self) -> dict:
        """Execute the full chain lifecycle. Returns serialized APIResponse."""
        try:
            self._setup()

            result = self._chain.execute(self._request.query)

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

    def _teardown(self, result: BlockResult) -> dict:
        """Finalize chain record, send callback, and update job status."""

        with Session(engine) as session:
            if result.success:
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
                    )
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
        logger.error(
            f"[ChainExecutor] Chain execution failed | "
            f"chain_id={self._context.chain_id}, job_id={self._context.job_id}, error={error}"
        )

        with Session(engine) as session:
            if self._request.callback_url:
                send_callback(
                    callback_url=str(self._request.callback_url),
                    data=callback_response.model_dump(),
                )

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

    def _handle_unexpected_error(self, e: Exception) -> dict:
        logger.error(
            f"[ChainExecutor.run] Unexpected error: {e} | "
            f"job_id={self._context.job_id}",
            exc_info=True,
        )
        return self._handle_error("Unexpected error occurred")
