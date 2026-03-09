import pytest
from uuid import uuid4

from sqlmodel import Session

from app.crud import JobCrud
from app.crud.llm_chain import (
    create_llm_chain,
    update_llm_chain_status,
    update_llm_chain_block_completed,
)
from app.models import JobType
from app.models.llm.request import ChainStatus
from app.tests.utils.utils import get_project


class TestCreateLlmChain:
    def test_creates_chain_record(self, db: Session):
        project = get_project(db)
        job = JobCrud(session=db).create(
            job_type=JobType.LLM_CHAIN, trace_id="test-trace"
        )
        db.commit()

        chain = create_llm_chain(
            db,
            job_id=job.id,
            project_id=project.id,
            organization_id=project.organization_id,
            total_blocks=3,
            input="Test input",
            configs=[{"completion": {"provider": "openai-native"}}],
        )

        assert chain.id is not None
        assert chain.job_id == job.id
        assert chain.project_id == project.id
        assert chain.status == ChainStatus.PENDING
        assert chain.total_blocks == 3
        assert chain.number_of_blocks_processed == 0
        assert chain.input == "Test input"
        assert chain.block_sequences == []


class TestUpdateLlmChainStatus:
    @pytest.fixture
    def chain(self, db: Session):
        project = get_project(db)
        job = JobCrud(session=db).create(
            job_type=JobType.LLM_CHAIN, trace_id="test-trace"
        )
        db.commit()
        chain = create_llm_chain(
            db,
            job_id=job.id,
            project_id=project.id,
            organization_id=project.organization_id,
            total_blocks=2,
            input="hello",
            configs=[],
        )
        return chain

    def test_update_to_running(self, db: Session, chain):
        updated = update_llm_chain_status(
            db, chain_id=chain.id, status=ChainStatus.RUNNING
        )

        assert updated.status == ChainStatus.RUNNING

    def test_update_to_completed(self, db: Session, chain):
        output = {"type": "text", "content": {"value": "result"}}
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

        updated = update_llm_chain_status(
            db,
            chain_id=chain.id,
            status=ChainStatus.COMPLETED,
            output=output,
            total_usage=usage,
        )

        assert updated.status == ChainStatus.COMPLETED
        assert updated.output == output
        assert updated.total_usage == usage

    def test_update_to_failed(self, db: Session, chain):
        usage = {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5}

        updated = update_llm_chain_status(
            db,
            chain_id=chain.id,
            status=ChainStatus.FAILED,
            error="Provider timeout",
            total_usage=usage,
        )

        assert updated.status == ChainStatus.FAILED
        assert updated.error == "Provider timeout"
        assert updated.total_usage == usage

    def test_raises_for_missing_chain(self, db: Session):
        with pytest.raises(ValueError, match="LLM chain not found"):
            update_llm_chain_status(db, chain_id=uuid4(), status=ChainStatus.RUNNING)


class TestUpdateLlmChainBlockCompleted:
    @pytest.fixture
    def chain(self, db: Session):
        project = get_project(db)
        job = JobCrud(session=db).create(
            job_type=JobType.LLM_CHAIN, trace_id="test-trace"
        )
        db.commit()
        chain = create_llm_chain(
            db,
            job_id=job.id,
            project_id=project.id,
            organization_id=project.organization_id,
            total_blocks=3,
            input="hello",
            configs=[],
        )
        return chain

    def test_appends_llm_call_id(self, db: Session, chain):
        call_id = uuid4()

        updated = update_llm_chain_block_completed(
            db, chain_id=chain.id, llm_call_id=call_id
        )

        assert str(call_id) in updated.block_sequences
        assert updated.number_of_blocks_processed == 1

    def test_appends_multiple_blocks(self, db: Session, chain):
        call_id_1 = uuid4()
        call_id_2 = uuid4()

        update_llm_chain_block_completed(db, chain_id=chain.id, llm_call_id=call_id_1)
        updated = update_llm_chain_block_completed(
            db, chain_id=chain.id, llm_call_id=call_id_2
        )

        assert len(updated.block_sequences) == 2
        assert updated.number_of_blocks_processed == 2

    def test_raises_for_missing_chain(self, db: Session):
        with pytest.raises(ValueError, match="LLM chain not found"):
            update_llm_chain_block_completed(db, chain_id=uuid4(), llm_call_id=uuid4())
