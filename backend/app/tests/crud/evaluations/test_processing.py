from typing import Any
import json
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, select

from app.crud.evaluations.processing import (
    check_and_process_evaluation,
    parse_evaluation_output,
    process_completed_embedding_batch,
    process_completed_evaluation,
    poll_all_pending_evaluations,
)
from app.models import BatchJob, Organization, Project, EvaluationDataset, EvaluationRun
from app.tests.utils.test_data import create_test_evaluation_dataset
from app.crud.evaluations.core import create_evaluation_run
from app.core.util import now


class TestParseEvaluationOutput:
    """Test parsing evaluation batch output."""

    def test_parse_evaluation_output_basic(self) -> None:
        """Test basic parsing with valid data."""
        raw_results = [
            {
                "custom_id": "item1",
                "response": {
                    "body": {
                        "id": "resp_123",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "The answer is 4"}
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                },
            }
        ]

        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "What is 2+2?"},
                "expected_output": {"answer": "4"},
            }
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 1
        assert results[0]["item_id"] == "item1"
        assert results[0]["question"] == "What is 2+2?"
        assert results[0]["generated_output"] == "The answer is 4"
        assert results[0]["ground_truth"] == "4"
        assert results[0]["response_id"] == "resp_123"
        assert results[0]["usage"]["total_tokens"] == 15

    def test_parse_evaluation_output_simple_string(self) -> None:
        """Test parsing with simple string output."""
        raw_results = [
            {
                "custom_id": "item1",
                "response": {
                    "body": {
                        "id": "resp_123",
                        "output": "Simple text response",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                },
            }
        ]

        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test?"},
                "expected_output": {"answer": "Test"},
            }
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 1
        assert results[0]["generated_output"] == "Simple text response"

    def test_parse_evaluation_output_with_error(self) -> None:
        """Test parsing item with error."""
        raw_results = [
            {
                "custom_id": "item1",
                "error": {"message": "Rate limit exceeded"},
                "response": {"body": {}},
            }
        ]

        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test?"},
                "expected_output": {"answer": "Test"},
            }
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 1
        assert "ERROR: Rate limit exceeded" in results[0]["generated_output"]

    def test_parse_evaluation_output_missing_custom_id(self) -> None:
        """Test parsing skips items without custom_id."""
        raw_results = [
            {
                "response": {
                    "body": {
                        "output": "Test",
                        "usage": {"total_tokens": 10},
                    }
                }
            }
        ]

        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test?"},
                "expected_output": {"answer": "Test"},
            }
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 0

    def test_parse_evaluation_output_missing_dataset_item(self) -> None:
        """Test parsing skips items not in dataset."""
        raw_results = [
            {
                "custom_id": "item999",
                "response": {"body": {"output": "Test", "usage": {"total_tokens": 10}}},
            }
        ]

        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test?"},
                "expected_output": {"answer": "Test"},
            }
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 0

    def test_parse_evaluation_output_json_string(self) -> None:
        """Test parsing JSON string output."""
        raw_results = [
            {
                "custom_id": "item1",
                "response": {
                    "body": {
                        "output": json.dumps(
                            [
                                {
                                    "type": "message",
                                    "content": [
                                        {"type": "output_text", "text": "Parsed JSON"}
                                    ],
                                }
                            ]
                        ),
                        "usage": {"total_tokens": 10},
                    }
                },
            }
        ]

        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test?"},
                "expected_output": {"answer": "Test"},
            }
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 1
        assert results[0]["generated_output"] == "Parsed JSON"

    def test_parse_evaluation_output_multiple_items(self) -> None:
        """Test parsing multiple items."""
        raw_results = [
            {
                "custom_id": f"item{i}",
                "response": {
                    "body": {
                        "output": f"Output {i}",
                        "usage": {"total_tokens": 10},
                    }
                },
            }
            for i in range(3)
        ]

        dataset_items = [
            {
                "id": f"item{i}",
                "input": {"question": f"Q{i}"},
                "expected_output": {"answer": f"A{i}"},
            }
            for i in range(3)
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["item_id"] == f"item{i}"
            assert result["generated_output"] == f"Output {i}"
            assert result["ground_truth"] == f"A{i}"


class TestProcessCompletedEvaluation:
    """Test processing completed evaluation batch."""

    @pytest.fixture
    def test_dataset(self, db: Session) -> EvaluationDataset:
        """Create a test dataset for evaluation runs."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        return create_test_evaluation_dataset(
            db=db,
            organization_id=org.id,
            project_id=project.id,
            name="test_dataset_processing",
            description="Test dataset",
            original_items_count=3,
            duplication_factor=1,
        )

    @pytest.fixture
    def eval_run_with_batch(self, db: Session, test_dataset) -> EvaluationRun:
        """Create evaluation run with batch job."""
        # Create batch job
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_abc123",
            provider_status="completed",
            job_type="evaluation",
            total_items=2,
            status="submitted",
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        db.add(batch_job)
        db.commit()
        db.refresh(batch_job)

        eval_run = create_evaluation_run(
            session=db,
            run_name="test_run",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config={"model": "gpt-4o"},
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        return eval_run

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.fetch_dataset_items")
    @patch("app.crud.evaluations.processing.create_langfuse_dataset_run")
    @patch("app.crud.evaluations.processing.start_embedding_batch")
    @patch("app.crud.evaluations.processing.upload_batch_results_to_object_store")
    async def test_process_completed_evaluation_success(
        self,
        mock_upload,
        mock_start_embedding,
        mock_create_langfuse,
        mock_fetch_dataset,
        mock_download,
        db: Session,
        eval_run_with_batch,
    ):
        """Test successfully processing completed evaluation."""
        # Mock batch results
        mock_download.return_value = [
            {
                "custom_id": "item1",
                "response": {
                    "body": {
                        "id": "resp_123",
                        "output": "Answer 1",
                        "usage": {"total_tokens": 10},
                    }
                },
            }
        ]

        # Mock dataset items
        mock_fetch_dataset.return_value = [
            {
                "id": "item1",
                "input": {"question": "Q1"},
                "expected_output": {"answer": "A1"},
            }
        ]

        # Mock Langfuse
        mock_create_langfuse.return_value = {"item1": "trace_123"}

        # Mock embedding batch
        mock_start_embedding.return_value = eval_run_with_batch

        # Mock upload
        mock_upload.return_value = "s3://bucket/results.jsonl"

        mock_openai = MagicMock()
        mock_langfuse = MagicMock()

        result = await process_completed_evaluation(
            eval_run=eval_run_with_batch,
            session=db,
            openai_client=mock_openai,
            langfuse=mock_langfuse,
        )

        assert result is not None
        mock_download.assert_called_once()
        mock_fetch_dataset.assert_called_once()
        mock_create_langfuse.assert_called_once()
        mock_start_embedding.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.fetch_dataset_items")
    async def test_process_completed_evaluation_no_results(
        self,
        mock_fetch_dataset,
        mock_download,
        db: Session,
        eval_run_with_batch,
    ):
        """Test processing with no valid results."""
        mock_download.return_value = []
        mock_fetch_dataset.return_value = [
            {
                "id": "item1",
                "input": {"question": "Q1"},
                "expected_output": {"answer": "A1"},
            }
        ]

        mock_openai = MagicMock()
        mock_langfuse = MagicMock()

        result = await process_completed_evaluation(
            eval_run=eval_run_with_batch,
            session=db,
            openai_client=mock_openai,
            langfuse=mock_langfuse,
        )

        db.refresh(result)
        assert result.status == "failed"
        assert "No valid results" in result.error_message

    @pytest.mark.asyncio
    async def test_process_completed_evaluation_no_batch_job_id(
        self, db: Session, test_dataset
    ):
        """Test processing without batch_job_id."""
        eval_run = create_evaluation_run(
            session=db,
            run_name="test_run",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config={"model": "gpt-4o"},
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )

        mock_openai = MagicMock()
        mock_langfuse = MagicMock()

        result = await process_completed_evaluation(
            eval_run=eval_run,
            session=db,
            openai_client=mock_openai,
            langfuse=mock_langfuse,
        )

        db.refresh(result)
        assert result.status == "failed"
        assert "no batch_job_id" in result.error_message


class TestProcessCompletedEmbeddingBatch:
    """Test processing completed embedding batch."""

    @pytest.fixture
    def test_dataset(self, db: Session) -> EvaluationDataset:
        """Create a test dataset."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        return create_test_evaluation_dataset(
            db=db,
            organization_id=org.id,
            project_id=project.id,
            name="test_dataset_embedding",
            description="Test dataset",
            original_items_count=2,
            duplication_factor=1,
        )

    @pytest.fixture
    def eval_run_with_embedding_batch(self, db: Session, test_dataset) -> EvaluationRun:
        """Create evaluation run with embedding batch job."""
        # Create embedding batch job
        embedding_batch = BatchJob(
            provider="openai",
            provider_batch_id="batch_embed_123",
            provider_status="completed",
            job_type="embedding",
            total_items=4,
            status="submitted",
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        db.add(embedding_batch)
        db.commit()
        db.refresh(embedding_batch)

        # Create evaluation run
        eval_run = create_evaluation_run(
            session=db,
            run_name="test_run_embedding",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config={"model": "gpt-4o"},
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.embedding_batch_job_id = embedding_batch.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        return eval_run

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.parse_embedding_results")
    @patch("app.crud.evaluations.processing.calculate_average_similarity")
    @patch("app.crud.evaluations.processing.update_traces_with_cosine_scores")
    async def test_process_completed_embedding_batch_success(
        self,
        mock_update_traces,
        mock_calculate,
        mock_parse,
        mock_download,
        db: Session,
        eval_run_with_embedding_batch,
    ):
        """Test successfully processing completed embedding batch."""
        mock_download.return_value = []
        mock_parse.return_value = [
            {
                "item_id": "item1",
                "trace_id": "trace_123",
                "output_embedding": [1.0, 0.0],
                "ground_truth_embedding": [1.0, 0.0],
            }
        ]
        mock_calculate.return_value = {
            "cosine_similarity_avg": 0.95,
            "cosine_similarity_std": 0.02,
            "total_pairs": 1,
            "per_item_scores": [
                {"item_id": "item1", "trace_id": "trace_123", "cosine_similarity": 0.95}
            ],
        }

        mock_openai = MagicMock()
        mock_langfuse = MagicMock()

        result = await process_completed_embedding_batch(
            eval_run=eval_run_with_embedding_batch,
            session=db,
            openai_client=mock_openai,
            langfuse=mock_langfuse,
        )

        db.refresh(result)
        assert result.status == "completed"
        assert result.score is not None
        assert "cosine_similarity" in result.score
        assert result.score["cosine_similarity"]["avg"] == 0.95
        mock_update_traces.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.parse_embedding_results")
    async def test_process_completed_embedding_batch_no_results(
        self,
        mock_parse,
        mock_download,
        db: Session,
        eval_run_with_embedding_batch,
    ):
        """Test processing with no valid embedding results."""
        mock_download.return_value = []
        mock_parse.return_value = []

        mock_openai = MagicMock()
        mock_langfuse = MagicMock()

        result = await process_completed_embedding_batch(
            eval_run=eval_run_with_embedding_batch,
            session=db,
            openai_client=mock_openai,
            langfuse=mock_langfuse,
        )

        db.refresh(result)
        assert result.status == "completed"
        assert "failed" in result.error_message.lower()


class TestCheckAndProcessEvaluation:
    """Test check and process evaluation function."""

    @pytest.fixture
    def test_dataset(self, db: Session) -> EvaluationDataset:
        """Create a test dataset."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        return create_test_evaluation_dataset(
            db=db,
            organization_id=org.id,
            project_id=project.id,
            name="test_dataset_check",
            description="Test dataset",
            original_items_count=2,
            duplication_factor=1,
        )

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.get_batch_job")
    @patch("app.crud.evaluations.processing.poll_batch_status")
    @patch("app.crud.evaluations.processing.process_completed_evaluation")
    async def test_check_and_process_evaluation_completed(
        self,
        mock_process,
        mock_poll,
        mock_get_batch,
        db: Session,
        test_dataset,
    ):
        """Test checking evaluation with completed batch."""
        # Create batch job
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_abc",
            provider_status="completed",
            job_type="evaluation",
            total_items=2,
            status="submitted",
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        db.add(batch_job)
        db.commit()
        db.refresh(batch_job)

        # Create evaluation run
        eval_run = create_evaluation_run(
            session=db,
            run_name="test_run",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config={"model": "gpt-4o"},
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        mock_get_batch.return_value = batch_job
        mock_process.return_value = eval_run

        mock_openai = MagicMock()
        mock_langfuse = MagicMock()

        result = await check_and_process_evaluation(
            eval_run=eval_run,
            session=db,
            openai_client=mock_openai,
            langfuse=mock_langfuse,
        )

        assert result["action"] == "processed"
        assert result["run_id"] == eval_run.id
        mock_process.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.get_batch_job")
    @patch("app.crud.evaluations.processing.poll_batch_status")
    async def test_check_and_process_evaluation_failed(
        self,
        mock_poll,
        mock_get_batch,
        db: Session,
        test_dataset,
    ):
        """Test checking evaluation with failed batch."""
        # Create failed batch job
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_fail",
            provider_status="failed",
            job_type="evaluation",
            total_items=2,
            status="submitted",
            error_message="Provider error",
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        db.add(batch_job)
        db.commit()
        db.refresh(batch_job)

        # Create evaluation run
        eval_run = create_evaluation_run(
            session=db,
            run_name="test_run_fail",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config={"model": "gpt-4o"},
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        mock_get_batch.return_value = batch_job

        mock_openai = MagicMock()
        mock_langfuse = MagicMock()

        result = await check_and_process_evaluation(
            eval_run=eval_run,
            session=db,
            openai_client=mock_openai,
            langfuse=mock_langfuse,
        )

        assert result["action"] == "failed"
        assert result["current_status"] == "failed"
        db.refresh(eval_run)
        assert eval_run.status == "failed"


class TestPollAllPendingEvaluations:
    """Test polling all pending evaluations."""

    @pytest.fixture
    def test_dataset(self, db: Session) -> EvaluationDataset:
        """Create a test dataset."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        return create_test_evaluation_dataset(
            db=db,
            organization_id=org.id,
            project_id=project.id,
            name="test_dataset_poll",
            description="Test dataset",
            original_items_count=2,
            duplication_factor=1,
        )

    @pytest.mark.asyncio
    async def test_poll_all_pending_evaluations_no_pending(
        self, db: Session, test_dataset
    ):
        """Test polling with no pending evaluations."""
        result = await poll_all_pending_evaluations(
            session=db, org_id=test_dataset.organization_id
        )

        assert result["total"] == 0
        assert result["processed"] == 0
        assert result["failed"] == 0
        assert result["still_processing"] == 0

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.check_and_process_evaluation")
    @patch("app.crud.evaluations.processing.get_openai_client")
    @patch("app.crud.evaluations.processing.get_langfuse_client")
    async def test_poll_all_pending_evaluations_with_runs(
        self,
        mock_langfuse_client,
        mock_openai_client,
        mock_check,
        db: Session,
        test_dataset,
    ):
        """Test polling with pending evaluations."""
        # Create batch job
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_test",
            provider_status="in_progress",
            job_type="evaluation",
            total_items=2,
            status="submitted",
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        db.add(batch_job)
        db.commit()
        db.refresh(batch_job)

        # Create pending evaluation run
        eval_run = create_evaluation_run(
            session=db,
            run_name="test_pending_run",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config={"model": "gpt-4o"},
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        mock_openai_client.return_value = MagicMock()
        mock_langfuse_client.return_value = MagicMock()
        mock_check.return_value = {
            "run_id": eval_run.id,
            "run_name": eval_run.run_name,
            "action": "no_change",
        }

        result = await poll_all_pending_evaluations(
            session=db, org_id=test_dataset.organization_id
        )

        assert result["total"] == 1
        assert result["still_processing"] == 1
        mock_check.assert_called_once()
