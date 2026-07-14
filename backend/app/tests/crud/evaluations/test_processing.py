import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, select

from app.core.util import now
from app.crud.evaluations.core import create_evaluation_run
from app.crud.evaluations.processing import (
    _extract_batch_error_message,
    _extract_gemini_usage,
    _get_batch_provider,
    check_and_process_evaluation,
    parse_evaluation_output,
    poll_all_pending_evaluations,
    process_completed_embedding_batch,
    process_completed_evaluation,
)
from app.models import BatchJob, EvaluationDataset, EvaluationRun, Organization, Project
from app.models.batch_job import BatchJobType
from app.models.evaluation import RunModeEnum
from app.tests.utils.test_data import create_test_config, create_test_evaluation_dataset


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
                "metadata": {"question_id": 1},
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
        assert results[0]["question_id"] == 1

    def test_parse_evaluation_output_without_question_id(self) -> None:
        """Test parsing dataset items without question_id (backwards compatibility)."""
        raw_results = [
            {
                "custom_id": "item1",
                "response": {
                    "body": {
                        "id": "resp_123",
                        "output": "Answer text",
                        "usage": {"total_tokens": 10},
                    }
                },
            }
        ]

        dataset_items = [
            {
                "id": "item1",
                "input": {"question": "Test question?"},
                "expected_output": {"answer": "Test answer"},
                # No metadata / question_id
            }
        ]

        results = parse_evaluation_output(raw_results, dataset_items)

        assert len(results) == 1
        assert results[0]["question_id"] is None

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


class TestParseEvaluationOutputGoogle:
    """Test parsing Gemini batch output (provider_name=google-aistudio).

    Gemini lines are keyed by ``key`` (not ``custom_id``) and carry a nested
    ``response`` dict with ``candidates``/``usageMetadata`` rather than the
    OpenAI ``response.body`` shape.
    """

    def _dataset_items(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "item1",
                "input": {"question": "What is 2+2?"},
                "expected_output": {"answer": "4"},
                "metadata": {"question_id": 7},
            }
        ]

    def test_parse_gemini_success(self) -> None:
        """Text is pulled from candidates and usage from usageMetadata."""
        raw_results = [
            {
                "key": "item1",
                "response": {
                    "responseId": "resp_g1",
                    "candidates": [
                        {"content": {"parts": [{"text": "The answer is 4"}]}}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                        "thoughtsTokenCount": 2,
                    },
                },
            }
        ]

        results = parse_evaluation_output(
            raw_results, self._dataset_items(), provider_name="google-aistudio"
        )

        assert len(results) == 1
        result = results[0]
        assert result["item_id"] == "item1"
        assert result["generated_output"] == "The answer is 4"
        assert result["response_id"] == "resp_g1"
        assert result["question_id"] == 7
        assert result["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 2,
        }

    def test_parse_gemini_error(self) -> None:
        """An ``error`` on the line yields an ERROR: generated_output."""
        raw_results = [{"key": "item1", "error": {"message": "quota exceeded"}}]

        results = parse_evaluation_output(
            raw_results, self._dataset_items(), provider_name="google-aistudio"
        )

        assert len(results) == 1
        assert results[0]["generated_output"].startswith("ERROR:")
        assert "quota exceeded" in results[0]["generated_output"]

    def test_parse_gemini_native_provider(self) -> None:
        """The -native provider variant is treated as Google too."""
        raw_results = [
            {
                "key": "item1",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                },
            }
        ]

        results = parse_evaluation_output(
            raw_results,
            self._dataset_items(),
            provider_name="google-aistudio-native",
        )

        assert results[0]["generated_output"] == "ok"
        assert results[0]["usage"] is None


class TestExtractGeminiUsage:
    """Test ``_extract_gemini_usage`` mapping of Gemini usage to OpenAI shape."""

    def test_camel_case(self) -> None:
        usage = _extract_gemini_usage(
            {
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                    "thoughtsTokenCount": 3,
                }
            }
        )
        assert usage == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 3,
        }

    def test_snake_case(self) -> None:
        usage = _extract_gemini_usage(
            {
                "usage_metadata": {
                    "prompt_token_count": 1,
                    "candidates_token_count": 2,
                    "total_token_count": 3,
                    "thoughts_token_count": 4,
                }
            }
        )
        assert usage == {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "reasoning_tokens": 4,
        }

    def test_missing_metadata_returns_none(self) -> None:
        assert _extract_gemini_usage({}) is None
        assert _extract_gemini_usage({"usageMetadata": None}) is None

    def test_partial_metadata_defaults_to_zero(self) -> None:
        usage = _extract_gemini_usage({"usageMetadata": {"promptTokenCount": 9}})
        assert usage == {
            "input_tokens": 9,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }


class TestGetBatchProvider:
    """Test _get_batch_provider dispatch by provider name."""

    @patch("app.crud.evaluations.processing.OpenAIBatchProvider")
    @patch("app.crud.evaluations.processing.get_openai_client")
    def test_openai(self, mock_get_client, mock_provider_cls) -> None:
        provider = _get_batch_provider(
            session=MagicMock(),
            provider_name="openai",
            organization_id=1,
            project_id=2,
        )
        mock_get_client.assert_called_once()
        assert provider is mock_provider_cls.return_value

    @patch("app.crud.evaluations.processing.GeminiBatchProvider")
    @patch("app.crud.evaluations.processing.GeminiClient")
    def test_google(self, mock_gemini_client, mock_provider_cls) -> None:
        provider = _get_batch_provider(
            session=MagicMock(),
            provider_name="google-aistudio",
            organization_id=1,
            project_id=2,
        )
        mock_gemini_client.from_credentials.assert_called_once()
        assert provider is mock_provider_cls.return_value

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported provider"):
            _get_batch_provider(
                session=MagicMock(),
                provider_name="anthropic",
                organization_id=1,
                project_id=2,
            )


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
        # Create a config for the evaluation run
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        # Create batch job
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_abc123",
            provider_status="completed",
            job_type=BatchJobType.EVALUATION,
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
            config_id=config.id,
            config_version=1,
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
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                        },
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

        # Cost tracking: response cost should be aggregated and persisted.
        db.refresh(result)
        assert result.cost is not None
        assert "response" in result.cost
        response_cost = result.cost["response"]
        assert response_cost["model"] == "gpt-4o"
        assert response_cost["input_tokens"] == 100
        assert response_cost["output_tokens"] == 50
        assert response_cost["total_tokens"] == 150
        assert response_cost["cost_usd"] > 0
        assert result.cost["total_cost_usd"] == response_cost["cost_usd"]
        # Embedding cost is added later by process_completed_embedding_batch.
        assert "embedding" not in result.cost

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.persist_score_traces")
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.fetch_dataset_items")
    @patch("app.crud.evaluations.processing.create_langfuse_dataset_run")
    @patch("app.crud.evaluations.processing.start_embedding_batch")
    @patch("app.crud.evaluations.processing.upload_batch_results_to_object_store")
    async def test_process_completed_evaluation_persists_trace_skeleton(
        self,
        mock_upload,
        mock_start_embedding,
        mock_create_langfuse,
        mock_fetch_dataset,
        mock_download,
        mock_persist_traces,
        db: Session,
        eval_run_with_batch,
    ):
        """Q&A trace skeleton is persisted durably before embeddings complete."""
        mock_download.return_value = [
            {
                "custom_id": "item1",
                "response": {
                    "body": {
                        "id": "resp_123",
                        "output": "Answer 1",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                        },
                    }
                },
            }
        ]
        mock_fetch_dataset.return_value = [
            {
                "id": "item1",
                "input": {"question": "Q1"},
                "expected_output": {"answer": "A1"},
                "metadata": {"question_id": 7},
            }
        ]
        mock_create_langfuse.return_value = {"item1": "trace_123"}
        mock_start_embedding.return_value = eval_run_with_batch
        mock_upload.return_value = "s3://bucket/results.jsonl"
        mock_persist_traces.return_value = eval_run_with_batch

        await process_completed_evaluation(
            eval_run=eval_run_with_batch,
            session=db,
            openai_client=MagicMock(),
            langfuse=MagicMock(),
        )

        # The Q&A-only skeleton is persisted via persist_score_traces (trace
        # pointer only — it never writes the `score` column, so the run stays
        # score-less / "processing" until cosine is computed), keyed by the
        # Langfuse trace_id so the embedding stage can attach cosine later.
        mock_persist_traces.assert_called_once()
        traces = mock_persist_traces.call_args.kwargs["traces"]
        assert len(traces) == 1
        assert traces[0]["trace_id"] == "trace_123"
        assert traces[0]["question"] == "Q1"
        assert traces[0]["llm_answer"] == "Answer 1"
        assert traces[0]["ground_truth_answer"] == "A1"
        assert traces[0]["question_id"] == 7
        assert traces[0]["scores"] == []

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
        # Create a config for the evaluation run
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        eval_run = create_evaluation_run(
            session=db,
            run_name="test_run",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config_id=config.id,
            config_version=1,
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
        # Create a config for the evaluation run
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        # Create embedding batch job
        embedding_batch = BatchJob(
            provider="openai",
            provider_batch_id="batch_embed_123",
            provider_status="completed",
            job_type=BatchJobType.EMBEDDING,
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
            config_id=config.id,
            config_version=1,
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
        # Pre-populate eval_run.cost with a response entry to verify that the
        # embedding stage merges (not overwrites) existing cost data.
        eval_run_with_embedding_batch.cost = {
            "response": {
                "model": "gpt-4o",
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost_usd": 0.000375,
            },
            "total_cost_usd": 0.000375,
        }
        db.add(eval_run_with_embedding_batch)
        db.commit()
        db.refresh(eval_run_with_embedding_batch)

        # Raw results carry the usage payload that _build_embedding_cost_entry reads.
        mock_download.return_value = [
            {
                "custom_id": "trace_123",
                "response": {
                    "body": {"usage": {"prompt_tokens": 200, "total_tokens": 200}}
                },
            }
        ]
        mock_parse.return_value = (
            [
                {
                    "item_id": "item1",
                    "trace_id": "trace_123",
                    "output_embedding": [1.0, 0.0],
                    "ground_truth_embedding": [1.0, 0.0],
                }
            ],
            [],
        )
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
        assert "summary_scores" in result.score
        summary_scores = result.score["summary_scores"]
        cosine_score = next(
            (s for s in summary_scores if s["name"] == "Cosine Similarity"), None
        )
        assert cosine_score is not None
        assert cosine_score["avg"] == 0.95
        # Denominator is surfaced for the UI alongside the scored pair count.
        assert cosine_score["total_items"] == result.total_items

        # Durable per-item scores are persisted as the resync source of truth,
        # so a lost/failed Langfuse write can always be backfilled later.
        assert result.per_item_scores == {"trace_123": pytest.approx(0.95)}

        # Cost tracking: embedding entry is added, response entry is preserved,
        # and total_cost_usd is the sum of both.
        assert result.cost is not None
        assert "response" in result.cost
        assert "embedding" in result.cost
        assert result.cost["response"]["cost_usd"] == 0.000375
        embedding_cost = result.cost["embedding"]
        assert embedding_cost["model"] == "text-embedding-3-large"
        assert embedding_cost["input_tokens"] == 200
        assert embedding_cost["output_tokens"] == 0
        assert embedding_cost["total_tokens"] == 200
        assert embedding_cost["cost_usd"] > 0
        assert result.cost["total_cost_usd"] == pytest.approx(
            0.000375 + embedding_cost["cost_usd"]
        )

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.save_score")
    @patch("app.crud.evaluations.processing._load_score_traces")
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.parse_embedding_results")
    @patch("app.crud.evaluations.processing.calculate_average_similarity")
    @patch("app.crud.evaluations.processing.update_traces_with_cosine_scores")
    async def test_process_completed_embedding_batch_persists_durable_traces(
        self,
        mock_update_traces,
        mock_calculate,
        mock_parse,
        mock_download,
        mock_load_traces,
        mock_save_score,
        db: Session,
        eval_run_with_embedding_batch,
    ):
        """When a Q&A skeleton exists, cosine is attached and full traces saved.

        Cosine display then comes straight from score_trace_url (no Langfuse).
        """
        mock_download.return_value = [
            {
                "custom_id": "trace_123",
                "response": {
                    "body": {"usage": {"prompt_tokens": 200, "total_tokens": 200}}
                },
            }
        ]
        mock_parse.return_value = (
            [
                {
                    "item_id": "item1",
                    "trace_id": "trace_123",
                    "output_embedding": [1.0, 0.0],
                    "ground_truth_embedding": [1.0, 0.0],
                }
            ],
            [],
        )
        mock_calculate.return_value = {
            "cosine_similarity_avg": 0.95,
            "cosine_similarity_std": 0.0,
            "total_pairs": 1,
            "per_item_scores": [
                {"item_id": "item1", "trace_id": "trace_123", "cosine_similarity": 0.95}
            ],
        }
        # A durable Q&A skeleton (no scores yet) persisted at the response stage.
        mock_load_traces.return_value = [
            {
                "trace_id": "trace_123",
                "question": "Q1",
                "llm_answer": "A1",
                "ground_truth_answer": "GT1",
                "question_id": 1,
                "scores": [],
            }
        ]
        mock_save_score.return_value = eval_run_with_embedding_batch

        result = await process_completed_embedding_batch(
            eval_run=eval_run_with_embedding_batch,
            session=db,
            openai_client=MagicMock(),
            langfuse=MagicMock(),
        )

        # The full trace unit is persisted with cosine attached to the skeleton.
        mock_save_score.assert_called_once()
        saved_score = mock_save_score.call_args.kwargs["score"]
        traces = saved_score["traces"]
        assert len(traces) == 1
        cosine = next(
            s for s in traces[0]["scores"] if s["name"] == "Cosine Similarity"
        )
        assert cosine["value"] == 0.95
        assert not cosine.get("unscoreable")
        assert saved_score["summary_scores"][0]["name"] == "Cosine Similarity"

        # The in-memory score carries the traces for the response.
        assert result.score["traces"][0]["trace_id"] == "trace_123"

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.save_score")
    @patch("app.crud.evaluations.processing._load_score_traces")
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.parse_embedding_results")
    @patch("app.crud.evaluations.processing.calculate_average_similarity")
    @patch("app.crud.evaluations.processing.update_traces_with_cosine_scores")
    async def test_process_completed_embedding_batch_unscoreable_placeholder(
        self,
        mock_update_traces,
        mock_calculate,
        mock_parse,
        mock_download,
        mock_load_traces,
        mock_save_score,
        db: Session,
        eval_run_with_embedding_batch,
    ):
        """Unscoreable items get a flagged 0-score placeholder on the durable trace."""
        # Flag trace_empty as unscoreable (empty model output).
        eval_run_with_embedding_batch.unscoreable = {"trace_empty": "empty_output"}
        db.add(eval_run_with_embedding_batch)
        db.commit()
        db.refresh(eval_run_with_embedding_batch)

        mock_download.return_value = []
        mock_parse.return_value = (
            [
                {
                    "item_id": "item1",
                    "trace_id": "trace_123",
                    "output_embedding": [1.0, 0.0],
                    "ground_truth_embedding": [1.0, 0.0],
                }
            ],
            [],
        )
        mock_calculate.return_value = {
            "cosine_similarity_avg": 0.95,
            "cosine_similarity_std": 0.0,
            "total_pairs": 1,
            "per_item_scores": [
                {"item_id": "item1", "trace_id": "trace_123", "cosine_similarity": 0.95}
            ],
        }
        mock_load_traces.return_value = [
            {
                "trace_id": "trace_123",
                "question": "Q1",
                "llm_answer": "A1",
                "ground_truth_answer": "GT1",
                "question_id": 1,
                "scores": [],
            },
            {
                "trace_id": "trace_empty",
                "question": "Q2",
                "llm_answer": "",
                "ground_truth_answer": "GT2",
                "question_id": 2,
                "scores": [],
            },
        ]
        mock_save_score.return_value = eval_run_with_embedding_batch

        await process_completed_embedding_batch(
            eval_run=eval_run_with_embedding_batch,
            session=db,
            openai_client=MagicMock(),
            langfuse=MagicMock(),
        )

        saved_score = mock_save_score.call_args.kwargs["score"]
        by_id = {t["trace_id"]: t for t in saved_score["traces"]}
        placeholder = by_id["trace_empty"]["scores"][0]
        assert placeholder["value"] == 0
        assert placeholder["unscoreable"] is True
        assert "empty_output" in placeholder["comment"]

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.save_score")
    @patch("app.crud.evaluations.processing._load_score_traces")
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.parse_embedding_results")
    @patch("app.crud.evaluations.processing.calculate_average_similarity")
    @patch("app.crud.evaluations.processing.update_traces_with_cosine_scores")
    async def test_process_completed_embedding_batch_flags_embedding_failed(
        self,
        mock_update_traces,
        mock_calculate,
        mock_parse,
        mock_download,
        mock_load_traces,
        mock_save_score,
        db: Session,
        eval_run_with_embedding_batch,
    ):
        """A failed embedding is flagged embedding_failed across all outputs."""
        mock_download.return_value = []
        # trace_ok scored normally; trace_fail dropped by parse_embedding_results.
        mock_parse.return_value = (
            [
                {
                    "item_id": "item1",
                    "trace_id": "trace_ok",
                    "output_embedding": [1.0, 0.0],
                    "ground_truth_embedding": [1.0, 0.0],
                }
            ],
            ["trace_fail"],
        )
        mock_calculate.return_value = {
            "cosine_similarity_avg": 0.95,
            "cosine_similarity_std": 0.0,
            "total_pairs": 1,
            "per_item_scores": [
                {"item_id": "item1", "trace_id": "trace_ok", "cosine_similarity": 0.95}
            ],
        }
        mock_update_traces.return_value = []
        mock_load_traces.return_value = [
            {
                "trace_id": "trace_ok",
                "question": "Q1",
                "llm_answer": "A1",
                "ground_truth_answer": "GT1",
                "question_id": 1,
                "scores": [],
            },
            {
                "trace_id": "trace_fail",
                "question": "Q2",
                "llm_answer": "A2",
                "ground_truth_answer": "GT2",
                "question_id": 2,
                "scores": [],
            },
        ]
        mock_save_score.return_value = eval_run_with_embedding_batch

        result = await process_completed_embedding_batch(
            eval_run=eval_run_with_embedding_batch,
            session=db,
            openai_client=MagicMock(),
            langfuse=MagicMock(),
        )

        # Flagged embedding_failed in the durable unscoreable map.
        assert result.unscoreable["trace_fail"] == "embedding_failed"

        # Counted in the cosine summary's unscoreable breakdown.
        cosine_summary = result.score["summary_scores"][0]
        assert cosine_summary["unscoreable"] == {"embedding_failed": 1}

        # Written to Langfuse as a 0-placeholder with the reason.
        write_items = mock_update_traces.call_args.kwargs["per_item_scores"]
        assert {
            "trace_id": "trace_fail",
            "unscoreable": True,
            "reason": "embedding_failed",
        } in write_items

        # Rendered as a flagged 0-score on the durable trace.
        saved_traces = {
            t["trace_id"]: t
            for t in mock_save_score.call_args.kwargs["score"]["traces"]
        }
        placeholder = saved_traces["trace_fail"]["scores"][0]
        assert placeholder["value"] == 0
        assert placeholder["unscoreable"] is True
        assert placeholder["comment"] == "Cannot compute: embedding_failed"

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.save_score")
    @patch("app.crud.evaluations.processing._load_score_traces")
    @patch("app.crud.evaluations.processing.download_batch_results")
    @patch("app.crud.evaluations.processing.parse_embedding_results")
    @patch("app.crud.evaluations.processing.calculate_average_similarity")
    @patch("app.crud.evaluations.processing.update_traces_with_cosine_scores")
    async def test_process_completed_embedding_batch_failed_does_not_override_empty(
        self,
        mock_update_traces,
        mock_calculate,
        mock_parse,
        mock_download,
        mock_load_traces,
        mock_save_score,
        db: Session,
        eval_run_with_embedding_batch,
    ):
        """A trace already flagged empty_* keeps its reason (no embedding_failed)."""
        eval_run_with_embedding_batch.unscoreable = {"trace_x": "empty_output"}
        db.add(eval_run_with_embedding_batch)
        db.commit()
        db.refresh(eval_run_with_embedding_batch)

        mock_download.return_value = []
        mock_parse.return_value = (
            [
                {
                    "item_id": "item1",
                    "trace_id": "trace_ok",
                    "output_embedding": [1.0, 0.0],
                    "ground_truth_embedding": [1.0, 0.0],
                }
            ],
            ["trace_x"],
        )
        mock_calculate.return_value = {
            "cosine_similarity_avg": 0.95,
            "cosine_similarity_std": 0.0,
            "total_pairs": 1,
            "per_item_scores": [
                {"item_id": "item1", "trace_id": "trace_ok", "cosine_similarity": 0.95}
            ],
        }
        mock_update_traces.return_value = []
        mock_load_traces.return_value = None
        mock_save_score.return_value = eval_run_with_embedding_batch

        result = await process_completed_embedding_batch(
            eval_run=eval_run_with_embedding_batch,
            session=db,
            openai_client=MagicMock(),
            langfuse=MagicMock(),
        )

        # setdefault preserves the original empty_output reason.
        assert result.unscoreable["trace_x"] == "empty_output"

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
        mock_parse.return_value = ([], [])

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
        # Create a config for the evaluation run
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        # Create batch job with output file (successful completion)
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_abc",
            provider_status="completed",
            provider_output_file_id="output-file-123",
            job_type=BatchJobType.EVALUATION,
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
            config_id=config.id,
            config_version=1,
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        mock_get_batch.return_value = batch_job
        mock_poll.return_value = {
            "provider_status": "completed",
            "provider_output_file_id": "output-file-123",
            "error_file_id": None,
            "request_counts": {"total": 2, "completed": 2, "failed": 0},
        }
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
        # Create a config for the evaluation run
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        # Create failed batch job
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_fail",
            provider_status="failed",
            job_type=BatchJobType.EVALUATION,
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
            config_id=config.id,
            config_version=1,
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        mock_get_batch.return_value = batch_job
        mock_poll.return_value = {
            "provider_status": "failed",
            "provider_output_file_id": None,
            "error_file_id": None,
            "error_message": "Provider error",
            "request_counts": {"total": 2, "completed": 0, "failed": 2},
        }

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

    @pytest.fixture
    def all_requests_failed_setup(
        self, db: Session, test_dataset
    ) -> tuple[BatchJob, EvaluationRun]:
        """Create a BatchJob (completed, no output file) and a processing EvaluationRun for the all-requests-failed scenario."""
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_all_fail",
            provider_status="completed",
            job_type=BatchJobType.EVALUATION,
            total_items=9,
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
            run_name="test_run_all_fail",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config_id=config.id,
            config_version=1,
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
        )
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        return batch_job, eval_run

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.get_batch_job")
    @patch("app.crud.evaluations.processing.poll_batch_status")
    @patch("app.crud.evaluations.processing.OpenAIBatchProvider")
    async def test_check_and_process_evaluation_completed_all_requests_failed(
        self,
        mock_provider_cls,
        mock_poll,
        mock_get_batch,
        db: Session,
        all_requests_failed_setup: tuple[BatchJob, EvaluationRun],
    ):
        """Test batch completed but all requests failed — both batch_job and eval_run get error_message."""
        batch_job, eval_run = all_requests_failed_setup

        mock_get_batch.return_value = batch_job
        mock_poll.return_value = {
            "provider_status": "completed",
            "provider_output_file_id": None,
            "error_file_id": "error-file-abc",
            "request_counts": {"total": 9, "completed": 0, "failed": 9},
        }

        # Mock the provider instance returned by OpenAIBatchProvider(client=...)
        # to return realistic error file content
        error_lines = "\n".join(
            [
                json.dumps(
                    {
                        "id": f"batch_req_{i}",
                        "custom_id": f"id-{i}",
                        "response": {
                            "status_code": 400,
                            "body": {
                                "error": {
                                    "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                                }
                            },
                        },
                        "error": None,
                    }
                )
                for i in range(9)
            ]
        )
        mock_provider_instance = mock_provider_cls.return_value
        mock_provider_instance.download_file.return_value = error_lines

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
        assert "temperature" in result["error"]
        assert "(9/9 requests)" in result["error"]

        # Verify eval_run updated with error
        db.refresh(eval_run)
        assert eval_run.status == "failed"
        assert "temperature" in eval_run.error_message

        # Verify batch_job updated with error
        db.refresh(batch_job)
        assert "temperature" in batch_job.error_message
        assert "(9/9 requests)" in batch_job.error_message


class TestExtractBatchErrorMessage:
    """Test extracting error messages from OpenAI error files."""

    def test_single_unique_error(self) -> None:
        """Test error file where all requests have the same error."""
        error_lines = []
        for i in range(5):
            error_lines.append(
                json.dumps(
                    {
                        "id": f"batch_req_{i}",
                        "custom_id": f"id-{i}",
                        "response": {
                            "status_code": 400,
                            "body": {
                                "error": {
                                    "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                                    "type": "invalid_request_error",
                                }
                            },
                        },
                        "error": None,
                    }
                )
            )
        error_content = "\n".join(error_lines)

        mock_provider = MagicMock()
        mock_provider.download_file.return_value = error_content

        mock_session = MagicMock()
        mock_batch_job = MagicMock()
        mock_batch_job.id = 1

        result = _extract_batch_error_message(
            provider=mock_provider,
            error_file_id="error-file-123",
            batch_job=mock_batch_job,
            session=mock_session,
        )

        assert "Unsupported parameter" in result
        assert "(5/5 requests)" in result
        mock_provider.download_file.assert_called_once_with("error-file-123")

    def test_multiple_unique_errors_picks_most_common(self) -> None:
        """Test error file with mixed errors; picks the most frequent one."""
        error_lines = []
        # 3 requests with temperature error
        for i in range(3):
            error_lines.append(
                json.dumps(
                    {
                        "id": f"batch_req_{i}",
                        "custom_id": f"id-{i}",
                        "response": {
                            "status_code": 400,
                            "body": {
                                "error": {
                                    "message": "Unsupported parameter: 'temperature'",
                                }
                            },
                        },
                        "error": None,
                    }
                )
            )
        # 1 request with rate limit error
        error_lines.append(
            json.dumps(
                {
                    "id": "batch_req_3",
                    "custom_id": "id-3",
                    "response": {
                        "status_code": 429,
                        "body": {
                            "error": {
                                "message": "Rate limit exceeded",
                            }
                        },
                    },
                    "error": None,
                }
            )
        )
        error_content = "\n".join(error_lines)

        mock_provider = MagicMock()
        mock_provider.download_file.return_value = error_content

        mock_session = MagicMock()
        mock_batch_job = MagicMock()
        mock_batch_job.id = 1

        result = _extract_batch_error_message(
            provider=mock_provider,
            error_file_id="error-file-123",
            batch_job=mock_batch_job,
            session=mock_session,
        )

        assert "Unsupported parameter: 'temperature'" in result
        assert "(3/4 requests)" in result


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
        result = await poll_all_pending_evaluations(session=db)

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
        # Create a config for the evaluation run
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        # Create batch job
        batch_job = BatchJob(
            provider="openai",
            provider_batch_id="batch_test",
            provider_status="in_progress",
            job_type=BatchJobType.EVALUATION,
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
            config_id=config.id,
            config_version=1,
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

        result = await poll_all_pending_evaluations(session=db)

        assert result["total"] == 1
        assert result["still_processing"] == 1
        mock_check.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.crud.evaluations.processing.check_and_process_evaluation")
    async def test_poll_all_pending_evaluations_excludes_fast_runs(
        self,
        mock_check,
        db: Session,
        test_dataset,
    ):
        """Fast-mode runs are handled synchronously and have no provider batch
        job, so the batch poller must skip them. Otherwise it picks up an
        in-flight fast run (status='processing', no batch_job_id yet) and wrongly
        marks it 'Checking failed: ... has no batch_job_id'."""
        config = create_test_config(
            db, project_id=test_dataset.project_id, use_kaapi_schema=True
        )

        eval_run = create_evaluation_run(
            session=db,
            run_name="test_fast_run",
            dataset_name=test_dataset.name,
            dataset_id=test_dataset.id,
            config_id=config.id,
            config_version=1,
            organization_id=test_dataset.organization_id,
            project_id=test_dataset.project_id,
            run_mode=RunModeEnum.FAST,
        )
        eval_run.status = "processing"
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)

        result = await poll_all_pending_evaluations(session=db)

        assert result["total"] == 0
        mock_check.assert_not_called()
        db.refresh(eval_run)
        assert eval_run.status == "processing"
        assert eval_run.error_message is None
