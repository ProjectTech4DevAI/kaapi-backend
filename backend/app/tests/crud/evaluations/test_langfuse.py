from typing import Any
from unittest.mock import MagicMock

import pytest

from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    update_traces_with_cosine_scores,
    upload_dataset_to_langfuse,
)


class TestCreateLangfuseDatasetRun:
    """Test creating Langfuse dataset runs."""

    def test_create_langfuse_dataset_run_success(self) -> None:
        """Test successfully creating a dataset run with traces."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()

        mock_item1 = MagicMock()
        mock_item1.id = "item_1"
        mock_item1.observe.return_value.__enter__.return_value = "trace_id_1"

        mock_item2 = MagicMock()
        mock_item2.id = "item_2"
        mock_item2.observe.return_value.__enter__.return_value = "trace_id_2"

        mock_dataset.items = [mock_item1, mock_item2]
        mock_langfuse.get_dataset.return_value = mock_dataset

        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "4",
                "ground_truth": "4",
                "response_id": "resp_123",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
            {
                "item_id": "item_2",
                "question": "What is the capital of France?",
                "generated_output": "Paris",
                "ground_truth": "Paris",
                "response_id": "resp_456",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                },
            },
        ]

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=results,
        )

        assert len(trace_id_mapping) == 2
        assert trace_id_mapping["item_1"] == "trace_id_1"
        assert trace_id_mapping["item_2"] == "trace_id_2"

        mock_langfuse.get_dataset.assert_called_once_with("test_dataset")
        mock_langfuse.flush.assert_called_once()
        assert mock_langfuse.trace.call_count == 2

    def test_create_langfuse_dataset_run_skips_missing_items(self) -> None:
        """Test that missing dataset items are skipped."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()

        mock_item1 = MagicMock()
        mock_item1.id = "item_1"
        mock_item1.observe.return_value.__enter__.return_value = "trace_id_1"

        mock_dataset.items = [mock_item1]
        mock_langfuse.get_dataset.return_value = mock_dataset

        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "4",
                "ground_truth": "4",
                "response_id": "resp_123",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
            {
                "item_id": "item_nonexistent",
                "question": "Invalid question",
                "generated_output": "Invalid",
                "ground_truth": "Invalid",
                "response_id": "resp_456",
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "total_tokens": 10,
                },
            },
        ]

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=results,
        )

        assert len(trace_id_mapping) == 1
        assert "item_1" in trace_id_mapping
        assert "item_nonexistent" not in trace_id_mapping

    def test_create_langfuse_dataset_run_handles_trace_error(self) -> None:
        """Test that trace creation errors are handled gracefully."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()

        mock_item1 = MagicMock()
        mock_item1.id = "item_1"
        mock_item1.observe.return_value.__enter__.return_value = "trace_id_1"

        mock_item2 = MagicMock()
        mock_item2.id = "item_2"
        mock_item2.observe.side_effect = Exception("Trace creation failed")

        mock_dataset.items = [mock_item1, mock_item2]
        mock_langfuse.get_dataset.return_value = mock_dataset

        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "4",
                "ground_truth": "4",
                "response_id": "resp_123",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
            {
                "item_id": "item_2",
                "question": "What is the capital?",
                "generated_output": "Paris",
                "ground_truth": "Paris",
                "response_id": "resp_456",
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "total_tokens": 10,
                },
            },
        ]

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=results,
        )

        assert len(trace_id_mapping) == 1
        assert "item_1" in trace_id_mapping
        assert "item_2" not in trace_id_mapping

    def test_create_langfuse_dataset_run_empty_results(self) -> None:
        """Test with empty results list."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = []
        mock_langfuse.get_dataset.return_value = mock_dataset

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=[],
        )

        assert len(trace_id_mapping) == 0
        mock_langfuse.flush.assert_called_once()

    def test_create_langfuse_dataset_run_with_cost_tracking(self) -> None:
        """Test that generation() is called with usage when model and usage are provided."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_generation = MagicMock()

        mock_item1 = MagicMock()
        mock_item1.id = "item_1"
        mock_item1.observe.return_value.__enter__.return_value = "trace_id_1"

        mock_item2 = MagicMock()
        mock_item2.id = "item_2"
        mock_item2.observe.return_value.__enter__.return_value = "trace_id_2"

        mock_dataset.items = [mock_item1, mock_item2]
        mock_langfuse.get_dataset.return_value = mock_dataset
        mock_langfuse.generation.return_value = mock_generation

        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "The answer is 4",
                "ground_truth": "4",
                "response_id": "resp_123",
                "usage": {
                    "input_tokens": 69,
                    "output_tokens": 258,
                    "total_tokens": 327,
                },
            },
            {
                "item_id": "item_2",
                "question": "What is the capital of France?",
                "generated_output": "Paris is the capital",
                "ground_truth": "Paris",
                "response_id": "resp_456",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "total_tokens": 150,
                },
            },
        ]

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=results,
            model="gpt-4o",
        )

        assert len(trace_id_mapping) == 2
        assert trace_id_mapping["item_1"] == "trace_id_1"
        assert trace_id_mapping["item_2"] == "trace_id_2"

        assert mock_langfuse.generation.call_count == 2

        first_call = mock_langfuse.generation.call_args_list[0]
        assert first_call.kwargs["name"] == "evaluation-response"
        assert first_call.kwargs["trace_id"] == "trace_id_1"
        assert first_call.kwargs["input"] == {"question": "What is 2+2?"}
        assert first_call.kwargs["metadata"]["ground_truth"] == "4"
        assert first_call.kwargs["metadata"]["response_id"] == "resp_123"

        assert mock_generation.end.call_count == 2

        first_end_call = mock_generation.end.call_args_list[0]
        assert first_end_call.kwargs["output"] == {"answer": "The answer is 4"}
        assert first_end_call.kwargs["model"] == "gpt-4o"
        assert first_end_call.kwargs["usage"] == {
            "input": 69,
            "output": 258,
            "total": 327,
            "unit": "TOKENS",
        }

        mock_langfuse.get_dataset.assert_called_once_with("test_dataset")
        mock_langfuse.flush.assert_called_once()
        assert mock_langfuse.trace.call_count == 2

    def test_create_langfuse_dataset_run_with_question_id(self) -> None:
        """Test that question_id is included in trace metadata."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_generation = MagicMock()

        mock_item1 = MagicMock()
        mock_item1.id = "item_1"
        mock_item1.observe.return_value.__enter__.return_value = "trace_id_1"

        mock_dataset.items = [mock_item1]
        mock_langfuse.get_dataset.return_value = mock_dataset
        mock_langfuse.generation.return_value = mock_generation

        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "4",
                "ground_truth": "4",
                "response_id": "resp_123",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                "question_id": 1,
            },
        ]

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=results,
            model="gpt-4o",
        )

        assert len(trace_id_mapping) == 1

        # Verify trace was called with question_id in metadata
        trace_call = mock_langfuse.trace.call_args
        assert trace_call.kwargs["metadata"]["question_id"] == 1

        # Verify generation was called with question_id in metadata
        generation_call = mock_langfuse.generation.call_args
        assert generation_call.kwargs["metadata"]["question_id"] == 1

    def test_create_langfuse_dataset_run_without_question_id(self) -> None:
        """Test that traces work without question_id (backwards compatibility)."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()

        mock_item1 = MagicMock()
        mock_item1.id = "item_1"
        mock_item1.observe.return_value.__enter__.return_value = "trace_id_1"

        mock_dataset.items = [mock_item1]
        mock_langfuse.get_dataset.return_value = mock_dataset

        # Results without question_id
        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "4",
                "ground_truth": "4",
                "response_id": "resp_123",
                "usage": None,
            },
        ]

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=results,
        )

        assert len(trace_id_mapping) == 1

        # Verify trace was called without question_id in metadata
        trace_call = mock_langfuse.trace.call_args
        assert "question_id" not in trace_call.kwargs["metadata"]


class TestUpdateTracesWithCosineScores:
    """Test updating Langfuse traces with cosine similarity scores."""

    def test_update_traces_with_cosine_scores_success(self) -> None:
        """Test successfully updating traces with scores."""
        mock_langfuse = MagicMock()

        per_item_scores = [
            {"trace_id": "trace_1", "cosine_similarity": 0.95},
            {"trace_id": "trace_2", "cosine_similarity": 0.87},
            {"trace_id": "trace_3", "cosine_similarity": 0.92},
        ]

        update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=per_item_scores
        )

        assert mock_langfuse.score.call_count == 3

        calls = mock_langfuse.score.call_args_list
        assert calls[0].kwargs["trace_id"] == "trace_1"
        assert calls[0].kwargs["name"] == "cosine_similarity"
        assert calls[0].kwargs["value"] == 0.95
        assert "cosine similarity" in calls[0].kwargs["comment"].lower()

        assert calls[1].kwargs["trace_id"] == "trace_2"
        assert calls[1].kwargs["value"] == 0.87

        mock_langfuse.flush.assert_called_once()

    def test_update_traces_with_cosine_scores_missing_trace_id(self) -> None:
        """Test that items without trace_id are skipped."""
        mock_langfuse = MagicMock()

        per_item_scores = [
            {"trace_id": "trace_1", "cosine_similarity": 0.95},
            {"cosine_similarity": 0.87},
            {"trace_id": "trace_3", "cosine_similarity": 0.92},
        ]

        update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=per_item_scores
        )

        assert mock_langfuse.score.call_count == 2

    def test_update_traces_with_cosine_scores_error_handling(self) -> None:
        """Test that score errors don't stop processing."""
        mock_langfuse = MagicMock()

        mock_langfuse.score.side_effect = [None, Exception("Score failed"), None]

        per_item_scores = [
            {"trace_id": "trace_1", "cosine_similarity": 0.95},
            {"trace_id": "trace_2", "cosine_similarity": 0.87},
            {"trace_id": "trace_3", "cosine_similarity": 0.92},
        ]

        update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=per_item_scores
        )

        assert mock_langfuse.score.call_count == 3
        mock_langfuse.flush.assert_called_once()

    def test_update_traces_with_cosine_scores_empty_list(self) -> None:
        """Test with empty scores list."""
        mock_langfuse = MagicMock()

        update_traces_with_cosine_scores(langfuse=mock_langfuse, per_item_scores=[])

        mock_langfuse.score.assert_not_called()
        mock_langfuse.flush.assert_called_once()


class TestUploadDatasetToLangfuse:
    """Test uploading datasets to Langfuse from pre-parsed items."""

    @pytest.fixture
    def valid_items(self) -> Any:
        """Valid parsed items."""
        return [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
        ]

    def test_upload_dataset_to_langfuse_success(self, valid_items):
        """Test successful upload with duplication."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        langfuse_id, total_items = upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=valid_items,
            dataset_name="test_dataset",
            duplication_factor=5,
        )

        assert langfuse_id == "dataset_123"
        assert total_items == 15

        mock_langfuse.create_dataset.assert_called_once_with(name="test_dataset")

        assert mock_langfuse.create_dataset_item.call_count == 15

        assert mock_langfuse.flush.call_count == 1

    def test_upload_dataset_to_langfuse_duplication_metadata(self, valid_items):
        """Test that duplication metadata is included."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=valid_items,
            dataset_name="test_dataset",
            duplication_factor=3,
        )

        calls = mock_langfuse.create_dataset_item.call_args_list

        duplicate_numbers = []
        for call_args in calls:
            metadata = call_args.kwargs.get("metadata", {})
            duplicate_numbers.append(metadata.get("duplicate_number"))
            assert metadata.get("duplication_factor") == 3

        assert duplicate_numbers.count(1) == 3
        assert duplicate_numbers.count(2) == 3
        assert duplicate_numbers.count(3) == 3

    def test_upload_dataset_to_langfuse_question_id_in_metadata(self, valid_items):
        """Test that question_id is included in metadata as integer."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=valid_items,
            dataset_name="test_dataset",
            duplication_factor=1,
        )

        calls = mock_langfuse.create_dataset_item.call_args_list
        assert len(calls) == 3

        question_ids = []
        for call_args in calls:
            metadata = call_args.kwargs.get("metadata", {})
            assert "question_id" in metadata
            assert metadata["question_id"] is not None
            # Verify it's an integer (1-based index)
            assert isinstance(metadata["question_id"], int)
            question_ids.append(metadata["question_id"])

        # Verify sequential IDs starting from 1
        assert sorted(question_ids) == [1, 2, 3]

    def test_upload_dataset_to_langfuse_same_question_id_for_duplicates(
        self, valid_items
    ):
        """Test that all duplicates of the same question share the same question_id."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=valid_items,
            dataset_name="test_dataset",
            duplication_factor=3,
        )

        calls = mock_langfuse.create_dataset_item.call_args_list
        assert len(calls) == 9  # 3 items * 3 duplicates

        # Group calls by original_question
        question_ids_by_question: dict[str, set[int]] = {}
        for call_args in calls:
            metadata = call_args.kwargs.get("metadata", {})
            original_question = metadata.get("original_question")
            question_id = metadata.get("question_id")

            # Verify question_id is an integer
            assert isinstance(question_id, int)

            if original_question not in question_ids_by_question:
                question_ids_by_question[original_question] = set()
            question_ids_by_question[original_question].add(question_id)

        # Verify each question has exactly one unique question_id across all duplicates
        for question, question_ids in question_ids_by_question.items():
            assert (
                len(question_ids) == 1
            ), f"Question '{question}' has multiple question_ids: {question_ids}"

        # Verify different questions have different question_ids (1, 2, 3)
        all_unique_ids: set[int] = set()
        for qid_set in question_ids_by_question.values():
            all_unique_ids.update(qid_set)
        assert all_unique_ids == {1, 2, 3}  # 3 unique questions = IDs 1, 2, 3

    def test_upload_dataset_to_langfuse_empty_items(self) -> None:
        """Test with empty items list."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        langfuse_id, total_items = upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=[],
            dataset_name="test_dataset",
            duplication_factor=1,
        )

        assert langfuse_id == "dataset_123"
        assert total_items == 0
        mock_langfuse.create_dataset_item.assert_not_called()
        assert mock_langfuse.flush.call_count == 1

    def test_upload_dataset_to_langfuse_single_duplication(self, valid_items):
        """Test upload with duplication factor of 1."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        langfuse_id, total_items = upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=valid_items,
            dataset_name="test_dataset",
            duplication_factor=1,
        )

        assert total_items == 3
        assert mock_langfuse.create_dataset_item.call_count == 3
        assert mock_langfuse.flush.call_count == 1

    def test_upload_dataset_to_langfuse_item_creation_error(self, valid_items):
        """Test that item creation errors are logged but don't stop processing."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        mock_langfuse.create_dataset_item.side_effect = [
            None,
            Exception("API error"),
            None,
        ]

        langfuse_id, total_items = upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=valid_items,
            dataset_name="test_dataset",
            duplication_factor=1,
        )

        assert total_items == 2
        assert mock_langfuse.create_dataset_item.call_count == 3
