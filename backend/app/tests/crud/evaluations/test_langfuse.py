from typing import Any
from unittest.mock import MagicMock

import pytest

from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    fetch_trace_scores_from_langfuse,
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


class TestFetchTraceScoresFromLangfuse:
    """Test fetching trace scores from Langfuse."""

    def test_fetch_trace_scores_success_with_question_id(self) -> None:
        """Test successfully fetching traces with question_id in metadata."""
        mock_langfuse = MagicMock()

        # Mock dataset run
        mock_run_item1 = MagicMock()
        mock_run_item1.trace_id = "trace_1"
        mock_run_item2 = MagicMock()
        mock_run_item2.trace_id = "trace_2"

        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [mock_run_item1, mock_run_item2]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        # Mock trace 1 with question_id
        mock_trace1 = MagicMock()
        mock_trace1.input = {"question": "What is 2+2?"}
        mock_trace1.output = {"answer": "4"}
        mock_trace1.metadata = {"ground_truth": "4", "question_id": 1}
        mock_score1 = MagicMock()
        mock_score1.name = "cosine_similarity"
        mock_score1.value = 0.95
        mock_score1.comment = "High similarity"
        mock_score1.data_type = "NUMERIC"
        mock_trace1.scores = [mock_score1]

        # Mock trace 2 with question_id
        mock_trace2 = MagicMock()
        mock_trace2.input = {"question": "What is the capital of France?"}
        mock_trace2.output = {"answer": "Paris"}
        mock_trace2.metadata = {"ground_truth": "Paris", "question_id": 2}
        mock_score2 = MagicMock()
        mock_score2.name = "cosine_similarity"
        mock_score2.value = 0.87
        mock_score2.comment = None
        mock_score2.data_type = "NUMERIC"
        mock_trace2.scores = [mock_score2]

        mock_langfuse.api.trace.get.side_effect = [mock_trace1, mock_trace2]

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify traces
        assert len(result["traces"]) == 2

        # Check trace 1
        trace1 = result["traces"][0]
        assert trace1["trace_id"] == "trace_1"
        assert trace1["question"] == "What is 2+2?"
        assert trace1["llm_answer"] == "4"
        assert trace1["ground_truth_answer"] == "4"
        assert trace1["question_id"] == 1
        assert len(trace1["scores"]) == 1
        assert trace1["scores"][0]["name"] == "cosine_similarity"
        assert trace1["scores"][0]["value"] == 0.95
        assert trace1["scores"][0]["comment"] == "High similarity"

        # Check trace 2
        trace2 = result["traces"][1]
        assert trace2["trace_id"] == "trace_2"
        assert trace2["question_id"] == 2

        # Verify summary scores
        assert len(result["summary_scores"]) == 1
        summary = result["summary_scores"][0]
        assert summary["name"] == "cosine_similarity"
        assert summary["avg"] == 0.91  # (0.95 + 0.87) / 2 = 0.91
        assert summary["total_pairs"] == 2
        assert summary["data_type"] == "NUMERIC"

    def test_fetch_trace_scores_without_question_id(self) -> None:
        """Test fetching traces without question_id (backwards compatibility)."""
        mock_langfuse = MagicMock()

        # Mock dataset run
        mock_run_item = MagicMock()
        mock_run_item.trace_id = "trace_1"
        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [mock_run_item]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        # Mock trace without question_id in metadata
        mock_trace = MagicMock()
        mock_trace.input = {"question": "What is 2+2?"}
        mock_trace.output = {"answer": "4"}
        mock_trace.metadata = {"ground_truth": "4"}  # No question_id
        mock_trace.scores = []

        mock_langfuse.api.trace.get.return_value = mock_trace

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify trace has empty string for question_id
        assert len(result["traces"]) == 1
        trace = result["traces"][0]
        assert trace["question_id"] == ""
        assert trace["trace_id"] == "trace_1"
        assert trace["question"] == "What is 2+2?"

    def test_fetch_trace_scores_with_categorical_scores(self) -> None:
        """Test fetching traces with categorical scores."""
        mock_langfuse = MagicMock()

        # Mock dataset run
        mock_run_item1 = MagicMock()
        mock_run_item1.trace_id = "trace_1"
        mock_run_item2 = MagicMock()
        mock_run_item2.trace_id = "trace_2"
        mock_run_item3 = MagicMock()
        mock_run_item3.trace_id = "trace_3"

        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [
            mock_run_item1,
            mock_run_item2,
            mock_run_item3,
        ]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        # Mock traces with categorical scores
        mock_trace1 = MagicMock()
        mock_trace1.input = {"question": "Q1"}
        mock_trace1.output = {"answer": "A1"}
        mock_trace1.metadata = {"ground_truth": "GT1", "question_id": 1}
        mock_score1 = MagicMock()
        mock_score1.name = "accuracy"
        mock_score1.value = "CORRECT"
        mock_score1.comment = None
        mock_score1.data_type = "CATEGORICAL"
        mock_trace1.scores = [mock_score1]

        mock_trace2 = MagicMock()
        mock_trace2.input = {"question": "Q2"}
        mock_trace2.output = {"answer": "A2"}
        mock_trace2.metadata = {"ground_truth": "GT2", "question_id": 2}
        mock_score2 = MagicMock()
        mock_score2.name = "accuracy"
        mock_score2.value = "CORRECT"
        mock_score2.comment = None
        mock_score2.data_type = "CATEGORICAL"
        mock_trace2.scores = [mock_score2]

        mock_trace3 = MagicMock()
        mock_trace3.input = {"question": "Q3"}
        mock_trace3.output = {"answer": "A3"}
        mock_trace3.metadata = {"ground_truth": "GT3", "question_id": 3}
        mock_score3 = MagicMock()
        mock_score3.name = "accuracy"
        mock_score3.value = "INCORRECT"
        mock_score3.comment = None
        mock_score3.data_type = "CATEGORICAL"
        mock_trace3.scores = [mock_score3]

        mock_langfuse.api.trace.get.side_effect = [
            mock_trace1,
            mock_trace2,
            mock_trace3,
        ]

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify summary scores for categorical data
        assert len(result["summary_scores"]) == 1
        summary = result["summary_scores"][0]
        assert summary["name"] == "accuracy"
        assert summary["data_type"] == "CATEGORICAL"
        assert summary["distribution"] == {"CORRECT": 2, "INCORRECT": 1}
        assert summary["total_pairs"] == 3

    def test_fetch_trace_scores_filters_incomplete_scores(self) -> None:
        """Test that scores present in only some traces are filtered out."""
        mock_langfuse = MagicMock()

        # Mock dataset run
        mock_run_item1 = MagicMock()
        mock_run_item1.trace_id = "trace_1"
        mock_run_item2 = MagicMock()
        mock_run_item2.trace_id = "trace_2"

        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [mock_run_item1, mock_run_item2]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        # Mock trace 1 with two scores
        mock_trace1 = MagicMock()
        mock_trace1.input = {"question": "Q1"}
        mock_trace1.output = {"answer": "A1"}
        mock_trace1.metadata = {"ground_truth": "GT1", "question_id": 1}
        mock_score1a = MagicMock()
        mock_score1a.name = "complete_score"
        mock_score1a.value = 0.9
        mock_score1a.comment = None
        mock_score1a.data_type = "NUMERIC"
        mock_score1b = MagicMock()
        mock_score1b.name = "incomplete_score"
        mock_score1b.value = 0.8
        mock_score1b.comment = None
        mock_score1b.data_type = "NUMERIC"
        mock_trace1.scores = [mock_score1a, mock_score1b]

        # Mock trace 2 with only one score (incomplete_score is missing)
        mock_trace2 = MagicMock()
        mock_trace2.input = {"question": "Q2"}
        mock_trace2.output = {"answer": "A2"}
        mock_trace2.metadata = {"ground_truth": "GT2", "question_id": 2}
        mock_score2 = MagicMock()
        mock_score2.name = "complete_score"
        mock_score2.value = 0.7
        mock_score2.comment = None
        mock_score2.data_type = "NUMERIC"
        mock_trace2.scores = [mock_score2]

        mock_langfuse.api.trace.get.side_effect = [mock_trace1, mock_trace2]

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify only complete_score is in results
        assert len(result["summary_scores"]) == 1
        assert result["summary_scores"][0]["name"] == "complete_score"

        # Verify traces only have complete_score
        for trace in result["traces"]:
            assert len(trace["scores"]) == 1
            assert trace["scores"][0]["name"] == "complete_score"

    def test_fetch_trace_scores_handles_string_input_output(self) -> None:
        """Test fetching traces with string (non-dict) input/output."""
        mock_langfuse = MagicMock()

        # Mock dataset run
        mock_run_item = MagicMock()
        mock_run_item.trace_id = "trace_1"
        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [mock_run_item]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        # Mock trace with string input/output
        mock_trace = MagicMock()
        mock_trace.input = "What is 2+2?"  # String instead of dict
        mock_trace.output = "The answer is 4"  # String instead of dict
        mock_trace.metadata = {"ground_truth": "4", "question_id": 1}
        mock_trace.scores = []

        mock_langfuse.api.trace.get.return_value = mock_trace

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify string values are handled
        assert len(result["traces"]) == 1
        trace = result["traces"][0]
        assert trace["question"] == "What is 2+2?"
        assert trace["llm_answer"] == "The answer is 4"

    def test_fetch_trace_scores_run_not_found(self) -> None:
        """Test error handling when run is not found."""
        mock_langfuse = MagicMock()
        mock_langfuse.api.datasets.get_run.side_effect = Exception("Run not found")

        with pytest.raises(ValueError, match="Run 'test_run' not found"):
            fetch_trace_scores_from_langfuse(
                langfuse=mock_langfuse,
                dataset_name="test_dataset",
                run_name="test_run",
            )

    def test_fetch_trace_scores_handles_trace_fetch_error(self) -> None:
        """Test that trace fetch errors are handled gracefully."""
        mock_langfuse = MagicMock()

        # Mock dataset run
        mock_run_item1 = MagicMock()
        mock_run_item1.trace_id = "trace_1"
        mock_run_item2 = MagicMock()
        mock_run_item2.trace_id = "trace_2"

        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [mock_run_item1, mock_run_item2]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        # Mock successful trace 1
        mock_trace1 = MagicMock()
        mock_trace1.input = {"question": "Q1"}
        mock_trace1.output = {"answer": "A1"}
        mock_trace1.metadata = {"ground_truth": "GT1", "question_id": 1}
        mock_trace1.scores = []

        # Second trace fetch fails
        mock_langfuse.api.trace.get.side_effect = [
            mock_trace1,
            Exception("Trace fetch failed"),
        ]

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify only successful trace is returned
        assert len(result["traces"]) == 1
        assert result["traces"][0]["trace_id"] == "trace_1"

    def test_fetch_trace_scores_empty_dataset_run(self) -> None:
        """Test fetching from dataset run with no items."""
        mock_langfuse = MagicMock()

        # Mock empty dataset run
        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = []
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify empty results
        assert len(result["traces"]) == 0
        assert len(result["summary_scores"]) == 0

    def test_fetch_trace_scores_mixed_question_id_types(self) -> None:
        """Test fetching traces with different question_id types (int vs string)."""
        mock_langfuse = MagicMock()

        # Mock dataset run
        mock_run_item1 = MagicMock()
        mock_run_item1.trace_id = "trace_1"
        mock_run_item2 = MagicMock()
        mock_run_item2.trace_id = "trace_2"

        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [mock_run_item1, mock_run_item2]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        # Mock trace 1 with integer question_id
        mock_trace1 = MagicMock()
        mock_trace1.input = {"question": "Q1"}
        mock_trace1.output = {"answer": "A1"}
        mock_trace1.metadata = {"ground_truth": "GT1", "question_id": 123}
        mock_trace1.scores = []

        # Mock trace 2 with string question_id
        mock_trace2 = MagicMock()
        mock_trace2.input = {"question": "Q2"}
        mock_trace2.output = {"answer": "A2"}
        mock_trace2.metadata = {"ground_truth": "GT2", "question_id": "abc-456"}
        mock_trace2.scores = []

        mock_langfuse.api.trace.get.side_effect = [mock_trace1, mock_trace2]

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        # Verify both types are handled correctly
        assert len(result["traces"]) == 2
        assert result["traces"][0]["question_id"] == 123
        assert result["traces"][1]["question_id"] == "abc-456"
