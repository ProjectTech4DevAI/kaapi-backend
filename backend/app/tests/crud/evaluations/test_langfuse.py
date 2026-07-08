from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    fetch_trace_scores_from_langfuse,
    update_traces_with_cosine_scores,
    upload_dataset_to_langfuse,
)


def make_dataset_run_mocks(
    trace_ids: list[str],
) -> tuple[MagicMock, list[MagicMock], list[MagicMock]]:
    """Build a mock v4 Langfuse client for create_langfuse_dataset_run.

    v4 flow: ``langfuse.start_observation(as_type="span")`` -> root span (trace),
    ``root.start_observation(as_type="generation")`` -> generation for cost tracking,
    then ``langfuse.api.dataset_run_items.create(...)`` links the trace to the run.
    Dataset items are named ``item_1..item_N`` to match the test result item_ids.
    """
    langfuse = MagicMock()
    dataset = MagicMock()
    items = []
    roots = []
    gens = []
    for i, trace_id in enumerate(trace_ids, start=1):
        item = MagicMock()
        item.id = f"item_{i}"
        items.append(item)

        root = MagicMock()
        root.trace_id = trace_id
        gen = MagicMock()
        root.start_observation.return_value = gen
        roots.append(root)
        gens.append(gen)

    dataset.items = items
    langfuse.get_dataset.return_value = dataset
    langfuse.start_observation.side_effect = roots
    return langfuse, roots, gens


class TestCreateLangfuseDatasetRun:
    """Test creating Langfuse dataset runs."""

    def test_create_langfuse_dataset_run_success(self) -> None:
        """Test successfully creating a dataset run with traces."""
        mock_langfuse, _, _ = make_dataset_run_mocks(["trace_id_1", "trace_id_2"])

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
        # One root span (trace) per item, each linked to the run via the v4 API
        assert mock_langfuse.start_observation.call_count == 2
        assert mock_langfuse.api.dataset_run_items.create.call_count == 2
        first_link = mock_langfuse.api.dataset_run_items.create.call_args_list[0]
        assert first_link.kwargs["run_name"] == "test_run"
        assert first_link.kwargs["dataset_item_id"] == "item_1"
        assert first_link.kwargs["trace_id"] == "trace_id_1"

    def test_create_langfuse_dataset_run_skips_missing_items(self) -> None:
        """Test that missing dataset items are skipped."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()

        mock_item1 = MagicMock()
        mock_item1.id = "item_1"

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
        mock_langfuse, roots, _ = make_dataset_run_mocks(["trace_id_1", "trace_id_2"])
        # Second item's root span creation fails
        mock_langfuse.start_observation.side_effect = [
            roots[0],
            Exception("Trace creation failed"),
        ]

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

    def test_create_langfuse_dataset_run_handles_rate_limit(self) -> None:
        """A 429 status_code error is handled (logged distinctly) and skipped."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()

        rate_limit_error = Exception("Too Many Requests")
        rate_limit_error.status_code = 429

        mock_item = MagicMock()
        mock_item.id = "item_1"
        mock_item.observe.side_effect = rate_limit_error

        mock_dataset.items = [mock_item]
        mock_langfuse.get_dataset.return_value = mock_dataset

        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "4",
                "ground_truth": "4",
                "response_id": "resp_123",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        ]

        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
            results=results,
        )

        assert trace_id_mapping == {}

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
        """Test that a generation is created with usage when model and usage are provided."""
        mock_langfuse, roots, gens = make_dataset_run_mocks(
            ["trace_id_1", "trace_id_2"]
        )

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

        # Each root span spawns one generation child for cost tracking
        roots[0].start_observation.assert_called_once()
        gen_call = roots[0].start_observation.call_args
        assert gen_call.kwargs["as_type"] == "generation"
        assert gen_call.kwargs["name"] == "evaluation-response"
        assert gen_call.kwargs["input"] == {"question": "What is 2+2?"}
        assert gen_call.kwargs["metadata"]["ground_truth"] == "4"
        assert gen_call.kwargs["metadata"]["response_id"] == "resp_123"

        # v4: generation.update(...) then end(); usage_details is int-only (no "unit")
        gens[0].update.assert_called_once()
        update_call = gens[0].update.call_args
        assert update_call.kwargs["output"] == {"answer": "The answer is 4"}
        assert update_call.kwargs["model"] == "gpt-4o"
        assert update_call.kwargs["usage_details"] == {
            "input": 69,
            "output": 258,
            "total": 327,
        }
        gens[0].end.assert_called_once()

        mock_langfuse.get_dataset.assert_called_once_with("test_dataset")
        mock_langfuse.flush.assert_called_once()
        assert mock_langfuse.start_observation.call_count == 2

    @patch("app.crud.evaluations.langfuse.set_trace_attributes")
    def test_create_langfuse_dataset_run_with_question_id(
        self, mock_set_trace_attributes: MagicMock
    ) -> None:
        """Test that question_id is included in trace and generation metadata."""
        mock_langfuse, roots, _ = make_dataset_run_mocks(["trace_id_1"])

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

        # Verify trace-level metadata carries question_id via set_trace_attributes
        trace_attr_call = mock_set_trace_attributes.call_args
        assert trace_attr_call.kwargs["metadata"]["question_id"] == 1

        # Verify the generation was also created with question_id in metadata
        generation_call = roots[0].start_observation.call_args
        assert generation_call.kwargs["metadata"]["question_id"] == 1

    @patch("app.crud.evaluations.langfuse.set_trace_attributes")
    def test_create_langfuse_dataset_run_without_question_id(
        self, mock_set_trace_attributes: MagicMock
    ) -> None:
        """Test that traces work without question_id (backwards compatibility)."""
        mock_langfuse, _, _ = make_dataset_run_mocks(["trace_id_1"])

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

        # Verify trace-level metadata has no question_id
        trace_attr_call = mock_set_trace_attributes.call_args
        assert "question_id" not in trace_attr_call.kwargs["metadata"]


class TestLangfuseOptOutGuards:
    """Tracing opt-out (langfuse=None) short-circuits before any API call."""

    def test_create_langfuse_dataset_run_returns_empty_when_disabled(self) -> None:
        assert (
            create_langfuse_dataset_run(
                langfuse=None,
                dataset_name="test_dataset",
                run_name="test_run",
                results=[{"item_id": "item_1"}],
            )
            == {}
        )

    def test_upload_dataset_to_langfuse_returns_none_zero_when_disabled(self) -> None:
        assert upload_dataset_to_langfuse(
            langfuse=None,
            items=[{"question": "q", "answer": "a"}],
            dataset_name="test_dataset",
            duplication_factor=3,
        ) == (None, 0)


class TestUpdateTracesWithCosineScores:
    """Test updating Langfuse traces with cosine similarity scores."""

    def test_update_traces_with_cosine_scores_success(self) -> None:
        """Test successfully updating traces with scores, returning no failures."""
        mock_langfuse = MagicMock()

        per_item_scores = [
            {"trace_id": "trace_1", "cosine_similarity": 0.95},
            {"trace_id": "trace_2", "cosine_similarity": 0.87},
            {"trace_id": "trace_3", "cosine_similarity": 0.92},
        ]

        failed = update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=per_item_scores
        )

        assert failed == []
        assert mock_langfuse.create_score.call_count == 3

        calls = mock_langfuse.create_score.call_args_list
        assert calls[0].kwargs["trace_id"] == "trace_1"
        assert calls[0].kwargs["name"] == "Cosine Similarity"
        assert calls[0].kwargs["value"] == 0.95
        assert "cosine similarity" in calls[0].kwargs["comment"].lower()

        assert calls[1].kwargs["trace_id"] == "trace_2"
        assert calls[1].kwargs["value"] == 0.87

        mock_langfuse.flush.assert_called_once()

    def test_update_traces_with_cosine_scores_unscoreable(self) -> None:
        """Unscoreable items are written as 0 with a 'Cannot compute' comment."""
        mock_langfuse = MagicMock()

        per_item_scores = [
            {"trace_id": "trace_1", "cosine_similarity": 0.95},
            {"trace_id": "trace_2", "unscoreable": True, "reason": "empty_output"},
        ]

        failed = update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=per_item_scores
        )

        assert failed == []
        calls = mock_langfuse.create_score.call_args_list
        assert calls[1].kwargs["trace_id"] == "trace_2"
        assert calls[1].kwargs["value"] == 0
        assert calls[1].kwargs["comment"] == "Cannot compute: empty_output"

    def test_update_traces_with_cosine_scores_missing_trace_id(self) -> None:
        """Test that items without trace_id are skipped."""
        mock_langfuse = MagicMock()

        per_item_scores = [
            {"trace_id": "trace_1", "cosine_similarity": 0.95},
            {"cosine_similarity": 0.87},
            {"trace_id": "trace_3", "cosine_similarity": 0.92},
        ]

        failed = update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=per_item_scores
        )

        assert failed == []
        assert mock_langfuse.create_score.call_count == 2

    def test_update_traces_with_cosine_scores_reports_failure(
        self,
    ) -> None:
        """A failing write is reported (not raised); a cron retries it later."""
        mock_langfuse = MagicMock()

        # trace_2 fails; trace_1 and trace_3 succeed.
        def score_side_effect(*args: Any, **kwargs: Any) -> None:
            if kwargs.get("trace_id") == "trace_2":
                raise Exception("Score failed")

        mock_langfuse.create_score.side_effect = score_side_effect

        per_item_scores = [
            {"trace_id": "trace_1", "cosine_similarity": 0.95},
            {"trace_id": "trace_2", "cosine_similarity": 0.87},
            {"trace_id": "trace_3", "cosine_similarity": 0.92},
        ]

        failed = update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=per_item_scores
        )

        # Only the failing trace is reported.
        assert failed == ["trace_2"]
        # No retries: one score call per trace.
        assert mock_langfuse.create_score.call_count == 3
        mock_langfuse.flush.assert_called_once()

    def test_update_traces_with_cosine_scores_empty_list(self) -> None:
        """Test with empty scores list."""
        mock_langfuse = MagicMock()

        failed = update_traces_with_cosine_scores(
            langfuse=mock_langfuse, per_item_scores=[]
        )

        assert failed == []
        mock_langfuse.create_score.assert_not_called()
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

    def test_upload_dataset_writes_category_only_when_item_has_it(self) -> None:
        """`category` is optional: items without it must not produce a Langfuse
        metadata `category` field, while items that have it must.

        This is what keeps the no-category-column upload path clean end-to-end.
        """
        items = [
            {"question": "q1", "answer": "a1"},  # no category
            {"question": "q2", "answer": "a2", "category": "Health"},
        ]
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset_123"
        mock_langfuse.create_dataset.return_value = mock_dataset

        upload_dataset_to_langfuse(
            langfuse=mock_langfuse,
            items=items,
            dataset_name="test_dataset",
            duplication_factor=1,
        )

        calls = mock_langfuse.create_dataset_item.call_args_list
        assert len(calls) == 2

        metadatas_by_question = {
            call.kwargs["input"]["question"]: call.kwargs["metadata"] for call in calls
        }
        assert "category" not in metadatas_by_question["q1"]
        assert metadatas_by_question["q2"]["category"] == "Health"


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
        # No category in metadata → no category on trace (omitted, not "Other")
        assert "category" not in trace

    def test_fetch_trace_scores_category_set_only_when_metadata_has_one(self) -> None:
        """`category` on the returned trace is omitted when the Langfuse metadata
        carries none, and title-cased when it does.

        This covers both branches of the conditional in ``_fetch_single_trace``
        and pins down the no-default-Other behaviour for datasets uploaded
        without a category column.
        """
        mock_langfuse = MagicMock()

        mock_run_item_a = MagicMock()
        mock_run_item_a.trace_id = "trace_with_cat"
        mock_run_item_b = MagicMock()
        mock_run_item_b.trace_id = "trace_no_cat"
        mock_dataset_run = MagicMock()
        mock_dataset_run.dataset_run_items = [mock_run_item_a, mock_run_item_b]
        mock_langfuse.api.datasets.get_run.return_value = mock_dataset_run

        mock_trace_a = MagicMock()
        mock_trace_a.input = {"question": "q1"}
        mock_trace_a.output = {"answer": "a1"}
        mock_trace_a.metadata = {
            "ground_truth": "a1",
            "question_id": 1,
            "category": "health",
        }
        mock_trace_a.scores = []

        mock_trace_b = MagicMock()
        mock_trace_b.input = {"question": "q2"}
        mock_trace_b.output = {"answer": "a2"}
        mock_trace_b.metadata = {"ground_truth": "a2", "question_id": 2}
        mock_trace_b.scores = []

        mock_langfuse.api.trace.get.side_effect = [mock_trace_a, mock_trace_b]

        result = fetch_trace_scores_from_langfuse(
            langfuse=mock_langfuse,
            dataset_name="test_dataset",
            run_name="test_run",
        )

        traces_by_id = {t["trace_id"]: t for t in result["traces"]}
        assert traces_by_id["trace_with_cat"]["category"] == "Health"
        assert "category" not in traces_by_id["trace_no_cat"]

        # v4: trace.get must request only core,io,scores to avoid fetching the
        # full trace (all observations/metrics) which would time out in production.
        _, kwargs = mock_langfuse.api.trace.get.call_args
        assert kwargs.get("fields") == "core,io,scores"

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

    def test_fetch_trace_scores_includes_partial_scores(self) -> None:
        """Test that scores present in only some traces are still included."""
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
        mock_score1b.name = "partial_score"
        mock_score1b.value = 0.8
        mock_score1b.comment = None
        mock_score1b.data_type = "NUMERIC"
        mock_trace1.scores = [mock_score1a, mock_score1b]

        # Mock trace 2 with only one score (partial_score is missing)
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

        # Verify both scores are included in summary
        assert len(result["summary_scores"]) == 2
        summary_names = {s["name"] for s in result["summary_scores"]}
        assert summary_names == {"complete_score", "partial_score"}

        # Verify complete_score summary (present in both traces)
        complete_summary = next(
            s for s in result["summary_scores"] if s["name"] == "complete_score"
        )
        assert complete_summary["avg"] == 0.8  # (0.9 + 0.7) / 2
        assert complete_summary["total_pairs"] == 2

        # Verify partial_score summary (present in only one trace)
        partial_summary = next(
            s for s in result["summary_scores"] if s["name"] == "partial_score"
        )
        assert partial_summary["avg"] == 0.8
        assert partial_summary["total_pairs"] == 1

        # Verify trace 1 has both scores, trace 2 has only one
        assert len(result["traces"][0]["scores"]) == 2
        assert len(result["traces"][1]["scores"]) == 1

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
