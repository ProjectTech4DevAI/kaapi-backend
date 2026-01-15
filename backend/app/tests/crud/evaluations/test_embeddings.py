import pytest

from app.crud.evaluations.embeddings import (
    build_embedding_jsonl,
    calculate_average_similarity,
    calculate_cosine_similarity,
    parse_embedding_results,
)


class TestBuildEmbeddingJsonl:
    """Tests for build_embedding_jsonl function."""

    def test_build_embedding_jsonl_basic(self) -> None:
        """Test building JSONL for basic evaluation results."""
        results = [
            {
                "item_id": "item_1",
                "question": "What is 2+2?",
                "generated_output": "The answer is 4",
                "ground_truth": "4",
            },
            {
                "item_id": "item_2",
                "question": "What is the capital of France?",
                "generated_output": "Paris",
                "ground_truth": "Paris",
            },
        ]

        trace_id_mapping = {
            "item_1": "trace_1",
            "item_2": "trace_2",
        }

        jsonl_data = build_embedding_jsonl(results, trace_id_mapping)

        assert len(jsonl_data) == 2

        assert jsonl_data[0]["custom_id"] == "trace_1"
        assert jsonl_data[0]["method"] == "POST"
        assert jsonl_data[0]["url"] == "/v1/embeddings"
        assert jsonl_data[0]["body"]["model"] == "text-embedding-3-large"
        assert jsonl_data[0]["body"]["input"] == ["The answer is 4", "4"]
        assert jsonl_data[0]["body"]["encoding_format"] == "float"

    def test_build_embedding_jsonl_custom_model(self) -> None:
        """Test building JSONL with custom embedding model."""
        results = [
            {
                "item_id": "item_1",
                "question": "Test?",
                "generated_output": "Output",
                "ground_truth": "Truth",
            }
        ]

        trace_id_mapping = {"item_1": "trace_1"}

        jsonl_data = build_embedding_jsonl(
            results, trace_id_mapping, embedding_model="text-embedding-3-small"
        )

        assert len(jsonl_data) == 1
        assert jsonl_data[0]["body"]["model"] == "text-embedding-3-small"

    def test_build_embedding_jsonl_skips_empty(self) -> None:
        """Test that items with empty output or ground_truth are skipped."""
        results = [
            {
                "item_id": "item_1",
                "question": "Test?",
                "generated_output": "",
                "ground_truth": "Truth",
            },
            {
                "item_id": "item_2",
                "question": "Test?",
                "generated_output": "Output",
                "ground_truth": "",
            },
            {
                "item_id": "item_3",
                "question": "Test?",
                "generated_output": "Output",
                "ground_truth": "Truth",
            },
        ]

        trace_id_mapping = {
            "item_1": "trace_1",
            "item_2": "trace_2",
            "item_3": "trace_3",
        }

        jsonl_data = build_embedding_jsonl(results, trace_id_mapping)

        assert len(jsonl_data) == 1
        assert jsonl_data[0]["custom_id"] == "trace_3"

    def test_build_embedding_jsonl_missing_item_id(self) -> None:
        """Test that items without item_id or trace_id are skipped."""
        results = [
            {
                "question": "Test?",
                "generated_output": "Output",
                "ground_truth": "Truth",
            },
            {
                "item_id": "item_2",
                "question": "Test?",
                "generated_output": "Output",
                "ground_truth": "Truth",
            },
        ]

        trace_id_mapping = {"item_2": "trace_2"}

        jsonl_data = build_embedding_jsonl(results, trace_id_mapping)

        assert len(jsonl_data) == 1
        assert jsonl_data[0]["custom_id"] == "trace_2"


class TestParseEmbeddingResults:
    """Tests for parse_embedding_results function."""

    def test_parse_embedding_results_basic(self) -> None:
        """Test parsing basic embedding results."""
        raw_results = [
            {
                "custom_id": "trace_1",
                "response": {
                    "body": {
                        "data": [
                            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                            {"index": 1, "embedding": [0.15, 0.22, 0.32]},
                        ]
                    }
                },
            },
            {
                "custom_id": "trace_2",
                "response": {
                    "body": {
                        "data": [
                            {"index": 0, "embedding": [0.5, 0.6, 0.7]},
                            {"index": 1, "embedding": [0.55, 0.65, 0.75]},
                        ]
                    }
                },
            },
        ]

        embedding_pairs = parse_embedding_results(raw_results)

        assert len(embedding_pairs) == 2

        assert embedding_pairs[0]["trace_id"] == "trace_1"
        assert embedding_pairs[0]["output_embedding"] == [0.1, 0.2, 0.3]
        assert embedding_pairs[0]["ground_truth_embedding"] == [0.15, 0.22, 0.32]

        assert embedding_pairs[1]["trace_id"] == "trace_2"
        assert embedding_pairs[1]["output_embedding"] == [0.5, 0.6, 0.7]
        assert embedding_pairs[1]["ground_truth_embedding"] == [0.55, 0.65, 0.75]

    def test_parse_embedding_results_with_error(self) -> None:
        """Test parsing results with errors."""
        raw_results = [
            {
                "custom_id": "trace_1",
                "error": {"message": "Rate limit exceeded"},
            },
            {
                "custom_id": "trace_2",
                "response": {
                    "body": {
                        "data": [
                            {"index": 0, "embedding": [0.1, 0.2]},
                            {"index": 1, "embedding": [0.15, 0.22]},
                        ]
                    }
                },
            },
        ]

        embedding_pairs = parse_embedding_results(raw_results)

        assert len(embedding_pairs) == 1
        assert embedding_pairs[0]["trace_id"] == "trace_2"

    def test_parse_embedding_results_missing_embedding(self) -> None:
        """Test parsing results with missing embeddings."""
        raw_results = [
            {
                "custom_id": "trace_1",
                "response": {
                    "body": {
                        "data": [
                            {"index": 0, "embedding": [0.1, 0.2]},
                            # Missing index 1
                        ]
                    }
                },
            },
            {
                "custom_id": "trace_2",
                "response": {
                    "body": {
                        "data": [
                            {"index": 0, "embedding": [0.1, 0.2]},
                            {"index": 1, "embedding": [0.15, 0.22]},
                        ]
                    }
                },
            },
        ]

        embedding_pairs = parse_embedding_results(raw_results)

        assert len(embedding_pairs) == 1
        assert embedding_pairs[0]["trace_id"] == "trace_2"


class TestCalculateCosineSimilarity:
    """Tests for calculate_cosine_similarity function."""

    def test_calculate_cosine_similarity_identical(self) -> None:
        """Test cosine similarity of identical vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = calculate_cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(1.0)

    def test_calculate_cosine_similarity_orthogonal(self) -> None:
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        similarity = calculate_cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0)

    def test_calculate_cosine_similarity_opposite(self) -> None:
        """Test cosine similarity of opposite vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]

        similarity = calculate_cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(-1.0)

    def test_calculate_cosine_similarity_partial(self) -> None:
        """Test cosine similarity of partially similar vectors."""
        vec1 = [1.0, 1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = calculate_cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.707, abs=0.01)

    def test_calculate_cosine_similarity_zero_vector(self) -> None:
        """Test cosine similarity with zero vector."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = calculate_cosine_similarity(vec1, vec2)

        assert similarity == 0.0


class TestCalculateAverageSimilarity:
    """Tests for calculate_average_similarity function."""

    def test_calculate_average_similarity_basic(self) -> None:
        """Test calculating average similarity for basic embedding pairs."""
        embedding_pairs = [
            {
                "trace_id": "trace_1",
                "output_embedding": [1.0, 0.0, 0.0],
                "ground_truth_embedding": [1.0, 0.0, 0.0],
            },
            {
                "trace_id": "trace_2",
                "output_embedding": [1.0, 0.0, 0.0],
                "ground_truth_embedding": [0.0, 1.0, 0.0],
            },
            {
                "trace_id": "trace_3",
                "output_embedding": [1.0, 1.0, 0.0],
                "ground_truth_embedding": [1.0, 0.0, 0.0],
            },
        ]

        stats = calculate_average_similarity(embedding_pairs)

        assert stats["total_pairs"] == 3
        # Average of [1.0, 0.0, 0.707] ≈ 0.569
        assert stats["cosine_similarity_avg"] == pytest.approx(0.569, abs=0.01)
        assert "cosine_similarity_std" in stats
        assert len(stats["per_item_scores"]) == 3

    def test_calculate_average_similarity_empty(self) -> None:
        """Test calculating average similarity for empty list."""
        embedding_pairs = []

        stats = calculate_average_similarity(embedding_pairs)

        assert stats["total_pairs"] == 0
        assert stats["cosine_similarity_avg"] == 0.0
        assert stats["cosine_similarity_std"] == 0.0
        assert stats["per_item_scores"] == []

    def test_calculate_average_similarity_per_item_scores(self) -> None:
        """Test that per-item scores are correctly calculated."""
        embedding_pairs = [
            {
                "trace_id": "trace_1",
                "output_embedding": [1.0, 0.0],
                "ground_truth_embedding": [1.0, 0.0],
            },
            {
                "trace_id": "trace_2",
                "output_embedding": [0.0, 1.0],
                "ground_truth_embedding": [0.0, 1.0],
            },
        ]

        stats = calculate_average_similarity(embedding_pairs)

        assert len(stats["per_item_scores"]) == 2
        assert stats["per_item_scores"][0]["trace_id"] == "trace_1"
        assert stats["per_item_scores"][0]["cosine_similarity"] == pytest.approx(1.0)
        assert stats["per_item_scores"][1]["trace_id"] == "trace_2"
        assert stats["per_item_scores"][1]["cosine_similarity"] == pytest.approx(1.0)

    def test_calculate_average_similarity_statistics(self) -> None:
        """Test that all statistics are calculated correctly."""
        embedding_pairs = [
            {
                "trace_id": "trace_1",
                "output_embedding": [1.0, 0.0],
                "ground_truth_embedding": [1.0, 0.0],
            },
            {
                "trace_id": "trace_2",
                "output_embedding": [1.0, 0.0],
                "ground_truth_embedding": [0.0, 1.0],
            },
            {
                "trace_id": "trace_3",
                "output_embedding": [1.0, 0.0],
                "ground_truth_embedding": [1.0, 0.0],
            },
            {
                "trace_id": "trace_4",
                "output_embedding": [1.0, 0.0],
                "ground_truth_embedding": [0.0, 1.0],
            },
        ]

        stats = calculate_average_similarity(embedding_pairs)

        # Similarities = [1.0, 0.0, 1.0, 0.0]
        assert stats["cosine_similarity_avg"] == pytest.approx(0.5)
        # Standard deviation of [1, 0, 1, 0] = 0.5
        assert stats["cosine_similarity_std"] == pytest.approx(0.5)
        assert stats["total_pairs"] == 4
