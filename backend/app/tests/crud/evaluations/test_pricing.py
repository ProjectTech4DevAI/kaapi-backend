import pytest

from app.crud.evaluations.pricing import (
    COST_USD_DECIMALS,
    MODEL_PRICING,
    build_cost_dict,
    build_embedding_cost_entry,
    build_response_cost_entry,
    calculate_token_cost,
)


class TestCalculateTokenCost:
    """Tests for calculate_token_cost function."""

    def test_known_chat_model_input_and_output(self) -> None:
        """Cost is sum of input and output token costs for a known chat model."""
        pricing = MODEL_PRICING["gpt-4o"]
        expected = (
            1000 * pricing["input_cost_per_token"]
            + 500 * pricing["output_cost_per_token"]
        )

        cost = calculate_token_cost(
            model="gpt-4o", input_tokens=1000, output_tokens=500
        )

        assert cost == pytest.approx(expected)

    def test_known_embedding_model_defaults_output_tokens_to_zero(self) -> None:
        """Embedding models charge only for input tokens; output_tokens defaults to 0."""
        pricing = MODEL_PRICING["text-embedding-3-large"]
        expected = 2000 * pricing["input_cost_per_token"]

        cost = calculate_token_cost(model="text-embedding-3-large", input_tokens=2000)

        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self) -> None:
        """Unknown models return 0.0 instead of raising."""
        cost = calculate_token_cost(
            model="not-a-real-model", input_tokens=100, output_tokens=50
        )

        assert cost == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        """Zero tokens for a known model returns zero cost."""
        cost = calculate_token_cost(model="gpt-4o", input_tokens=0, output_tokens=0)

        assert cost == 0.0

    def test_embedding_model_with_explicit_output_tokens(self) -> None:
        """Passing output_tokens to an embedding model adds 0 cost (no output rate)."""
        pricing = MODEL_PRICING["text-embedding-3-large"]
        expected = 100 * pricing["input_cost_per_token"]

        cost = calculate_token_cost(
            model="text-embedding-3-large", input_tokens=100, output_tokens=999
        )

        assert cost == pytest.approx(expected)


class TestBuildResponseCostEntry:
    """Tests for build_response_cost_entry function."""

    def test_basic_aggregation(self) -> None:
        """Sums input/output/total tokens across results and computes USD cost."""
        results = [
            {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                }
            },
            {
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 75,
                    "total_tokens": 275,
                }
            },
        ]

        entry = build_response_cost_entry(model="gpt-4o", results=results)

        assert entry["model"] == "gpt-4o"
        assert entry["input_tokens"] == 300
        assert entry["output_tokens"] == 125
        assert entry["total_tokens"] == 425
        pricing = MODEL_PRICING["gpt-4o"]
        expected_cost = round(
            300 * pricing["input_cost_per_token"]
            + 125 * pricing["output_cost_per_token"],
            COST_USD_DECIMALS,
        )
        assert entry["cost_usd"] == expected_cost

    def test_empty_results(self) -> None:
        """Empty results yields zero tokens and zero cost."""
        entry = build_response_cost_entry(model="gpt-4o", results=[])

        assert entry["input_tokens"] == 0
        assert entry["output_tokens"] == 0
        assert entry["total_tokens"] == 0
        assert entry["cost_usd"] == 0.0

    def test_results_missing_usage_are_skipped(self) -> None:
        """Items without a usage dict are skipped without raising."""
        results = [
            {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            {},  # No usage key
            {"usage": None},  # Explicit None
        ]

        entry = build_response_cost_entry(model="gpt-4o", results=results)

        assert entry["input_tokens"] == 10
        assert entry["output_tokens"] == 5
        assert entry["total_tokens"] == 15

    def test_unknown_model_yields_zero_cost(self) -> None:
        """Unknown model still aggregates token counts but reports zero cost."""
        results = [
            {"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
        ]

        entry = build_response_cost_entry(model="mystery-model", results=results)

        assert entry["input_tokens"] == 100
        assert entry["output_tokens"] == 50
        assert entry["cost_usd"] == 0.0


class TestBuildEmbeddingCostEntry:
    """Tests for build_embedding_cost_entry function."""

    def test_basic_aggregation(self) -> None:
        """Sums prompt/total tokens from raw batch results and computes USD cost."""
        raw_results = [
            {
                "response": {
                    "body": {"usage": {"prompt_tokens": 100, "total_tokens": 100}}
                }
            },
            {
                "response": {
                    "body": {"usage": {"prompt_tokens": 250, "total_tokens": 250}}
                }
            },
        ]

        entry = build_embedding_cost_entry(
            model="text-embedding-3-large", raw_results=raw_results
        )

        assert entry["model"] == "text-embedding-3-large"
        assert entry["prompt_tokens"] == 350
        assert entry["total_tokens"] == 350
        pricing = MODEL_PRICING["text-embedding-3-large"]
        expected_cost = round(350 * pricing["input_cost_per_token"], COST_USD_DECIMALS)
        assert entry["cost_usd"] == expected_cost

    def test_empty_raw_results(self) -> None:
        """Empty raw_results yields zero tokens and zero cost."""
        entry = build_embedding_cost_entry(
            model="text-embedding-3-large", raw_results=[]
        )

        assert entry["prompt_tokens"] == 0
        assert entry["total_tokens"] == 0
        assert entry["cost_usd"] == 0.0

    def test_results_missing_usage_are_skipped(self) -> None:
        """Items without nested usage are skipped (e.g., error rows)."""
        raw_results = [
            {
                "response": {
                    "body": {"usage": {"prompt_tokens": 50, "total_tokens": 50}}
                }
            },
            {"error": {"message": "Rate limited"}},  # No response.body.usage
            {"response": {"body": {}}},  # body present, usage missing
        ]

        entry = build_embedding_cost_entry(
            model="text-embedding-3-large", raw_results=raw_results
        )

        assert entry["prompt_tokens"] == 50
        assert entry["total_tokens"] == 50

    def test_unknown_model_yields_zero_cost(self) -> None:
        """Unknown embedding model still aggregates tokens but reports zero cost."""
        raw_results = [
            {
                "response": {
                    "body": {"usage": {"prompt_tokens": 100, "total_tokens": 100}}
                }
            }
        ]

        entry = build_embedding_cost_entry(
            model="mystery-embed", raw_results=raw_results
        )

        assert entry["prompt_tokens"] == 100
        assert entry["cost_usd"] == 0.0


class TestBuildCostDict:
    """Tests for build_cost_dict function."""

    def test_response_only(self) -> None:
        """Only response entry → embedding key absent, total = response cost."""
        response_entry = {
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.001234,
        }

        cost = build_cost_dict(response_entry=response_entry)

        assert cost["response"] == response_entry
        assert "embedding" not in cost
        assert cost["total_cost_usd"] == 0.001234

    def test_embedding_only(self) -> None:
        """Only embedding entry → response key absent, total = embedding cost."""
        embedding_entry = {
            "model": "text-embedding-3-large",
            "prompt_tokens": 200,
            "total_tokens": 200,
            "cost_usd": 0.000013,
        }

        cost = build_cost_dict(embedding_entry=embedding_entry)

        assert cost["embedding"] == embedding_entry
        assert "response" not in cost
        assert cost["total_cost_usd"] == 0.000013

    def test_both_entries(self) -> None:
        """Both entries → both keys present, total = sum of both costs."""
        response_entry = {"cost_usd": 0.001234}
        embedding_entry = {"cost_usd": 0.000013}

        cost = build_cost_dict(
            response_entry=response_entry, embedding_entry=embedding_entry
        )

        assert cost["response"] == response_entry
        assert cost["embedding"] == embedding_entry
        assert cost["total_cost_usd"] == round(0.001234 + 0.000013, COST_USD_DECIMALS)

    def test_neither_entry(self) -> None:
        """No entries → only total_cost_usd present, equal to 0.0."""
        cost = build_cost_dict()

        assert cost == {"total_cost_usd": 0.0}

    def test_total_is_rounded(self) -> None:
        """total_cost_usd is rounded to COST_USD_DECIMALS."""
        response_entry = {"cost_usd": 0.0000001}
        embedding_entry = {"cost_usd": 0.0000002}

        cost = build_cost_dict(
            response_entry=response_entry, embedding_entry=embedding_entry
        )

        # 0.0000003 rounded to 6 decimals → 0.0
        assert cost["total_cost_usd"] == 0.0

    def test_entry_missing_cost_usd_treated_as_zero(self) -> None:
        """Entries without a cost_usd key default to 0 in the total."""
        response_entry = {"model": "gpt-4o"}  # No cost_usd
        embedding_entry = {"cost_usd": 0.000050}

        cost = build_cost_dict(
            response_entry=response_entry, embedding_entry=embedding_entry
        )

        assert cost["total_cost_usd"] == 0.000050
