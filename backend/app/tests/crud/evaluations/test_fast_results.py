"""Per-item result shapes extracted from the fast-eval stages (`fast_results.py`).

These are the units uploaded to S3, so the key set and the "missing counter reads
as 0" rule are part of the on-disk contract, not implementation detail.
"""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.crud.evaluations.fast_results import (
    EMBEDDING_USAGE_KEYS,
    RESPONSE_USAGE_KEYS,
    build_embedding_failure,
    build_response_result,
    extract_usage,
    is_failure_threshold_breached,
    parse_embedding_pair,
    sum_usage,
)


class TestBuildResponseResult:
    def test_carries_every_key_the_s3_unit_needs(self) -> None:
        result = build_response_result(
            item_id="item_0_0",
            question="q",
            ground_truth="gt",
            question_id=1,
            generated_output="out",
            failed=False,
        )
        assert set(result) == {
            "item_id",
            "question",
            "generated_output",
            "ground_truth",
            "response_id",
            "usage",
            "question_id",
            "failed",
            "retrieved_chunks",
        }

    def test_optional_fields_default_to_none(self) -> None:
        result = build_response_result(
            item_id="i",
            question="q",
            ground_truth="gt",
            question_id=None,
            generated_output="ERROR: boom",
            failed=True,
        )
        assert result["response_id"] is None
        assert result["usage"] is None
        assert result["retrieved_chunks"] is None


class TestBuildEmbeddingFailure:
    def test_marks_failed_with_both_vectors_absent(self) -> None:
        result = build_embedding_failure("item_1_0", "empty output or ground_truth")
        assert result["failed"] is True
        assert result["output_embedding"] is None
        assert result["ground_truth_embedding"] is None
        assert result["error"] == "empty output or ground_truth"


class TestExtractUsage:
    def test_reads_requested_counters(self) -> None:
        usage = SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14)
        assert extract_usage(usage, RESPONSE_USAGE_KEYS) == {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
        }

    def test_missing_or_none_counters_read_as_zero(self) -> None:
        assert extract_usage(None, EMBEDDING_USAGE_KEYS) == {
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        assert extract_usage(
            SimpleNamespace(prompt_tokens=None), ("prompt_tokens",)
        ) == {"prompt_tokens": 0}


class TestParseEmbeddingPair:
    @staticmethod
    def _response(pairs: list[tuple[int, list[float]]]) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=v) for i, v in pairs],
            usage=SimpleNamespace(prompt_tokens=6, total_tokens=6),
        )

    def test_index_zero_is_output_and_index_one_is_ground_truth(self) -> None:
        result = parse_embedding_pair(
            item_id="i", response=self._response([(0, [1.0]), (1, [2.0])])
        )
        assert result["output_embedding"] == [1.0]
        assert result["ground_truth_embedding"] == [2.0]
        assert result["failed"] is False
        assert result["usage"] == {"prompt_tokens": 6, "total_tokens": 6}

    def test_order_in_the_payload_does_not_matter(self) -> None:
        result = parse_embedding_pair(
            item_id="i", response=self._response([(1, [2.0]), (0, [1.0])])
        )
        assert result["output_embedding"] == [1.0]
        assert result["ground_truth_embedding"] == [2.0]

    def test_short_payload_is_a_failure(self) -> None:
        result = parse_embedding_pair(
            item_id="i", response=self._response([(0, [1.0])])
        )
        assert result["failed"] is True
        assert "expected 2 embeddings" in result["error"]


class TestSumUsage:
    def test_sums_requested_keys_across_results(self) -> None:
        results = [
            {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}},
            {"usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}},
        ]
        assert sum_usage(results, RESPONSE_USAGE_KEYS) == {
            "input_tokens": 11,
            "output_tokens": 22,
            "total_tokens": 33,
        }

    def test_absent_or_null_usage_contributes_zero(self) -> None:
        results = [{"usage": None}, {}, {"usage": {"total_tokens": 5}}]
        assert sum_usage(results, ("total_tokens",)) == {"total_tokens": 5}

    def test_empty_results_give_zeroed_totals(self) -> None:
        assert sum_usage([], EMBEDDING_USAGE_KEYS) == {
            "prompt_tokens": 0,
            "total_tokens": 0,
        }


class TestIsFailureThresholdBreached:
    def test_empty_run_never_breaches(self) -> None:
        assert is_failure_threshold_breached(failed_rows=0, total_rows=0) is False

    @pytest.mark.parametrize("failed_rows", [0, 1, 5])
    def test_at_or_below_threshold_passes(self, failed_rows: int) -> None:
        total_rows = 10
        assert (
            failed_rows / total_rows <= settings.EVAL_FAST_FAILURE_THRESHOLD
        ), "fixture assumes the default 0.5 threshold"
        assert (
            is_failure_threshold_breached(
                failed_rows=failed_rows, total_rows=total_rows
            )
            is False
        )

    def test_strictly_above_threshold_breaches(self) -> None:
        total_rows = 10
        failed_rows = int(settings.EVAL_FAST_FAILURE_THRESHOLD * total_rows) + 1
        assert (
            is_failure_threshold_breached(
                failed_rows=failed_rows, total_rows=total_rows
            )
            is True
        )
