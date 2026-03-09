"""Test cases for STT evaluation metrics calculation."""

from unittest.mock import patch

import pytest

from app.services.stt_evaluations.metrics import (
    METRIC_CER,
    METRIC_LENIENT_WER,
    METRIC_NAMES,
    METRIC_WER,
    METRIC_WIP,
    SCORE_DATA_TYPE_NUMERIC,
    calculate_stt_metrics,
    collapse_whitespace,
    compute_run_aggregate_scores,
    normalize_text,
)


class TestCollapseWhitespace:
    """Test cases for collapse_whitespace."""

    def test_single_spaces_unchanged(self) -> None:
        assert collapse_whitespace("hello world") == "hello world"

    def test_multiple_spaces_collapsed(self) -> None:
        assert collapse_whitespace("hello   world") == "hello world"

    def test_tabs_and_newlines_collapsed(self) -> None:
        assert collapse_whitespace("hello\t\nworld") == "hello world"

    def test_leading_trailing_stripped(self) -> None:
        assert collapse_whitespace("  hello world  ") == "hello world"

    def test_empty_string(self) -> None:
        assert collapse_whitespace("") == ""

    def test_only_whitespace(self) -> None:
        assert collapse_whitespace("   \t\n  ") == ""

    def test_mixed_whitespace(self) -> None:
        assert collapse_whitespace("  hello  \t world \n foo  ") == "hello world foo"


class TestNormalizeText:
    """Test cases for normalize_text."""

    def test_empty_string(self) -> None:
        assert normalize_text("", "en") == ""

    def test_none_language_code_whitespace_only(self) -> None:
        result = normalize_text("hello   world", None)
        assert result == "hello world"

    def test_unsupported_language_whitespace_only(self) -> None:
        result = normalize_text("hello   world", "ur")
        assert result == "hello world"

    def test_indic_language_hi(self) -> None:
        """Test Hindi normalization uses indic-nlp-library."""
        result = normalize_text("नमस्ते  दुनिया", "hi")
        # Should at least collapse whitespace; normalization may alter characters
        assert "  " not in result
        assert len(result) > 0

    def test_indic_language_mr_maps_to_hi(self) -> None:
        """Test Marathi uses Hindi normalizer (Devanagari script)."""
        result = normalize_text("नमस्ते  दुनिया", "mr")
        assert "  " not in result
        assert len(result) > 0

    def test_whisper_normalizer_en(self) -> None:
        """Test English normalization uses whisper-normalizer."""
        result = normalize_text("Hello, World!", "en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_whisper_normalizer_as(self) -> None:
        """Test Assamese normalization uses Bengali normalizer."""
        result = normalize_text("test text", "as")
        assert isinstance(result, str)

    @patch("app.services.stt_evaluations.metrics._normalizer_factory")
    def test_indic_normalization_failure_falls_back(self, mock_factory) -> None:
        """Test that indic normalization failure falls back to whitespace-only."""
        mock_normalizer = mock_factory.get_normalizer.return_value
        mock_normalizer.normalize.side_effect = RuntimeError("normalizer error")

        result = normalize_text("hello  world", "hi")
        assert result == "hello world"

    @patch("app.services.stt_evaluations.metrics._whisper_normalizers")
    def test_whisper_normalization_failure_falls_back(self, mock_normalizers) -> None:
        """Test that whisper normalization failure falls back to whitespace-only."""
        mock_normalizers.__contains__ = lambda self, key: key == "en"
        mock_normalizers.__getitem__ = lambda self, key: (_ for _ in ()).throw(
            RuntimeError("whisper error")
        )

        result = normalize_text("hello  world", "en")
        assert result == "hello world"

    @pytest.mark.parametrize(
        "lang", ["hi", "bn", "gu", "pa", "kn", "ml", "or", "ta", "te", "mr"]
    )
    def test_all_indic_languages_accepted(self, lang: str) -> None:
        """Test that all listed Indic languages are handled without error."""
        result = normalize_text("test", lang)
        assert isinstance(result, str)


class TestCalculateSTTMetrics:
    """Test cases for calculate_stt_metrics."""

    def test_perfect_match(self) -> None:
        """Test identical hypothesis and reference."""
        scores = calculate_stt_metrics(
            hypothesis="hello world",
            reference="hello world",
            language_code=None,
        )
        assert scores[METRIC_WER] == 0.0
        assert scores[METRIC_CER] == 0.0
        assert scores[METRIC_LENIENT_WER] == 0.0
        assert scores[METRIC_WIP] == 1.0

    def test_completely_wrong(self) -> None:
        """Test completely different hypothesis and reference."""
        scores = calculate_stt_metrics(
            hypothesis="abc def",
            reference="xyz uvw",
            language_code=None,
        )
        assert scores[METRIC_WER] == 1.0
        assert scores[METRIC_CER] > 0.0
        assert scores[METRIC_WIP] == 0.0

    def test_partial_match(self) -> None:
        """Test partially matching hypothesis."""
        scores = calculate_stt_metrics(
            hypothesis="hello world foo",
            reference="hello world bar",
            language_code=None,
        )
        assert 0.0 < scores[METRIC_WER] < 1.0
        assert 0.0 < scores[METRIC_CER] < 1.0

    def test_empty_reference_empty_hypothesis(self) -> None:
        """Test both reference and hypothesis empty."""
        scores = calculate_stt_metrics(
            hypothesis="",
            reference="",
            language_code=None,
        )
        assert scores[METRIC_WER] == 0.0
        assert scores[METRIC_CER] == 0.0
        assert scores[METRIC_LENIENT_WER] == 0.0
        assert scores[METRIC_WIP] == 1.0

    def test_empty_reference_nonempty_hypothesis(self) -> None:
        """Test empty reference with non-empty hypothesis."""
        scores = calculate_stt_metrics(
            hypothesis="hello",
            reference="   ",
            language_code=None,
        )
        assert scores[METRIC_WER] == 1.0
        assert scores[METRIC_CER] == 1.0
        assert scores[METRIC_LENIENT_WER] == 1.0
        assert scores[METRIC_WIP] == 0.0

    def test_whitespace_variations_ignored(self) -> None:
        """Test that extra whitespace doesn't affect metrics."""
        scores = calculate_stt_metrics(
            hypothesis="  hello   world  ",
            reference="hello world",
            language_code=None,
        )
        assert scores[METRIC_WER] == 0.0

    def test_all_metric_keys_present(self) -> None:
        """Test that all expected metric keys are in the result."""
        scores = calculate_stt_metrics(
            hypothesis="hello",
            reference="hello",
            language_code=None,
        )
        for metric in METRIC_NAMES:
            assert metric in scores

    def test_scores_are_rounded(self) -> None:
        """Test that all scores are rounded to 2 decimal places."""
        scores = calculate_stt_metrics(
            hypothesis="the cat sat on a mat",
            reference="the cat sat on the mat",
            language_code=None,
        )
        for metric in METRIC_NAMES:
            value = scores[metric]
            assert value == round(value, 2)

    def test_lenient_wer_with_language_code(self) -> None:
        """Test that lenient WER uses language-aware normalization."""
        scores = calculate_stt_metrics(
            hypothesis="hello world",
            reference="hello world",
            language_code="en",
        )
        assert scores[METRIC_LENIENT_WER] == 0.0

    def test_lenient_wer_fallback_when_norm_ref_empty(self) -> None:
        """Test lenient WER falls back to raw WER when normalized reference is empty."""
        with patch(
            "app.services.stt_evaluations.metrics.normalize_text",
            return_value="",
        ):
            scores = calculate_stt_metrics(
                hypothesis="hello",
                reference="hello",
                language_code="en",
            )
            # When norm_ref is empty, lenient_wer should equal raw wer
            assert scores[METRIC_LENIENT_WER] == scores[METRIC_WER]

    def test_wip_between_zero_and_one(self) -> None:
        """Test that WIP is always in [0, 1] range."""
        scores = calculate_stt_metrics(
            hypothesis="the quick brown fox",
            reference="the slow brown fox",
            language_code=None,
        )
        assert 0.0 <= scores[METRIC_WIP] <= 1.0


class TestComputeRunAggregateScores:
    """Test cases for compute_run_aggregate_scores."""

    def test_empty_list(self) -> None:
        result = compute_run_aggregate_scores([])
        assert result == {"summary_scores": []}

    def test_single_result(self) -> None:
        scores = [{"wer": 0.5, "cer": 0.3, "lenient_wer": 0.4, "wip": 0.6}]
        result = compute_run_aggregate_scores(scores)

        summary = result["summary_scores"]
        assert len(summary) == 4

        by_name = {s["name"]: s for s in summary}
        assert by_name["wer"]["avg"] == 0.5
        assert by_name["wer"]["std"] == 0.0
        assert by_name["wer"]["total_pairs"] == 1
        assert by_name["wer"]["data_type"] == SCORE_DATA_TYPE_NUMERIC

    def test_multiple_results(self) -> None:
        scores = [
            {"wer": 0.2, "cer": 0.1, "lenient_wer": 0.15, "wip": 0.8},
            {"wer": 0.4, "cer": 0.3, "lenient_wer": 0.35, "wip": 0.6},
        ]
        result = compute_run_aggregate_scores(scores)

        summary = result["summary_scores"]
        by_name = {s["name"]: s for s in summary}

        assert by_name["wer"]["avg"] == 0.3
        assert by_name["wer"]["total_pairs"] == 2
        assert by_name["cer"]["avg"] == 0.2
        assert by_name["wip"]["avg"] == 0.7

    def test_std_calculation(self) -> None:
        scores = [
            {"wer": 0.0, "cer": 0.0, "lenient_wer": 0.0, "wip": 1.0},
            {"wer": 1.0, "cer": 1.0, "lenient_wer": 1.0, "wip": 0.0},
        ]
        result = compute_run_aggregate_scores(scores)

        by_name = {s["name"]: s for s in result["summary_scores"]}
        assert by_name["wer"]["std"] == 0.5
        assert by_name["wer"]["avg"] == 0.5

    def test_all_metric_names_in_output(self) -> None:
        scores = [{"wer": 0.1, "cer": 0.2, "lenient_wer": 0.15, "wip": 0.9}]
        result = compute_run_aggregate_scores(scores)

        names = [s["name"] for s in result["summary_scores"]]
        for metric in METRIC_NAMES:
            assert metric in names

    def test_missing_metric_key_skipped(self) -> None:
        """Test that a metric missing from all dicts is skipped in output."""
        scores = [{"wer": 0.5, "cer": 0.3}]
        result = compute_run_aggregate_scores(scores)

        names = [s["name"] for s in result["summary_scores"]]
        assert "wer" in names
        assert "cer" in names
        assert "lenient_wer" not in names
        assert "wip" not in names

    def test_values_rounded_to_two_decimals(self) -> None:
        scores = [
            {"wer": 0.123, "cer": 0.456, "lenient_wer": 0.789, "wip": 0.321},
            {"wer": 0.234, "cer": 0.567, "lenient_wer": 0.890, "wip": 0.432},
        ]
        result = compute_run_aggregate_scores(scores)

        for entry in result["summary_scores"]:
            assert entry["avg"] == round(entry["avg"], 2)
            assert entry["std"] == round(entry["std"], 2)

    def test_summary_score_structure(self) -> None:
        """Test that each summary score has the expected keys."""
        scores = [{"wer": 0.5, "cer": 0.3, "lenient_wer": 0.4, "wip": 0.6}]
        result = compute_run_aggregate_scores(scores)

        for entry in result["summary_scores"]:
            assert "name" in entry
            assert "avg" in entry
            assert "std" in entry
            assert "total_pairs" in entry
            assert "data_type" in entry
            assert entry["data_type"] == SCORE_DATA_TYPE_NUMERIC
