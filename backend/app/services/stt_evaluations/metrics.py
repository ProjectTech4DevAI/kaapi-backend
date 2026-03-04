"""STT evaluation metrics calculation using jiwer and indic-nlp-library.

Provides WER, CER, lenient WER (with Indic normalization), and WIP
for comparing STT transcriptions against ground truth.
"""

import logging
import re
from typing import Any

import jiwer
import numpy as np
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

logger = logging.getLogger(__name__)

# Indic language codes supported by indic-nlp-library normalizer
INDIC_LANGUAGE_CODES = {"hi", "bn", "gu", "pa", "kn", "ml", "or", "ta", "te"}

# Singleton factory instance
_normalizer_factory = IndicNormalizerFactory()


def normalize_text(text: str, language_code: str | None) -> str:
    """Normalize text for lenient comparison.

    Applies Indic script normalization for supported languages,
    then strips extra whitespace for all languages.

    Args:
        text: Input text to normalize
        language_code: ISO 639-1 language code (e.g., "hi", "bn")

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    normalized = text

    # Apply Indic normalization if language is supported
    if language_code and language_code in INDIC_LANGUAGE_CODES:
        try:
            normalizer = _normalizer_factory.get_normalizer(language_code)
            normalized = normalizer.normalize(normalized)
        except Exception as e:
            logger.warning(
                f"[normalize_text] Indic normalization failed | "
                f"language_code: {language_code}, error: {e}"
            )

    # Strip extra whitespace for all languages
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def calculate_stt_metrics(
    hypothesis: str,
    reference: str,
    language_code: str | None,
) -> dict[str, float]:
    """Calculate STT evaluation metrics for a single result.

    Args:
        hypothesis: Generated transcription from STT provider
        reference: Ground truth transcription
        language_code: ISO 639-1 language code for normalization

    Returns:
        Dict with keys: wer, cer, lenient_wer, wip
    """
    # Strip whitespace for raw metrics
    hyp = re.sub(r"\s+", " ", hypothesis).strip()
    ref = re.sub(r"\s+", " ", reference).strip()

    # Handle empty reference edge case
    if not ref:
        return {
            "wer": 0.0 if not hyp else 1.0,
            "cer": 0.0 if not hyp else 1.0,
            "lenient_wer": 0.0 if not hyp else 1.0,
            "wip": 1.0 if not hyp else 0.0,
        }

    # Raw metrics
    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    wip = jiwer.wip(ref, hyp)

    # Lenient WER: after Indic normalization
    norm_hyp = normalize_text(hypothesis, language_code)
    norm_ref = normalize_text(reference, language_code)

    if norm_ref:
        lenient_wer = jiwer.wer(norm_ref, norm_hyp)
    else:
        lenient_wer = wer

    return {
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "lenient_wer": round(lenient_wer, 2),
        "wip": round(wip, 2),
    }


def compute_run_aggregate_scores(
    result_scores: list[dict[str, float]],
) -> dict[str, Any]:
    """Aggregate per-result metrics into run-level summary scores.

    Follows the summary_scores pattern used by text evaluations.

    Args:
        result_scores: List of score dicts from calculate_stt_metrics

    Returns:
        Dict with summary_scores list matching EvaluationRun.score format
    """
    if not result_scores:
        return {"summary_scores": []}

    metric_names = ["wer", "cer", "lenient_wer", "wip"]
    summary_scores = []

    for metric in metric_names:
        values = [s[metric] for s in result_scores if metric in s]
        if not values:
            continue

        arr = np.array(values)
        summary_scores.append(
            {
                "name": metric,
                "avg": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "total_pairs": len(values),
                "data_type": "NUMERIC",
            }
        )

    return {"summary_scores": summary_scores}
