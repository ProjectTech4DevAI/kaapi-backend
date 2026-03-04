"""STT evaluation metrics calculation using jiwer, indic-nlp-library, and whisper-normalizer.

Provides WER, CER, lenient WER (with script-aware normalization), and WIP
for comparing STT transcriptions against ground truth.

Normalization strategy:
- indic-nlp-library: hi, bn, gu, pa, kn, ml, or, ta, te, mr (Marathi uses Hindi/Devanagari normalizer)
- whisper-normalizer: as (BengaliNormalizer), en (EnglishTextNormalizer)
- Unsupported languages (e.g., ur): whitespace-only normalization
"""

import logging
import re
from typing import Any, Callable

import jiwer
import numpy as np
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.indic import BengaliNormalizer

logger = logging.getLogger(__name__)

# Metric name constants
METRIC_WER = "wer"
METRIC_CER = "cer"
METRIC_LENIENT_WER = "lenient_wer"
METRIC_WIP = "wip"
METRIC_NAMES = [METRIC_WER, METRIC_CER, METRIC_LENIENT_WER, METRIC_WIP]

# Data type constant for summary scores
SCORE_DATA_TYPE_NUMERIC = "NUMERIC"

# Indic language codes supported by indic-nlp-library normalizer
INDIC_LANGUAGE_CODES = {"hi", "bn", "gu", "pa", "kn", "ml", "or", "ta", "te", "mr"}

# Marathi uses the same Devanagari script as Hindi
INDIC_LANGUAGE_CODE_MAP: dict[str, str] = {"mr": "hi"}

# Singleton factory instance for indic-nlp-library
_normalizer_factory = IndicNormalizerFactory()

# Whisper-normalizer instances for languages not covered by indic-nlp-library
_whisper_normalizers: dict[str, Callable[[str], str]] = {
    "as": BengaliNormalizer(),  # Assamese uses Bengali script
    "en": EnglishTextNormalizer(),
}


def normalize_text(text: str, language_code: str | None) -> str:
    """Normalize text for lenient comparison.

    Uses indic-nlp-library for most Indic languages, whisper-normalizer
    for Assamese and English, and whitespace-only for unsupported languages.

    Args:
        text: Input text to normalize
        language_code: ISO 639-1 language code (e.g., "hi", "bn", "en")

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    normalized = text

    if language_code and language_code in INDIC_LANGUAGE_CODES:
        # Use indic-nlp-library (map mr → hi for Devanagari)
        normalizer_code = INDIC_LANGUAGE_CODE_MAP.get(language_code, language_code)
        try:
            normalizer = _normalizer_factory.get_normalizer(normalizer_code)
            normalized = normalizer.normalize(normalized)
        except Exception as e:
            logger.warning(
                f"[normalize_text] Indic normalization failed | "
                f"language_code: {language_code}, error: {e}"
            )
    elif language_code and language_code in _whisper_normalizers:
        # Use whisper-normalizer for as, en
        try:
            normalized = _whisper_normalizers[language_code](normalized)
        except Exception as e:
            logger.warning(
                f"[normalize_text] Whisper normalization failed | "
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

    # Lenient WER: after script-aware normalization (pass pre-stripped text)
    norm_hyp = normalize_text(hyp, language_code)
    norm_ref = normalize_text(ref, language_code)

    if norm_ref:
        lenient_wer = jiwer.wer(norm_ref, norm_hyp)
    else:
        lenient_wer = wer

    return {
        METRIC_WER: round(wer, 2),
        METRIC_CER: round(cer, 2),
        METRIC_LENIENT_WER: round(lenient_wer, 2),
        METRIC_WIP: round(wip, 2),
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

    summary_scores = []

    for metric in METRIC_NAMES:
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
                "data_type": SCORE_DATA_TYPE_NUMERIC,
            }
        )

    return {"summary_scores": summary_scores}
