"""Record per-request Gemini TTS token usage so cost can be worked out later.

Gemini billing is not visible on a dashboard this account can reach, so the
token counts are captured from the API response itself and stored locally. The
pricing is applied separately — this script only measures.
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tts_benchmarking import (  # noqa: E402
    DEFAULT_INPUT_CSV,
    DEFAULT_TEXT_COLUMN,
    GEMINI_LANGUAGE,
    GEMINI_PROMPT_TEMPLATE,
    GEMINI_VOICE,
    JITTER_MAX_SECONDS,
    JITTER_MIN_SECONDS,
    MAX_WORKERS,
    OUTPUT_DIR,
    gemini_client,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GEMINI_MODELS = [
    "gemini-3.1-flash-tts-preview",
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
]

DEFAULT_ROW_LIMIT = 1_000

# Tokens are billed per modality at different rates, so each modality is kept as
# its own column instead of being folded into the totals.
RESULT_FIELDNAMES = [
    "model",
    "row_index",
    "char_count",
    "input_text_tokens",
    "input_audio_tokens",
    "total_input_tokens",
    "output_text_tokens",
    "output_audio_tokens",
    "total_output_tokens",
    "total_cached_tokens",
    "total_thought_tokens",
    "total_tokens",
    "elapsed_seconds",
    "status",
    "error",
]

SUMMED_FIELDS = [
    "char_count",
    "input_text_tokens",
    "input_audio_tokens",
    "total_input_tokens",
    "output_text_tokens",
    "output_audio_tokens",
    "total_output_tokens",
    "total_cached_tokens",
    "total_thought_tokens",
    "total_tokens",
]


def tokens_by_modality(entries: list[Any] | None, modality: str) -> int:
    if not entries:
        return 0
    return sum(e.tokens or 0 for e in entries if e.modality == modality)


def usage_row(usage: Any) -> dict[str, int]:
    return {
        "input_text_tokens": tokens_by_modality(usage.input_tokens_by_modality, "text"),
        "input_audio_tokens": tokens_by_modality(
            usage.input_tokens_by_modality, "audio"
        ),
        "total_input_tokens": usage.total_input_tokens or 0,
        "output_text_tokens": tokens_by_modality(
            usage.output_tokens_by_modality, "text"
        ),
        "output_audio_tokens": tokens_by_modality(
            usage.output_tokens_by_modality, "audio"
        ),
        "total_output_tokens": usage.total_output_tokens or 0,
        "total_cached_tokens": usage.total_cached_tokens or 0,
        "total_thought_tokens": usage.total_thought_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }


def measure_row(text: str, model: str, row_index: int) -> dict[str, Any]:
    """One synthesis request, kept only for its token counts — audio is discarded."""
    time.sleep(random.uniform(JITTER_MIN_SECONDS, JITTER_MAX_SECONDS))
    record: dict[str, Any] = {
        "model": model,
        "row_index": row_index,
        "char_count": len(text),
        "status": "success",
        "error": "",
    }
    started = time.perf_counter()
    try:
        interaction = gemini_client.interactions.create(
            model=model,
            input=GEMINI_PROMPT_TEMPLATE.format(text=text),
            response_format={"type": "audio"},
            generation_config={
                "speech_config": [{"voice": GEMINI_VOICE, "language": GEMINI_LANGUAGE}],
            },
        )
        record.update(usage_row(interaction.usage))
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        logger.info(
            f"[measure_row] model: {model} | row: {row_index} | "
            f"in: {record['total_input_tokens']} | out: {record['total_output_tokens']} | "
            f"total: {record['total_tokens']}"
        )
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        logger.error(f"[measure_row] model: {model} | row: {row_index} | failed: {exc}")
    return record


def output_paths(input_csv: Path) -> tuple[Path, Path]:
    """Name the outputs after the dataset, so two datasets never overwrite each other."""
    stem = input_csv.stem
    return (
        OUTPUT_DIR / f"{stem}_token_usage.csv",
        OUTPUT_DIR / f"{stem}_token_usage_summary.json",
    )


def read_texts(path: Path, column: str, limit: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        texts = [row[column] for row in csv.DictReader(handle) if row.get(column)]
    return texts[:limit]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in sorted({r["model"] for r in records}):
        done = [r for r in records if r["model"] == model and r["status"] == "success"]
        failed = sum(
            1 for r in records if r["model"] == model and r["status"] != "success"
        )
        totals = {field: sum(r.get(field, 0) for r in done) for field in SUMMED_FIELDS}
        summary[model] = {
            "requests_measured": len(done),
            "requests_failed": failed,
            "totals": totals,
            "mean_per_request": {
                field: round(value / len(done), 1) if done else 0
                for field, value in totals.items()
            },
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument(
        "--models", action="append", help=f"Defaults to: {', '.join(GEMINI_MODELS)}"
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_ROW_LIMIT)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.models or GEMINI_MODELS
    texts = read_texts(args.input, args.column, args.limit)
    logger.info(f"[main] Measuring {len(texts)} rows across {len(models)} models")

    jobs = [
        (text, model, i) for model in models for i, text in enumerate(texts, start=1)
    ]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(measure_row, *job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda r: (r["model"], r["row_index"]))
    result_csv, summary_json = output_paths(args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)

    summary = summarise(records)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for model, stats in summary.items():
        totals = stats["totals"]
        logger.info(
            f"[main] {model} | measured: {stats['requests_measured']} "
            f"(failed {stats['requests_failed']}) | input: {totals['total_input_tokens']} | "
            f"output audio: {totals['output_audio_tokens']} | total: {totals['total_tokens']}"
        )
    logger.info(f"[main] Wrote {result_csv} and {summary_json}")


if __name__ == "__main__":
    main()
